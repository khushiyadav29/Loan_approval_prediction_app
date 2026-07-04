import shap
import pandas as pd
import numpy as np
from config import FEATURE_COLS

def get_shap_values(model, X_train, sample_input):
    """
    Computes SHAP values for a single sample input (shape 1xK) using the trained model
    and a background dataset X_train.
    
    Explains the probability of class 1 (Approved).
    
    Returns:
        shap_vals: 1D numpy array of shape (K,) representing feature contributions.
        base_value: expected probability of approval (scalar).
    """
    # Ensure background dataset is a numpy array
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.to_numpy()
    else:
        X_train_np = np.asarray(X_train)
        
    # Ensure sample_input is 2D numpy array
    if isinstance(sample_input, pd.DataFrame):
        sample_np = sample_input.to_numpy()
    else:
        sample_np = np.asarray(sample_input)
    if len(sample_np.shape) == 1:
        sample_np = sample_np.reshape(1, -1)
        
    # Function wrapper that outputs the probability of class 1 (Approved)
    def predict_class1_probability(x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        elif hasattr(model, "decision_function"):
            df = model.decision_function(x)
            return 1.0 / (1.0 + np.exp(-df))
        else:
            # Fallback to binary predictions
            return model.predict(x)

    # Sample background to 50 samples for speed (KernelExplainer is O(N_samples * N_features))
    # 50 samples provides an excellent trade-off between explanation accuracy and instantaneous UI updates.
    background = X_train_np
    if len(background) > 50:
        background = shap.sample(background, 50, random_state=42)

    try:
        # First try the standard model-agnostic Exact / Permutation explainer
        explainer = shap.Explainer(predict_class1_probability, background, seed=42)
        explanation = explainer(sample_np)
        shap_vals = explanation.values[0]
        base_value = explanation.base_values[0]
        
        # If output is multidimensional (sometimes occurs with Explainer object configuration)
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[:, 1]
            
        return shap_vals, base_value
    except Exception:
        try:
            # Fallback to KernelExplainer which is highly robust for any python function
            explainer = shap.KernelExplainer(predict_class1_probability, background)
            shap_vals = explainer.shap_values(sample_np, silent=True)
            
            # Extract values
            if isinstance(shap_vals, list):
                # Class 1 is usually second, but since our custom function returns class 1 probability directly,
                # the output shape will be (1, K)
                shap_vals = shap_vals[0]
            if len(shap_vals.shape) == 2:
                shap_vals = shap_vals[0]
                
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
                
            return shap_vals, base_value
        except Exception:
            # Critical fallback: return all zeros to avoid app crash
            return np.zeros(sample_np.shape[1]), 0.5

def format_shap_report(shap_vals, feature_names, features_dict):
    """
    Sorts and formats features into positive and negative drivers of the prediction.
    """
    column_to_display_map = {
        'Gender': 'Gender',
        'Married': 'Married',
        'Dependents': 'Dependents',
        'Education': 'Education',
        'Self_Employed': 'Self Employed',
        'ApplicantIncome': 'Applicant Income',
        'CoapplicantIncome': 'CoapplicantIncome',
        'LoanAmount': 'Loan Amount',
        'Loan_Amount_Term': 'Loan Term',
        'Credit_History': 'Credit History',
        'Property_Area': 'Property Area'
    }
    
    contributions = []
    for i, name in enumerate(feature_names):
        display_name = column_to_display_map.get(name, name)
        
        # Safe lookup in inputs dictionary, with fallbacks
        val = features_dict.get(display_name)
        if val is None:
            val = features_dict.get(name, "N/A")
            
        contributions.append({
            'feature': display_name,
            'val': val,
            'shap': shap_vals[i]
        })
        
    # Sort by absolute SHAP values (magnitude of impact)
    contributions = sorted(contributions, key=lambda x: abs(x['shap']), reverse=True)
    
    positives = [c for c in contributions if c['shap'] > 0.005]
    negatives = [c for c in contributions if c['shap'] < -0.005]
    
    return positives, negatives

