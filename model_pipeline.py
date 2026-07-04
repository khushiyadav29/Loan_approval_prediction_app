import os
import pickle
import shutil
import numpy as np
import pandas as pd
from config import MODEL_PATH, BACKUP_MODEL_PATH, FEATURE_COLS

def load_trained_model():
    """
    Loads the trained model from the pickle file.
    Gracefully falls back to the backup model if the main model is missing.
    """
    if not os.path.exists(MODEL_PATH):
        if os.path.exists(BACKUP_MODEL_PATH):
            shutil.copy(BACKUP_MODEL_PATH, MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found. Please ensure '{MODEL_PATH}' exists.")
            
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

def ensure_dataframe(input_data):
    """
    Ensures input_data is a pandas DataFrame with proper column names.
    """
    if isinstance(input_data, pd.DataFrame):
        return input_data
        
    input_array = np.asarray(input_data)
    if len(input_array.shape) == 1:
        input_array = input_array.reshape(1, -1)
        
    return pd.DataFrame(input_array, columns=FEATURE_COLS)

def predict_loan(model, input_data):
    """
    Runs model prediction on the preprocessed input_data.
    Returns:
        prediction: 0 (Rejected) or 1 (Approved)
    """
    df_input = ensure_dataframe(input_data)
    return int(model.predict(df_input)[0])

def get_confidence_score(model, input_data):
    """
    Calculates the confidence score (probability) of the model's prediction.
    If the model supports predict_proba(), returns the probability for the predicted class.
    If the model only supports decision_function(), converts the score via a sigmoid function.
    
    Returns:
        prob_approved: probability of class 1 (Approved)
        prob_rejected: probability of class 0 (Rejected)
        confidence: probability of the predicted class
    """
    df_input = ensure_dataframe(input_data)
    prediction = predict_loan(model, df_input)
    
    # 1. Use predict_proba if available
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(df_input)[0]
            prob_rejected = probas[0]
            prob_approved = probas[1]
            confidence = prob_approved if prediction == 1 else prob_rejected
            return float(prob_approved), float(prob_rejected), float(confidence)
        except Exception:
            pass
            
    # 2. Fallback to decision_function
    if hasattr(model, "decision_function"):
        try:
            decision = float(model.decision_function(df_input)[0])
            # Apply sigmoid to convert to probability of class 1
            prob_approved = 1.0 / (1.0 + np.exp(-decision))
            prob_rejected = 1.0 - prob_approved
            confidence = prob_approved if prediction == 1 else prob_rejected
            return prob_approved, prob_rejected, confidence
        except Exception:
            pass
            
    # 3. Hard fallback if neither is available (e.g. some custom estimators)
    confidence = 1.0
    prob_approved = 1.0 if prediction == 1 else 0.0
    prob_rejected = 0.0 if prediction == 1 else 1.0
    return prob_approved, prob_rejected, confidence

