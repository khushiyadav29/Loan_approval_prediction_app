import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import DATASET_PATH, FEATURE_COLS, MAPPINGS, clean_dependents, MODEL_PATH

def preprocess_data(df):
    """
    Cleans and preprocesses the dataset according to standard banking requirements.
    """
    # 1. Drop rows with missing values
    df_cleaned = df.dropna().copy()
    
    # 2. Map target variable Loan_Status
    df_cleaned['Loan_Status'] = df_cleaned['Loan_Status'].replace({'N': 0, 'Y': 1})
    
    # 3. Clean Dependents
    df_cleaned['Dependents'] = df_cleaned['Dependents'].apply(clean_dependents)
    
    # 4. Map categorical variables using config mappings
    for col, mapping in MAPPINGS.items():
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].replace(mapping)
            
    # Convert numerical columns explicitly
    df_cleaned['ApplicantIncome'] = pd.to_numeric(df_cleaned['ApplicantIncome'])
    df_cleaned['CoapplicantIncome'] = pd.to_numeric(df_cleaned['CoapplicantIncome'])
    df_cleaned['LoanAmount'] = pd.to_numeric(df_cleaned['LoanAmount'])
    df_cleaned['Loan_Amount_Term'] = pd.to_numeric(df_cleaned['Loan_Amount_Term'])
    df_cleaned['Credit_History'] = pd.to_numeric(df_cleaned['Credit_History'])
    
    return df_cleaned

def main():
    print("==================================================")
    print("🏦 LOAN APPLICATION CREDIT MODEL TRAINING PIPELINE")
    print("==================================================")
    
    # Load dataset
    print(f"Loading dataset from: {DATASET_PATH}...")
    df_raw = pd.read_csv(DATASET_PATH)
    
    # Preprocess
    print("Preprocessing and encoding data...")
    df_processed = preprocess_data(df_raw)
    
    X = df_processed[FEATURE_COLS]
    y = df_processed['Loan_Status']
    
    print(f"Total samples after preprocessing: {len(X)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define candidate models
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42
        )
        print("XGBoost is available and added to the candidate list.")
    else:
        print("XGBoost is not available, skipping.")

    best_model_name = None
    best_accuracy = 0.0
    best_pipeline = None
    results = []

    print("\nTraining and evaluating candidate models...")
    for name, clf in classifiers.items():
        # Build pipeline (with scaling)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Confusion Matrix': cm,
            'Pipeline': pipeline
        })
        
        print(f"\nModel: {name}")
        print(f"  • Accuracy:  {acc:.4f}")
        print(f"  • Precision: {prec:.4f}")
        print(f"  • Recall:    {rec:.4f}")
        print(f"  • F1-Score:  {f1:.4f}")
        print(f"  • Confusion Matrix:\n{cm}")
        
        # Keep track of the best model based on validation accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_pipeline = pipeline

    print("\n==================================================")
    print("📊 MODEL COMPARISON SUMMARY")
    print("==================================================")
    summary_df = pd.DataFrame(results).drop(columns=['Pipeline', 'Confusion Matrix'])
    print(summary_df.to_string(index=False))
    
    print("\n==================================================")
    print(f"🏆 BEST MODEL SELECTED: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print("==================================================")
    
    # Save the best model pipeline
    print(f"Saving best pipeline to '{MODEL_PATH}'...")
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_pipeline, file)
        
    print("Model pipeline saved successfully!")

if __name__ == '__main__':
    main()
