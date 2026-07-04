import os

# Paths
MODEL_PATH = 'loan_model_pickle.sav'
BACKUP_MODEL_PATH = 'loan_model_pickle (2).sav'
DATASET_PATH = 'loan_dataset.csv'

# Features configuration
FEATURE_COLS = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area'
]

# Value mappings for model compatibility
MAPPINGS = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
}

def clean_dependents(val):
    """
    Cleans Dependents value. Standardises '3+' to 3.
    """
    if val is None:
        return 0
    val_str = str(val).strip()
    if val_str == '3+':
        return 3
    try:
        # Convert to float first, then int, to handle cases like '2.0'
        return int(float(val_str))
    except (ValueError, TypeError):
        return 0
