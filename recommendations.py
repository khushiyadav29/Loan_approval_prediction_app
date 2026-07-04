import numpy as np

def recommend_applicant_income(model, base_features, current_income):
    """
    Searches for the minimum applicant income required to flip the prediction to Approved (1).
    Returns the target income (int) or None if not possible.
    """
    test_features = list(base_features)
    test_features[5] = current_income
    
    # If already approved, no recommendation needed
    if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
        return None
        
    # Check if a very high income flips it to Approved
    max_test = current_income + 300000
    test_features[5] = max_test
    if model.predict(np.array(test_features).reshape(1, -1))[0] != 1:
        return None  # Even extremely high income doesn't approve it alone
        
    # Binary search to find the minimum income
    low = current_income
    high = max_test
    best_income = high
    while high - low > 100:  # Search with precision of ₹100
        mid = (low + high) // 2
        test_features[5] = mid
        if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
            best_income = mid
            high = mid
        else:
            low = mid
            
    return int(best_income)

def recommend_coapplicant_income(model, base_features, current_coincome):
    """
    Searches for the coapplicant income adjustment required for approval.
    """
    test_features = list(base_features)
    test_features[6] = current_coincome
    
    if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
        return None
        
    # Check if increasing coapplicant income helps
    max_test = current_coincome + 300000
    test_features[6] = max_test
    if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
        low = current_coincome
        high = max_test
        best_coincome = high
        while high - low > 100:
            mid = (low + high) // 2
            test_features[6] = mid
            if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
                best_coincome = mid
                high = mid
            else:
                low = mid
        return int(best_coincome)
        
    return None

def recommend_loan_amount(model, base_features, current_loan_thousands):
    """
    Searches for the maximum loan amount that would allow approval (i.e. reducing loan amount).
    """
    test_features = list(base_features)
    test_features[7] = current_loan_thousands
    
    if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
        return None
        
    # Check if reducing loan amount to 10k (or similar small amount) helps
    test_features[7] = 10.0  # 10 thousand Rupees
    if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
        low = 10.0
        high = current_loan_thousands
        best_loan = 10.0
        while high - low > 1.0:  # Search with precision of 1 (₹1000)
            mid = (low + high) / 2
            test_features[7] = mid
            if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
                best_loan = mid
                low = mid
            else:
                high = mid
        return int(best_loan)
        
    return None

def recommend_credit_history(model, base_features, current_credit):
    """
    Checks if simply improving the credit history to 1.0 (Good) enables approval.
    """
    test_features = list(base_features)
    test_features[9] = current_credit
    
    if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
        return None
        
    if current_credit == 0.0:
        test_features[9] = 1.0
        if model.predict(np.array(test_features).reshape(1, -1))[0] == 1:
            return 1.0
            
    return None

def generate_recommendations(model, base_features):
    """
    Generates all targeted recommendations for a rejected loan application.
    """
    current_income = base_features[5]
    current_coincome = base_features[6]
    current_loan = base_features[7]
    current_credit = base_features[9]
    
    target_income = recommend_applicant_income(model, base_features, current_income)
    target_coincome = recommend_coapplicant_income(model, base_features, current_coincome)
    target_loan = recommend_loan_amount(model, base_features, current_loan)
    target_credit = recommend_credit_history(model, base_features, current_credit)
    
    recs = {}
    if target_income is not None:
        recs['applicant_income'] = {
            'current': current_income,
            'target': target_income,
            'increase': target_income - current_income
        }
    if target_coincome is not None:
        recs['coapplicant_income'] = {
            'current': current_coincome,
            'target': target_coincome,
            'increase': target_coincome - current_coincome
        }
    if target_loan is not None:
        recs['loan_amount'] = {
            'current': current_loan,
            'target': target_loan,
            'reduction': current_loan - target_loan
        }
    if target_credit is not None:
        recs['credit_history'] = {
            'current': current_credit,
            'target': 1.0
        }
        
    return recs
