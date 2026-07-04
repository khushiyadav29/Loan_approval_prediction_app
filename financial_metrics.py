def calculate_financial_metrics(applicant_income, coapplicant_income, loan_amount_model, loan_term_months, annual_rate=9.5):
    """
    Computes standard banking metrics: Total Monthly Income, Monthly EMI, DTI Ratio,
    Loan-to-Income (LTI) Ratio, Credit Risk Category, and an Eligibility Score (0-100).
    
    - loan_amount_model: Loan amount in model scale (thousands of Rupees)
    - loan_term_months: Loan term in months
    """
    total_income = float(applicant_income + coapplicant_income)
    principal = float(loan_amount_model * 1000)
    term_months = max(1.0, float(loan_term_months))
    monthly_rate = float(annual_rate) / 12.0 / 100.0

    # Calculate EMI
    if principal <= 0:
        emi = 0.0
    elif monthly_rate > 0:
        try:
            pow_val = (1.0 + monthly_rate) ** term_months
            emi = principal * monthly_rate * pow_val / (pow_val - 1.0)
        except (ZeroDivisionError, OverflowError):
            emi = principal / term_months
    else:
        emi = principal / term_months
        
    # Calculate Debt-to-Income (DTI) ratio
    if total_income > 0:
        dti = (emi / total_income) * 100.0
    else:
        dti = 100.0 if emi > 0 else 0.0

    # Calculate Loan-to-Income (LTI) ratio (monthly principal / total income)
    # Or more standard: Principal / Total Monthly Income (how many months of income does the loan represent)
    if total_income > 0:
        lti = principal / total_income
    else:
        lti = 999.0 if principal > 0 else 0.0

    # Calculate Credit Risk rating based on DTI and LTI thresholds
    # Low: DTI <= 35% and LTI <= 36 (3 years of income)
    # Medium: DTI <= 50% and LTI <= 60 (5 years of income)
    # High: otherwise
    if dti <= 35.0 and lti <= 36.0:
        risk = "Low"
    elif dti <= 50.0 and lti <= 60.0:
        risk = "Medium"
    else:
        risk = "High"

    return {
        'total_income': total_income,
        'emi': round(emi, 2),
        'dti': round(dti, 2),
        'lti': round(lti, 2),
        'risk': risk
    }


def calculate_eligibility_score(credit_history, dti, lti, self_employed_val, education_val):
    """
    Computes a transparent credit underwriting score from 0 to 100:
    1. Credit History (40 pts)
    2. Debt-to-Income Ratio (25 pts)
    3. Loan-to-Income Ratio (15 pts)
    4. Employment Type Stability (10 pts)
    5. Education Level (10 pts)
    """
    score = 0
    
    # 1. Credit History (40 pts)
    if credit_history == 1.0:
        score += 40
        
    # 2. Debt-to-Income (25 pts)
    if dti <= 20.0:
        score += 25
    elif dti <= 35.0:
        score += 20
    elif dti <= 50.0:
        score += 10
        
    # 3. Loan-to-Income (15 pts)
    if lti <= 12.0:
        score += 15
    elif lti <= 36.0:
        score += 12
    elif lti <= 60.0:
        score += 8
    elif lti <= 84.0:
        score += 4

    # 4. Employment Stability (10 pts)
    # Salaried (Self_Employed = 0) gets full stability points
    if self_employed_val == 0:
        score += 10
    else:
        score += 5

    # 5. Education Level (10 pts)
    if education_val == 1:
        score += 10
    else:
        score += 5
        
    return score
