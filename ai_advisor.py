import os
import google.generativeai as genai

def get_gemini_model_name():
    """
    Dynamically lists available Gemini models and picks the best available one.
    """
    try:
        available_models = [m.name.split('/')[-1] for m in genai.list_models()]
        for model in ['gemini-3.5-flash', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']:
            if model in available_models:
                return model
    except Exception:
        pass
    return 'gemini-2.5-flash'  # Safe default

def get_ai_advisor_advice(prediction_result, inputs, fin_metrics, elig_score, shap_positives, shap_negatives, recs=None):
    """
    Calls the Gemini API to get friendly personal financial advisor guidance.
    Integrates ML predictions, financial analysis, SHAP feature importances, and binary search targets.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ **AI Advisor Offline**: Configure a `GEMINI_API_KEY` in your `.env` file to enable AI explanations."

    try:
        genai.configure(api_key=api_key)
        model_name = get_gemini_model_name()
        model = genai.GenerativeModel(model_name)
        
        status_text = "Approved" if prediction_result == 1 else "Rejected"
        raw_loan_rupees = int(inputs.get('Loan Amount', inputs.get('LoanAmount', 0)) * 1000)
        credit_history_val = inputs.get('Credit History', inputs.get('Credit_History', 1.0))
        credit_history_text = "Good/Active (1.0)" if credit_history_val == 1.0 else "Poor/None (0.0)"
        
        # Build SHAP factor contexts
        pos_factors_text = ""
        for item in shap_positives[:3]:
            pos_factors_text += f"- {item['feature']} (Current value: {item['val']}) helped increase approval chances (contribution value: +{item['shap']:.3f})\n"
            
        neg_factors_text = ""
        for item in shap_negatives[:3]:
            neg_factors_text += f"- {item['feature']} (Current value: {item['val']}) reduced approval chances (contribution value: {item['shap']:.3f})\n"
            
        # Build recommendation context
        recs_context = ""
        if recs and prediction_result == 0:
            recs_context += "Smart Targets Calculated by Model:\n"
            if 'credit_history' in recs:
                recs_context += "- Credit History needs to be improved to Good (1.0).\n"
            if 'applicant_income' in recs:
                app_income_val = inputs.get('Applicant Income', inputs.get('ApplicantIncome', 0))
                recs_context += f"- Applicant Income needs to increase from ₹{app_income_val} to ₹{recs['applicant_income']['target']} (increase of ₹{recs['applicant_income']['increase']}).\n"
            if 'coapplicant_income' in recs:
                coapp_income_val = inputs.get('Coapplicant Income', inputs.get('CoapplicantIncome', 0))
                recs_context += f"- Coapplicant Income needs to increase from ₹{coapp_income_val} to ₹{recs['coapplicant_income']['target']} (increase of ₹{recs['coapplicant_income']['increase']}).\n"
            if 'loan_amount' in recs:
                recs_context += f"- Loan Amount requested needs to reduce from ₹{raw_loan_rupees} to ₹{int(recs['loan_amount']['target'] * 1000)} (reduction of ₹{int(recs['loan_amount']['reduction'] * 1000)}).\n"

        prompt = f"""
You are an expert Personal Financial Advisor and Loan Consultant. Your objective is to explain the credit decision in a user-friendly way and give practical, encouraging guidance to the applicant.
DO NOT sound like a cold compliance officer, risk auditor, or bank employee. Be encouraging, professional, and clear.

Applicant Information:
- Gender: {inputs.get('Gender', 'N/A')}
- Married: {inputs.get('Married', 'N/A')}
- Dependents: {inputs.get('Dependents', 'N/A')}
- Education: {inputs.get('Education', 'N/A')}
- Self Employed: {inputs.get('Self Employed', inputs.get('Self_Employed', 'N/A'))}
- Applicant Income: ₹{inputs.get('Applicant Income', inputs.get('ApplicantIncome', 0))}/month
- Coapplicant Income: ₹{inputs.get('Coapplicant Income', inputs.get('CoapplicantIncome', 0))}/month
- Loan Amount Requested: ₹{raw_loan_rupees}
- Loan Term: {inputs.get('Loan Term', inputs.get('Loan_Amount_Term', 360))} months
- Credit History: {credit_history_text}
- Property Area: {inputs.get('Property Area', inputs.get('Property_Area', 'N/A'))}


Underwriting Metrics Calculated:
- Monthly EMI: ₹{fin_metrics['emi']}
- Debt-to-Income (DTI) Ratio: {fin_metrics['dti']}%
- Loan-to-Income (LTI) Ratio: {fin_metrics['lti']} (Months of income requested)
- Risk Category: {fin_metrics['risk']}
- Credit Eligibility Score: {elig_score}/100

Machine Learning Model Explanation (SHAP Values):
Positive Drivers (things that helped the application):
{pos_factors_text if pos_factors_text else "None significant"}

Negative Drivers (things that held the application back):
{neg_factors_text if neg_factors_text else "None significant"}

Decision: {status_text}

{recs_context}

Negative Constraints (STRICT):
1. Never mention terms like "manual review", "human auditor", "correcting documents", "verify employment", "typos", or "internal bank processes".
2. Only write about factors in the provided data.
3. Keep descriptions concise and direct. The user must be able to read it in under 30 seconds.
4. If approved, congratulate them and highlight their strong factors. If rejected, focus on encouraging improvements with the exact numbers calculated.

Formatting:
Your response must follow this exact format:

### 📊 Understanding Your Underwriting Decision
<A brief 2-3 sentence summary explaining the outcome from a friendly advisor's perspective, referencing the Eligibility Score ({elig_score}/100) and overall risk ({fin_metrics['risk']})>

### ✨ Positive Contributing Factors
• **<Factor Name>** - <friendly explanation of how this helped, referencing the SHAP feedback>
• **<Factor Name>** - <friendly explanation of how this helped, referencing the SHAP feedback>

### ⚠️ Areas for Improvement (Only if Rejected or Medium/High Risk)
• **<Factor Name>** - <friendly explanation of what held you back, referencing the SHAP feedback>
• **<Factor Name>** - <friendly explanation of what held you back, referencing the SHAP feedback>

### 🛠️ Personalized Action Plan (Only if Rejected)
• **<Action Name>** - <actionable step integrating the recommended target values (e.g. increase income to ₹X, reduce loan to ₹Y)>
• **<Action Name>** - <actionable step integrating the recommended target values>
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ **Could not fetch AI advice**: {str(e)}"
