import streamlit as st
import numpy as np
import pandas as pd
import os
import altair as alt
from dotenv import load_dotenv

# Load env variables (for GEMINI_API_KEY)
load_dotenv()

# Modular imports
from config import FEATURE_COLS, MAPPINGS, DATASET_PATH, clean_dependents
from model_pipeline import load_trained_model, predict_loan, get_confidence_score
from financial_metrics import calculate_financial_metrics, calculate_eligibility_score
from recommendations import generate_recommendations
from shap_explainer import get_shap_values, format_shap_report
from ai_advisor import get_ai_advisor_advice

# ====================================================
# BACKGROUND DATA CACHING (FOR SHAP BACKGROUND)
# ====================================================
@st.cache_data
def load_background_data():
    """
    Loads and preprocesses background data for the SHAP explainer.
    Caches the results to prevent loading file on every rerun.
    """
    from train_models import preprocess_data
    if os.path.exists(DATASET_PATH):
        df_raw = pd.read_csv(DATASET_PATH)
        df_processed = preprocess_data(df_raw)
        return df_processed[FEATURE_COLS]
    else:
        # Fallback empty dataframe matching shape
        return pd.DataFrame(columns=FEATURE_COLS)

# ====================================================
# STYLING & CUSTOM CARD RENDERERS
# ====================================================
def render_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Dashboard Container Card */
        .dashboard-card {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Mini Banking Stat Box */
        .stat-box {
            background: #0f172a;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #1e293b;
            text-align: center;
        }
        .stat-value {
            font-size: 22px;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 4px;
            font-family: monospace;
        }
        .stat-label {
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        
        /* Status Badges */
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .badge-approved {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid #10b981;
        }
        .badge-rejected {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid #ef4444;
        }
        .badge-risk-low {
            background-color: rgba(16, 185, 129, 0.1);
            color: #10b981;
        }
        .badge-risk-medium {
            background-color: rgba(245, 158, 11, 0.1);
            color: #f59e0b;
        }
        .badge-risk-high {
            background-color: rgba(239, 68, 68, 0.1);
            color: #ef4444;
        }
        
        /* Modern Header Banner */
        .banking-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-bottom: 3px solid #38bdf8;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .banner-title {
            color: #38bdf8;
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .banner-subtitle {
            color: #94a3b8;
            margin: 4px 0 0 0;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

# ====================================================
# MAIN STREAMLIT APP
# ====================================================
def main():
    # Page setup
    st.set_page_config(
        page_title="Loan Approval Prediction System",
        page_icon="🏦",
        layout="wide"
    )
    
    # Inject Custom Banking CSS
    render_custom_css()

    # Load Model Pipeline
    try:
        model = load_trained_model()
    except Exception as e:
        st.error(f"🛑 **Critical System Error**: Machine learning models could not be loaded. Please ensure a valid model is trained and saved at `loan_model_pickle.sav`.\nDetails: {e}")
        return

    # Header Banner
    st.markdown("""
    <div class="banking-banner">
        <div>
            <h1 class="banner-title">🏦 Loan Approval Prediction System</h1>
            <p class="banner-subtitle">Enterprise Risk Intelligence & Explainable Credit Analytics Dashboard</p>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 12px; color: #64748b; font-weight: 600;">PLATFORM VERSION</span><br/>
            <span style="background-color: #38bdf8; color: #0f172a; font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: 700;">V2.1.0-AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Columns Layout
    col_inputs, col_results = st.columns([1.1, 1.4], gap="large")

    with col_inputs:
        st.markdown("### 📋 Application Data Intake")
        st.write("Complete the credit application fields below to evaluate creditworthiness.")
        
        # Intake tabs
        tab_profile, tab_income, tab_loan = st.tabs([
            "🏦 1. Applicant Profile", 
            "💰 2. Financial Standing", 
            "📄 3. Loan Requirements"
        ])
        
        with tab_profile:
            st.markdown("##### 👤 Demographic & Employment Profile")
            gender = st.selectbox("Applicant Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Employment Structure", ["No", "Yes"], help="Is the applicant self-employed?")
            
        with tab_income:
            st.markdown("##### 💵 Monthly Cashflow & Repayment Capacity")
            applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=45000, step=1000)
            coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, value=0, step=1000)
            credit_history = st.selectbox(
                "Credit Bureau Status", 
                [1.0, 0.0], 
                format_func=lambda x: "Good Credit History (1.0)" if x == 1.0 else "Poor / No Credit History (0.0)",
                help="Active credit rating report value."
            )
            
        with tab_loan:
            st.markdown("##### 📝 Requested Credit Terms")
            loan_amount_input = st.number_input("Requested Principal Amount (₹)", min_value=10000, value=150000, step=5000)
            loan_term_months = st.number_input("Amortization Period (Months)", min_value=6, value=360, step=12)
            property_area = st.selectbox("Collateral Location (Property Area)", ["Rural", "Semiurban", "Urban"])

        # Clean categories & map features
        dependents_clean = clean_dependents(dependents)
        gender_val = MAPPINGS['Gender'][gender]
        married_val = MAPPINGS['Married'][married]
        education_val = MAPPINGS['Education'][education]
        self_employed_val = MAPPINGS['Self_Employed'][self_employed]
        property_val = MAPPINGS['Property_Area'][property_area]
        
        # Scaling loan amount to model format (K of Rupees)
        loan_amount_model = loan_amount_input / 1000.0
        
        # Create input features list matching columns exactly
        raw_features = [
            gender_val, married_val, dependents_clean, education_val, self_employed_val,
            applicant_income, coapplicant_income, loan_amount_model,
            loan_term_months, credit_history, property_val
        ]
        
        inputs_dict = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self Employed': self_employed,
            'Applicant Income': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'Loan Amount': loan_amount_model,
            'Loan Term': loan_term_months,
            'Credit History': credit_history,
            'Property Area': property_area
        }
        
        st.write("")
        evaluate_btn = st.button("🚀 Process Credit Evaluation", use_container_width=True)
        
        if evaluate_btn:
            # Predict and store in session state
            pred = predict_loan(model, raw_features)
            
            st.session_state.prediction_made = True
            st.session_state.main_prediction = pred
            st.session_state.main_features = raw_features
            st.session_state.main_inputs = inputs_dict
            
            # Clear old cached generative responses
            if 'cached_ai_response' in st.session_state:
                del st.session_state.cached_ai_response

    # Underwriting Output Dashboard
    with col_results:
        st.markdown("### 📊 Underwriting Analytics Dashboard")
        
        if not st.session_state.get('prediction_made', False):
            # Placeholder State
            st.markdown("""
            <div style="border: 2px dashed #475569; border-radius: 12px; padding: 60px; text-align: center; color: #64748b; margin-top: 10px;">
                <span style="font-size: 50px;">💳</span>
                <h4 style="margin: 15px 0 5px 0; color: #94a3b8; font-weight: 600;">System Idle & Ready</h4>
                <p style="margin: 0; font-size: 13.5px;">Click <strong>Process Credit Evaluation</strong> on the left to initiate underwriting calculations.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            main_pred = st.session_state.main_prediction
            main_features = st.session_state.main_features
            main_inputs = st.session_state.main_inputs
            
            # 1. Financial Metrics Pre-Calculation
            fin_metrics = calculate_financial_metrics(
                main_inputs['Applicant Income'],
                main_inputs['CoapplicantIncome'],
                main_inputs['Loan Amount'],
                main_inputs['Loan Term']
            )
            
            elig_score = calculate_eligibility_score(
                main_inputs['Credit History'],
                fin_metrics['dti'],
                fin_metrics['lti'],
                self_employed_val,
                education_val
            )
            
            prob_app, prob_rej, confidence = get_confidence_score(model, main_features)
            
            # 2. RENDER CARD: Loan Status & Badges
            status_text = "Approved" if main_pred == 1 else "Rejected"
            status_badge = '<span class="badge badge-approved">Approved</span>' if main_pred == 1 else '<span class="badge badge-rejected">Rejected</span>'
            
            risk_class = "badge-risk-low" if fin_metrics['risk'] == "Low" else ("badge-risk-medium" if fin_metrics['risk'] == "Medium" else "badge-risk-high")
            risk_badge = f'<span class="badge {risk_class}">{fin_metrics["risk"]} Risk</span>'
            
            st.markdown(f"""
            <div class="dashboard-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h4 style="margin: 0; font-weight: 700; font-size: 16px; color: #f8fafc;">📊 LOAN EVALUATION VERDICT</h4>
                    <div>
                        {status_badge}
                        {risk_badge}
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div style="background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; text-align: center;">
                        <span style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Eligibility Score</span>
                        <div style="font-size: 28px; font-weight: 800; color: #38bdf8; margin-top: 4px;">{elig_score} <span style="font-size: 13px; color: #475569; font-weight: 500;">/ 100</span></div>
                    </div>
                    <div style="background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; text-align: center;">
                        <span style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Model Confidence</span>
                        <div style="font-size: 28px; font-weight: 800; color: #38bdf8; margin-top: 4px;">{confidence:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. RENDER SECTION: Financial Analysis
            st.markdown("#### 📈 Financial Analysis")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">₹{int(fin_metrics['total_income']):,}</div>
                    <div class="stat-label">Total Monthly</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">₹{int(fin_metrics['emi']):,}</div>
                    <div class="stat-label">Monthly EMI</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{fin_metrics['dti']}%</div>
                    <div class="stat-label">DTI Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col4:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{fin_metrics['lti']}x</div>
                    <div class="stat-label">LTI Ratio</div>
                </div>
                """, unsafe_allow_html=True)

            # 4. RENDER SECTION: Recommendations
            recs = generate_recommendations(model, main_features)
            
            st.markdown("#### 💡 Smart Target Recommendations")
            if main_pred == 1:
                st.success("🎉 **Approved**: Applicant profile meets all criteria. No adjustments needed.")
            else:
                if recs:
                    for rec_key, val in recs.items():
                        if rec_key == 'applicant_income':
                            st.info(f"📈 **Increase Income**: Monthly income needs to reach **₹{val['target']:,}** (an increase of **₹{val['increase']:,}**)")
                        elif rec_key == 'coapplicant_income':
                            st.info(f"👥 **Co-applicant Support**: Adding a co-applicant with monthly income of **₹{val['target']:,}** will qualify this application.")
                        elif rec_key == 'loan_amount':
                            st.info(f"📉 **Reduce Loan Principal**: Reduce the loan request to **₹{int(val['target']*1000):,}** (a decrease of **₹{int(val['reduction']*1000):,}**)")
                        elif rec_key == 'credit_history':
                            st.warning("💳 **Credit Rating**: Resolve bureau issues to establish a **Good Credit History (1.0)** score.")
                else:
                    st.warning("⚠️ **Highly Constrained Profile**: A single factor adjustment cannot qualify this loan. Multi-factor correction required (e.g. increase income while simultaneously reducing loan amount).")

            # 5. RENDER SECTION: Explainable AI (SHAP)
            st.markdown("#### 🔎 Explainable AI: Feature Importance")
            background_data = load_background_data()
            if not background_data.empty:
                with st.spinner("Computing Shapley feature values..."):
                    shap_vals, base_value = get_shap_values(model, background_data, main_features)
                    
                # Format positive and negative factors
                shap_pos, shap_neg = format_shap_report(shap_vals, FEATURE_COLS, main_inputs)
                
                # Plot SHAP Chart
                df_chart = pd.DataFrame({
                    'Feature': [
                        'Credit History' if f == 'Credit_History' else
                        'Applicant Income' if f == 'ApplicantIncome' else
                        'Co-applicant Income' if f == 'CoapplicantIncome' else
                        'Loan Amount' if f == 'LoanAmount' else
                        'Loan Term' if f == 'Loan_Amount_Term' else
                        'Property Area' if f == 'Property_Area' else
                        'Self Employed' if f == 'Self_Employed' else f
                        for f in FEATURE_COLS
                    ],
                    'Contribution': shap_vals,
                    'Impact': ['Positive (Helps Approval)' if v > 0 else 'Negative (Reduces Approval)' for v in shap_vals]
                })
                df_chart = df_chart.reindex(df_chart['Contribution'].abs().sort_values(ascending=False).index)
                
                shap_chart = alt.Chart(df_chart).mark_bar().encode(
                    y=alt.Y('Feature:N', sort=None, title='Feature'),
                    x=alt.X('Contribution:Q', title='Contribution to Approval Chance'),
                    color=alt.Color('Impact:N', scale=alt.Scale(domain=['Positive (Helps Approval)', 'Negative (Reduces Approval)'], range=['#10b981', '#ef4444']), title='Impact Type'),
                    tooltip=['Feature', 'Contribution', 'Impact']
                ).properties(height=300)
                
                st.altair_chart(shap_chart, use_container_width=True)
            else:
                shap_vals, base_value = np.zeros(len(FEATURE_COLS)), 0.5
                shap_pos, shap_neg = [], []
                st.info("SHAP Background dataset is not available. Please ensure `loan_dataset.csv` exists to load feature explanations.")

            # 6. RENDER SECTION: AI Loan Advisor (narrative)
            st.markdown("#### 🤖 AI Loan Advisor")
            if 'cached_ai_response' not in st.session_state:
                with st.spinner("🤖 Consulting Personal Financial Advisor..."):
                    st.session_state.cached_ai_response = get_ai_advisor_advice(
                        main_pred, main_inputs, fin_metrics, elig_score, shap_pos, shap_neg, recs
                    )
            
            st.markdown(st.session_state.cached_ai_response)

            # 7. RENDER SECTION: What-If Simulator
            st.markdown("#### ⚙️ Real-Time What-If Simulator")
            st.write("Dynamically test how profile adjustments change underwriting decisions instantly.")
            
            sim_init_loan = int(main_inputs['Loan Amount'] * 1000)
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                sim_app_income = st.slider("Simulated Applicant Income (₹)", min_value=0, max_value=max(150000, int(main_inputs['Applicant Income']) * 2), value=int(main_inputs['Applicant Income']), step=1000)
                sim_coapp_income = st.slider("Simulated Co-applicant Income (₹)", min_value=0, max_value=max(100000, int(main_inputs['CoapplicantIncome']) * 2), value=int(main_inputs['CoapplicantIncome']), step=1000)
            with s_col2:
                sim_loan_amount = st.slider("Simulated Loan Principal (₹)", min_value=10000, max_value=max(500000, sim_init_loan * 2), value=sim_init_loan, step=5000)
                sim_credit = st.selectbox(
                    "Simulated Credit History", 
                    [1.0, 0.0],
                    index=0 if main_inputs['Credit History'] == 1.0 else 1,
                    format_func=lambda x: "Good Credit History (1.0)" if x == 1.0 else "Poor / No Credit History (0.0)",
                    key="sim_credit_sel"
                )

            # Reconstruct simulated input
            sim_loan_model = sim_loan_amount / 1000.0
            sim_features = [
                gender_val, married_val, dependents_clean, education_val, self_employed_val,
                sim_app_income, sim_coapp_income, sim_loan_model,
                main_inputs['Loan Term'], sim_credit, property_val
            ]
            
            sim_pred = predict_loan(model, sim_features)
            sim_prob_app, sim_prob_rej, sim_conf = get_confidence_score(model, sim_features)
            
            prev_status = "Approved" if main_pred == 1 else "Rejected"
            sim_status = "Approved" if sim_pred == 1 else "Rejected"
            
            # Displays side by side comparison
            comp_c1, comp_arrow, comp_c2 = st.columns([2, 1, 2])
            with comp_c1:
                st.markdown(f"""
                <div class="stat-box" style="border: 1px solid #475569;">
                    <div style="font-size: 10px; color: #64748b; font-weight: 700; text-transform: uppercase;">Previous Prediction</div>
                    <div style="font-size: 20px; font-weight: 700; color: {'#10b981' if main_pred == 1 else '#ef4444'}; margin-top: 4px;">{prev_status}</div>
                    <div style="font-size: 11px; color: #475569; margin-top: 2px;">Conf: {confidence:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
            with comp_arrow:
                st.markdown("""
                <div style="text-align: center; padding-top: 15px; font-size: 26px; color: #475569; font-weight: 700;">➔</div>
                """, unsafe_allow_html=True)
            with comp_c2:
                sim_border_color = '#10b981' if sim_pred == 1 else '#ef4444'
                st.markdown(f"""
                <div class="stat-box" style="border: 1.5px dashed {sim_border_color};">
                    <div style="font-size: 10px; color: #64748b; font-weight: 700; text-transform: uppercase;">Simulated Prediction</div>
                    <div style="font-size: 20px; font-weight: 700; color: {sim_border_color}; margin-top: 4px;">{sim_status}</div>
                    <div style="font-size: 11px; color: #475569; margin-top: 2px;">Conf: {sim_conf:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Dynamic delta explanation message
            if main_pred == 0 and sim_pred == 1:
                deltas = []
                
                # Check Applicant Income
                delta_app = sim_app_income - main_inputs['Applicant Income']
                if delta_app > 0:
                    deltas.append(f"increasing monthly income by ₹{delta_app:,}")
                    
                # Check Co-applicant Income
                delta_coapp = sim_coapp_income - main_inputs['CoapplicantIncome']
                if delta_coapp > 0:
                    deltas.append(f"increasing co-applicant income by ₹{delta_coapp:,}")
                    
                # Check Loan Amount
                delta_loan = sim_init_loan - sim_loan_amount
                if delta_loan > 0:
                    deltas.append(f"reducing loan principal requested by ₹{delta_loan:,}")
                    
                # Check Credit History
                if sim_credit == 1.0 and main_inputs['Credit History'] == 0.0:
                    deltas.append("re-establishing a good credit bureau history (1.0)")
                    
                if deltas:
                    explanation_text = ", ".join(deltas[:-1]) + (" and " + deltas[-1] if len(deltas) > 1 else deltas[0])
                    st.success(f"💡 **Simulator Analytics**: Your credit score flips to **Approved** by successfully {explanation_text}.")
            elif main_pred == 1 and sim_pred == 0:
                st.error("⚠️ **Simulator Analytics**: This simulated adjustment degrades the applicant profile metrics, causing the loan request to be **Rejected**.")

if __name__ == '__main__':
    main()
