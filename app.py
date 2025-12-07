import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

#Loading model & scaler
model = tf.keras.models.load_model("model2.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #0d1117;
        color: #ffffff;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding-bottom: 10px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 20px;
        color: #ffffff;
        margin-top: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 12px 24px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
    }
    div[class^="stNumberInput"] input, select {
        background-color: #1c1f26;
        color: white;
    }
    .stSelectbox, .stCheckbox {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏦 Loan Default Prediction</div>", unsafe_allow_html=True)
st.write("### <span style='color:white;'>Fill in the details below to get a default probability prediction.</span>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Numeric Inputs
with col1:
    st.markdown("<div class='section-title'>📊 Applicant Info</div>", unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=0, value=50000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

with col2:
    st.markdown("<div class='section-title'>💰 Loan Info</div>", unsafe_allow_html=True)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.3)

with col3:
    st.markdown("<div class='section-title'>📈 Financial Profile</div>", unsafe_allow_html=True)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)

#Dropdowns with one-hot encoding
st.markdown("<div class='section-title'>📘 Background Info</div>", unsafe_allow_html=True)
col4, col5, col6, col7 = st.columns(4)

with col4:
    education_options = ["High School", "Master's", "PhD"]
    education = st.selectbox("Education", education_options)

with col5:
    employment_options = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    employment = st.selectbox("Employment Type", employment_options)

with col6:
    marital_options = ["Married", "Single"]
    marital_status = st.selectbox("Marital Status", marital_options)

with col7:
    loan_purpose_options = ["Business", "Education", "Home", "Other"]
    loan_purpose = st.selectbox("Loan Purpose", loan_purpose_options)

#Binary Inputs
st.markdown("<div class='section-title'>📌 Other Details</div>", unsafe_allow_html=True)
col8, col9, col10 = st.columns(3)

with col8:
    has_mortgage = st.checkbox("Has Mortgage?")

with col9:
    has_dependents = st.checkbox("Has Dependents?")

with col10:
    has_cosigner = st.checkbox("Has Co-Signer?")

#Building input dictionary
input_dict = {
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan_amount],
    "CreditScore": [credit_score],
    "MonthsEmployed": [months_employed],
    "NumCreditLines": [num_credit_lines],
    "InterestRate": [interest_rate],
    "DTIRatio": [dti_ratio],
    # Education
    "Education_High School": [1 if education=="High School" else 0],
    "Education_Master's": [1 if education=="Master's" else 0],
    "Education_PhD": [1 if education=="PhD" else 0],
    # EmploymentType
    "EmploymentType_Part-time": [1 if employment=="Part-time" else 0],
    "EmploymentType_Self-employed": [1 if employment=="Self-employed" else 0],
    "EmploymentType_Unemployed": [1 if employment=="Unemployed" else 0],
    # Marital Status
    "MaritalStatus_Married": [1 if marital_status=="Married" else 0],
    "MaritalStatus_Single": [1 if marital_status=="Single" else 0],
    # Has Mortgage / Dependents / Co-signer
    "HasMortgage_Yes": [1 if has_mortgage else 0],
    "HasDependents_Yes": [1 if has_dependents else 0],
    "LoanPurpose_Business": [1 if loan_purpose=="Business" else 0],
    "LoanPurpose_Education": [1 if loan_purpose=="Education" else 0],
    "LoanPurpose_Home": [1 if loan_purpose=="Home" else 0],
    "LoanPurpose_Other": [1 if loan_purpose=="Other" else 0],
    "HasCoSigner_Yes": [1 if has_cosigner else 0]
}

input_df = pd.DataFrame(input_dict)

#Scaling and Predicting
scaled_input = scaler.transform(input_df)
pred = model.predict(scaled_input)[0][0]

#Displaying prediction as simple text (no box)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<p class='result-text'>Prediction (probability of default): <strong>{round(pred, 3)}</strong></p>", unsafe_allow_html=True)

