import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

#Loading model and scaler
model = tf.keras.models.load_model("model2.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

#Page Config
st.set_page_config(
    page_title="🏦 Loan Default Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Header
st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🏦 Loan Default Prediction</h1>", unsafe_allow_html=True)
st.write("Fill in the borrower details below to predict the probability of loan default.")

#Layout Columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000, step=500)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)

with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0, step=0.1, format="%.2f")
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.3, step=0.01, format="%.2f")
    
    education_options = ["High School", "Master's", "PhD"]
    education = st.selectbox("Education", education_options)

    employment_options = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    employment = st.selectbox("Employment Type", employment_options)

    marital_options = ["Married", "Single"]
    marital_status = st.selectbox("Marital Status", marital_options)

    loan_purpose_options = ["Business", "Education", "Home", "Other"]
    loan_purpose = st.selectbox("Loan Purpose", loan_purpose_options)

#Binary Options
st.markdown("### Additional Details")
has_mortgage = st.checkbox("Has Mortgage?")
has_dependents = st.checkbox("Has Dependents?")
has_cosigner = st.checkbox("Has Co-Signer?")

#Build Input Dictionary
input_dict = {
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed],
    'NumCreditLines': [num_credit_lines],
    'InterestRate': [interest_rate],
    'DTIRatio': [dti_ratio],
    'Education_High School': [1 if education == "High School" else 0],
    "Education_Master's": [1 if education == "Master's" else 0],
    'Education_PhD': [1 if education == "PhD" else 0],
    'EmploymentType_Part-time': [1 if employment == "Part-time" else 0],
    'EmploymentType_Self-employed': [1 if employment == "Self-employed" else 0],
    'EmploymentType_Unemployed': [1 if employment == "Unemployed" else 0],
    'MaritalStatus_Married': [1 if marital_status == "Married" else 0],
    'MaritalStatus_Single': [1 if marital_status == "Single" else 0],
    'HasMortgage_Yes': [1 if has_mortgage else 0],
    'HasDependents_Yes': [1 if has_dependents else 0],
    'LoanPurpose_Business': [1 if loan_purpose == "Business" else 0],
    'LoanPurpose_Education': [1 if loan_purpose == "Education" else 0],
    'LoanPurpose_Home': [1 if loan_purpose == "Home" else 0],
    'LoanPurpose_Other': [1 if loan_purpose == "Other" else 0],
    'HasCoSigner_Yes': [1 if has_cosigner else 0]
}

input_df = pd.DataFrame(input_dict)

#Prediction
if st.button("Predict Default Probability"):
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0][0]
    st.success(f"📊 Probability of Loan Default: **{pred:.3f}**")

