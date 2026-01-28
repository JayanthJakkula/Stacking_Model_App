import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write("This system uses a Stacking Ensemble Machine Learning model to predict whether a loan will be approved by combining multiple ML models.")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Loan_Dataset.csv")   # Kaggle dataset

df = load_data()

# -------------------------------
# DATA CLEANING
# -------------------------------
df.drop(columns=["Loan_ID"], inplace=True)

# Fix Dependents column
df["Dependents"] = df["Dependents"].replace("3+", 3)

# Fill Missing Values
df.fillna({
    "Gender": df["Gender"].mode()[0],
    "Married": df["Married"].mode()[0],
    "Dependents": df["Dependents"].mode()[0],
    "Self_Employed": df["Self_Employed"].mode()[0],
    "LoanAmount": df["LoanAmount"].median(),
    "Loan_Amount_Term": df["Loan_Amount_Term"].median(),
    "Credit_History": df["Credit_History"].mode()[0]
}, inplace=True)

# -------------------------------
# ENCODING
# -------------------------------
le = LabelEncoder()
cat_cols = ["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status"]

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN STACKING MODEL
# -------------------------------
@st.cache_resource
def train_model(X_train, y_train):

    base_models = [
        ("lr", LogisticRegression(max_iter=300)),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier())
    ]

    meta_model = LogisticRegression()

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("Enter Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", value=5000)
co_income = st.sidebar.number_input("Co-Applicant Income", value=0)
loan_amount = st.sidebar.number_input("Loan Amount", value=100)
loan_term = st.sidebar.number_input("Loan Amount Term", value=360)

credit = st.sidebar.radio("Credit History", ["Yes","No"])
credit = 1 if credit=="Yes" else 0

employment = st.sidebar.selectbox("Employment Status",["Salaried","Self-Employed"])
employment = 0 if employment=="Salaried" else 1

property_area = st.sidebar.selectbox("Property Area",["Urban","Semi-Urban","Rural"])
property_map = {"Urban":2,"Semi-Urban":1,"Rural":0}
property_area = property_map[property_area]

# -------------------------------
# MODEL ARCHITECTURE
# -------------------------------
st.subheader("🧩 Model Architecture")
st.info("""
Base Models:
• Logistic Regression  
• Decision Tree  
• Random Forest  

Meta Model:
• Logistic Regression
""")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = np.array([[1,          # Gender
                        1,          # Married
                        0,          # Dependents
                        1,          # Education
                        employment, # Self_Employed
                        app_income,
                        co_income,
                        loan_amount,
                        loan_term,
                        credit,
                        property_area]])


    input_scaled = scaler.transform(input_data)

    result = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    base_preds = model.transform(input_scaled)[0]

    # Base Predictions
    st.subheader("📊 Base Model Predictions")
    st.write("Logistic Regression →", "Approved" if base_preds[0] > 0.5 else "Rejected")
    st.write("Decision Tree →", "Approved" if base_preds[1] > 0.5 else "Rejected")
    st.write("Random Forest →", "Approved" if base_preds[2] > 0.5 else "Rejected")

    # Final Result
    st.subheader("🧠 Final Stacking Decision")

    if result == 1:
        st.success("✅ Loan Approved")
        decision = "approved"
    else:
        st.error("❌ Loan Rejected")
        decision = "rejected"

    st.write(f"📈 Confidence Score: {prob:.2f}%")

    # Business Explanation
    st.subheader("💼 Business Explanation")
    st.write(f"""
Based on income, credit history, and combined predictions from multiple models,
the applicant is likely to {"repay" if result==1 else "not repay"} the loan.
Therefore, the stacking model predicts loan {decision}.
""")
