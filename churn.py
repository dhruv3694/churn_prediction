import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\Baijnath\Downloads\archive_Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    if column != 'Churn':
        df[column] = le.fit_transform(df[column])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("Customer Churn Prediction App")
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

gender = st.sidebar.selectbox("Gender", df['gender'].unique())
senior = st.sidebar.selectbox("Senior Citizen", ['Yes', 'No'])
partner = st.sidebar.selectbox("Partner", df['Partner'].unique())
dependents = st.sidebar.selectbox("Dependents", df['Dependents'].unique())
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", df['PhoneService'].unique())
multiple_lines = st.sidebar.selectbox("Multiple Lines", df['MultipleLines'].unique())
internet_service = st.sidebar.selectbox("Internet Service", df['InternetService'].unique())
online_security = st.sidebar.selectbox("Online Security", df['OnlineSecurity'].unique())
online_backup = st.sidebar.selectbox("Online Backup", df['OnlineBackup'].unique())
device_protection = st.sidebar.selectbox("Device Protection", df['DeviceProtection'].unique())
tech_support = st.sidebar.selectbox("Tech Support", df['TechSupport'].unique())
streaming_tv = st.sidebar.selectbox("Streaming TV", df['StreamingTV'].unique())
streaming_movies = st.sidebar.selectbox("Streaming Movies", df['StreamingMovies'].unique())
contract = st.sidebar.selectbox("Contract", df['Contract'].unique())
paperless_billing = st.sidebar.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
payment_method = st.sidebar.selectbox("Payment Method", df['PaymentMethod'].unique())
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 0.0, 9000.0, 2500.0)

input_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior == 'Yes' else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

for col in input_df.columns:
    if input_df[col].dtype == 'object':
        input_df[col] = le.fit(df[col]).transform(input_df[col])

prediction = model.predict(input_df)

if prediction[0] == 1:
    st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")
