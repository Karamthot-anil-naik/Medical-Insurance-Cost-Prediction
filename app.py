import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# Load model
model = joblib.load("medical_insurance_model.pkl")

st.title("🏥 Medical Insurance Cost Prediction using Machine Learning")
st.write("Predict medical insurance charges using Machine Learning.")
st.info("""
### 📊 Model Performance

✅ Best Model: Random Forest Regressor

📈 R² Score: **0.865**

🎯 Mean Absolute Error: **2550**

📉 RMSE: **4576**
""")

st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 100, 30)

bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)

children = st.sidebar.slider("Children", 0, 5, 0)

sex = st.sidebar.selectbox("Gender", ["Male", "Female"])

smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])

region = st.sidebar.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

# Convert inputs
sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

sample = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_male": [sex_male],
    "smoker_yes": [smoker_yes],
    "region_northwest": [region_northwest],
    "region_southeast": [region_southeast],
    "region_southwest": [region_southwest]
})
st.subheader("📋 Customer Details")

st.write({
    "Age": age,
    "BMI": bmi,
    "Children": children,
    "Gender": sex,
    "Smoker": smoker,
    "Region": region
})

if st.button("Predict Insurance Cost"):
    prediction = model.predict(sample)[0]

    st.success("💰 Estimated Medical Insurance Cost")

    st.metric(
    label="Predicted Cost",
    value=f"₹ {prediction:,.2f}"
)

st.markdown("---")
st.write("Model Used: Random Forest Regressor")
st.markdown("---")

st.subheader("📁 Dataset Information")

st.write("""
- Dataset Size: **1338 Records**
- Features: **Age, BMI, Children, Gender, Smoker, Region**
- Target: **Insurance Charges**
""")

st.markdown("---")
st.caption("Developed by KARAMTHOT ANIL NAIK")
