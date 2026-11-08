import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_tourism_model_v1.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Wellness Tourism Package - Purchase Predictor")
st.write("Enter customer details to predict whether the customer is likely to purchase the Wellness Tourism Package.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person = st.number_input("Number of Persons Visiting", 1, 10, 3)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips per Year", 0, 30, 3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
with col2:
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children = st.number_input("Children Visiting (under 5)", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 20000, step=500)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    pitch_duration = st.number_input("Duration of Pitch (minutes)", 1, 150, 15)

# Save the inputs into a dataframe (column names/order match the training data)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": pitch_duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

st.subheader("Input data")
st.dataframe(input_df)

THRESHOLD = 0.5
if st.button("Predict"):
    proba = float(model.predict_proba(input_df)[0, 1])
    if proba >= THRESHOLD:
        st.success(f"Likely to PURCHASE the package (probability {proba:.1%}).")
    else:
        st.warning(f"Unlikely to purchase the package (probability {proba:.1%}).")
