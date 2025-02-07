import streamlit as st
from predict_page import predict_comfort
from predict_consumption_level import predict_consumption



st.title('Forecasting Thermal Comfort of Occupants and Energy Consumption Level')

st.header('eXtreme Gradient Boosting Models for Prediction')

st.write(
    """
    Prediction is valuable for anomaly detection, load profile-based building control and measurement and verification procedures.
    Below, we predict Thermal Comfort Levels of building occupants in the predict page and also predict Energy Consumption Levels as 
    either high or Low in the explore page. We have used XBoost algorithm to create models for either of the cases. 
    
    """
)

# Sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ("Predict", "Explore"))

if page == "Predict":
    predict_comfort()
elif page == "Explore":
    predict_consumption()