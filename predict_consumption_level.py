import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
from predict_page import load_models

# Load the models
model_th, model_clrf, xgb_model, xgb_model_cl = load_models()

# Calculate Dew Point
def calculate_dewpoint(tempC, humidity):
    """
    Calculate the dew point in Celsius given the temperature (tempC) in Celsius and relative humidity (humidity) in percent.
    
    Uses the Magnus-Tetens approximation:
        dew_point = (b * alpha) / (a - alpha)
    where:
        alpha = (a * tempC) / (b + tempC) + ln(RH/100)
        a = 17.27, b = 237.7°C
    
    Parameters:
        tempC (float): Temperature in Celsius.
        humidity (float): Relative humidity in percent.
    
    Returns:
        float: Calculated dew point in Celsius.
    """
    a = 17.27
    b = 237.7
    alpha = (a * tempC) / (b + tempC) + np.log(humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def predit_consumption():
    st.header("Prediction of Thermal Sensation (Comfort)")
    st.write("""
        Predicting building energy consumption is key to boosting efficiency and environmental sustainability.
        Datasets used are the ASHRAE Great Energy Predictor III (3 years of hourly readings from over 1,000 
        buildings, Kaggle) and the Building Data Genome Project 2 (2 years of data from 3,053 non-residential 
        meters, Nature).
        We have used XGBoost predicitve model to forecast energy consumption levels. 
        Accurate predictions allow building managers to optimize energy use, lower emissions and costs, 
        and promote sustainable building practices (PMC).
        
        """)

    st.write("### Input data for prediction:")

    # User input widgets
    air_temp = st.slider("Select air Temperature Value (°C):", min_value=-40, max_value=50, step=1)
    rel_humidity = st.slider("Select relative humidity (%):", min_value=0, max_value=100, step=5)
    pressure = st.number_input(
        label="Enter Sea Level Pressure (hPa):",
        min_value=750.00,
        max_value=1200.00,
        value=1013.25,  # Default value set to standard atmospheric pressure
        step=0.01,
        format="%.2f"
    )
    wind_direction = st.number_input(
        label="Enter Wind Direction (degrees):",
        min_value=0.00,
        max_value=360.00,
        value=0.00,  # Default value
        step=0.01,
        format="%.2f"
    ) 
    date = st.date_input("Select Date:", datetime.date.today())
    time = st.time_input("Select Time:", datetime.datetime.now().time())

    # Combine date and time into a single datetime object
    timestamp = datetime.datetime.combine(date, time)

    # Calculate Dew Point
    dew_point = calculate_dewpoint(air_temp, rel_humidity)

    # Create a DataFrame with the specified structure
    data = {
        'TemperatureC': [air_temp],
        'Dew PointC': [dew_point],
        'Humidity': [rel_humidity],
        'Sea Level PressurehPa': [pressure],
        'VisibilityKm': [16.1],
        'WindDirDegrees': [wind_direction]
    }

    df = pd.DataFrame(data, index=[timestamp])

    # Display the DataFrame
    st.write("### Input DataFrame:")
    st.dataframe(df)

    # Ensure the DataFrame columns match the order and names expected by the model
    feature_columns = ['TemperatureC', 'Dew PointC', 'Humidity', 'Sea Level PressurehPa', 'VisibilityKm', 'WindDirDegrees']
    X_input = df[feature_columns]

    # Convert the DataFrame to a DMatrix
    dinput = xgb.DMatrix(X_input)

    # Make the prediction
    prediction = (xgb_model_cl.predict(dinput) > 0.5).astype(int)

    # Interpret the prediction
    if prediction == 0:
        st.write("**Prediction:** Energy consumption is Low.")
    else:
        st.write("**Prediction:** Energy consumption is High.")
