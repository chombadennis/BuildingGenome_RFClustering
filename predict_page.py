import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load models once (this can be cached)
@st.cache_data
def load_models():
    with open ('rf_models.pkl', 'rb') as file:
        data = pickle.load(file)
        model_th = data['model_th']
        model_clrf = data['model_clrf']
        xgb_model = data['xgb_model']
        xgb_model_cl = data['xgb_model_cl']
    return model_th, model_clrf, xgb_model, xgb_model_cl

# Function to map the predicted encoded value to the original comfort scale (-3 to 3)
def map_comfort(value):
    # Map predicted values back to the original scale (-3 to 3)
    comfort_mapping = {
        0: 'Very Cold (-3)',   # Mapped to -3
        1: 'Cold (-2)',        # Mapped to -2
        2: 'Mildly Uncomfortable (-1)',  # Mapped to -1
        3: 'Neutral/Comfortable (0)',   # Mapped to 0
        4: 'Mildly Uncomfortable (1)',  # Mapped to 1
        5: 'Hot (2)',          # Mapped to 2
        6: 'Very Hot (3)'      # Mapped to 3
    }
    return comfort_mapping.get(value, "Unknown")

# Load the models
model_th, model_clrf, xgb_model, xgb_model_cl = load_models() 

# Main function for prediction
def predict_comfort():
    st.header("Prediction of Thermal Sensation (Comfort)")
    st.write("""
                Predicting the comfort of buildings  involves uisng different environmental, geographical and physical 
                features to determine whether indoor conditions are comfortable or not. We have used data from the ASHRAE 
                dataset, which is a collection of  datasets from publicly available sources that are freely available online 
                and from closed systems that required extraction by the facility management teams from many of the data donor 
                locations.
                We have used the eXtreme Gradient Boosting (XBoost) algorithm to create a model that classifies features into
                distinct comfort zones. The model learns from historical data and the associated comfort level ratings to make 
                real-time predictions.
                We aim to ensure that indoor environments are maintained within the optimal parameters subject to different 
                environmental, physical and geographical conditions.This  will consequently improve overall comfort and energy 
                efficiency. This approach supports proactive adjustments to HVAC systems, reducing energy waste while maintaining 
                a comfortable environment.
                """
                )

    st.write("""### Input data for prediction:""")

    # User input widgets outside the cached function
    Season = ('Autumn', 'Spring', 'Summer', 'Winter')
    Climate = ('Cool-summer mediterranean', 'Hot semi-arid', 'Hot-summer Mediterranean', 'Hot-summer mediterranean',
               'Humid subtropical', 'Monsoon-influenced humid subtropical', 'Oceanic', 
               'Subtropical highland', 'Temperate oceanic', 'Tropical monsoon', 
               'Tropical rainforest', 'Tropical wet savanna', 'Warm-summer Mediterranean', 
               'Warm-summer humid continental')
    City = ('Ahmedabad', 'Athens', 'Auburn', 'Bangalore', 'Bangkok', 'Bedong', 
            'Berkeley', 'Beverly Hills', 'Brisbane', 'Cardiff', 'Chennai', 'Darwin', 
            'Delhi', 'Florianopolis', 'Gothenburg', 'Goulburn', 'Halmstad', 'Hampshire', 
            'Hyderabad', 'Ilam', 'Imola', 'Jaipur', 'Kalgoorlie', 'Karlsruhe', 'Kinarut', 
            'Kota Kinabalu', 'Kuala Lumpur', 'Kuching', 'Lisbon', 'Lodi', 'London', 
            'Lyon', 'Maceio', 'Makati', 'Malmo', 'Melbourne', 'Montreal', 'Oxford', 
            'Palo Alto', 'Porto', 'Putra Jaya', 'San Francisco', 'San Ramon', 'Shimla', 
            'Singapore', 'Sydney', 'Townsville', 'Varese', 'Walnut Creek', 'Wollongong')
    Country = ('Australia', 'Brazil', 'Canada', 'France', 'Germany', 'Greece', 'India', 
               'Iran', 'Italy', 'Malaysia', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 
               'Thailand', 'UK', 'USA')
    Building_Type = ('Classroom', 'Multifamily housing', 'Office', 'Others', 'Senior center')
    Cooling_Strategy = ('Air Conditioned', 'Mechanically Ventilated', 'Mixed Mode', 'Naturally Ventilated')
    Sex = ('Female', 'Male')

    Year = st.selectbox("Select Year:", list(range(2010, 2031)))
    season = st.selectbox("Select Season:", Season)
    climate = st.selectbox("Select Climate:", Climate)
    city = st.selectbox("Select City:", City)
    country = st.selectbox("Select Country:", Country)
    building_type = st.selectbox("Select Building Type:", Building_Type)
    cooling_strategy = st.selectbox("Select Cooling Strategy:", Cooling_Strategy)
    sex = st.selectbox("Select Sex:", Sex)
    
    Clo_value = st.slider("Select Clo value (clo):", min_value=0.0, max_value=3.0, step=0.1)
    Met_value = st.slider("Select Metabollic Value (MET):", min_value=0.0, max_value=12.0, step=0.1)
    Air_temp = st.slider("Select air Temperature Value(°C):", min_value=-40, max_value=50, step=1)
    Rel_humidity = st.slider("Select relative humidity(%):", min_value=0, max_value=100, step=5)
    Air_velocity = st.slider("Select air velocity(m/s):", min_value=0.0, max_value=10.0, step=0.1)

    # Create DataFrame for prediction
    data = {
        "Year": [Year],
        "Clo": [Clo_value],
        "Met": [Met_value],
        "Air temperature (C)": [Air_temp],
        "Relative humidity (%)": [Rel_humidity],
        "Air velocity (m/s)": [Air_velocity],
        "Season": pd.Categorical([season], categories=Season),
        "Climate": pd.Categorical([climate], categories=Climate),
        "City": pd.Categorical([city], categories=City),
        "Country": pd.Categorical([country], categories=Country),
        "Building type": pd.Categorical([building_type], categories=Building_Type),
        "Cooling startegy_building level": pd.Categorical([cooling_strategy], categories=Cooling_Strategy),
        "Sex": pd.Categorical([sex], categories=Sex)
    }

    df = pd.DataFrame(data)

    # Convert categorical variables into dummy variables
    df_dummies = pd.get_dummies(df)

    # Debugging: Print the dummy-encoded DataFrame
    st.write("Dummy-encoded DataFrame for prediction:", df_dummies)
    
    # Check the column names of df_dummies to ensure they match the training set
    st.write("Columns in the dummy-encoded DataFrame:", df_dummies.columns)

    # Ensure the model receives the same number of features it was trained with
    # Debugging: Check if the length of df_dummies matches the number of features the model expects
    st.write(f"Number of features in df_dummies: {df_dummies.shape[1]}")

    # Predict using the model
    try:
        thermal_comfort = xgb_model.predict(df_dummies)
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        return

    # Debugging: Print raw prediction
    st.write(f"Raw prediction output from the model: {thermal_comfort}")

    # Map the prediction to the comfort scale
    comfort_level = [map_comfort(x) for x in thermal_comfort]

    # Display the result
    st.write(f"The Predicted Thermal Comfort Level: {comfort_level[0]}")
