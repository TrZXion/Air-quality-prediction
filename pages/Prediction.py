# trunk-ignore-all(isort)
import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_js_eval import streamlit_js_eval, get_geolocation
import requests


# Function to make API call using latitude and longitude
def make_api_call(latitude, longitude):
    # Your API endpoint URL
    api_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid=9284128446974ce6bf1808e0aced0115"
    response = requests.get(api_url, timeout=5)

    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error("Error fetching data from API")


if st.sidebar.checkbox("Check my location"):
    loc = get_geolocation()

    # Check if loc is None
    if loc is not None:
        # st.write(f"Your coordinates are {loc}")
        latitude = loc.get("coords", {}).get("latitude")
        longitude = loc.get("coords", {}).get("longitude")

        if latitude is not None and longitude is not None:
            # Call the API using latitude and longitude
            api_data = make_api_call(latitude, longitude)

            # Extract pollutant data from API response
            pollutants = api_data.get("list", [{}])[0].get("components", {})

            # Display pollutant values using st.metric in sidebar
            st.sidebar.title("Pollutant Data:")
            st.sidebar.metric(label="CO", value=pollutants.get("co"))
            st.sidebar.metric(label="NO", value=pollutants.get("no"))
            st.sidebar.metric(label="NO2", value=pollutants.get("no2"))
            st.sidebar.metric(label="O3", value=pollutants.get("o3"))
            st.sidebar.metric(label="SO2", value=pollutants.get("so2"))
            st.sidebar.metric(label="PM2.5", value=pollutants.get("pm2_5"))
            st.sidebar.metric(label="PM10", value=pollutants.get("pm10"))
            st.sidebar.metric(label="NH3", value=pollutants.get("nh3"))
        else:
            st.error("Latitude or longitude is missing.")


st.title("Air Quality Particulate Matter Prediction")

model = pickle.load(open("forest_search_model_new.pkl", "rb"))

# Add user input section
st.header("Enter the parameters:")
year = st.number_input("Year")
pm25_tempcov = st.number_input("Temperature")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
city_encoded = st.number_input("City Number")
user_submit = st.button("Predict")

# Prepare user data for prediction
if user_submit:
    # Create a DataFrame using user input
    user_data = pd.DataFrame(
        {
            "year": [year],
            "pm25_tempcov": [pm25_tempcov],
            "latitude": [latitude],
            "longitude": [longitude],
            "city_encoded": [city_encoded],
        }
    )
    prediction = model.predict(user_data)
    st.subheader("Prediction Result:")
    st.write(
        "Based on the provided information, PM 2.5 Concentration is",
        prediction,
        "µg/m3",
    )
