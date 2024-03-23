import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image

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
