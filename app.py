import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import StandardScaler

# Load the trained model and the scaler
with open('GradientBoostingModel', 'rb') as f:
    gbr = pickle.load(f)

with open('scaler', 'rb') as f:
    scaler = pickle.load(f)

def welcome():
    return "Welcome All"

def predict_aqi(pm2_5, pm10, no, no2, nox, nh3, co, so2):
    try:
        # Prepare the input data and normalize it
        input_data = np.array([[pm2_5, pm10, no, no2, nox, nh3, co, so2]])
        input_data_normalized = scaler.transform(input_data)
        
        # Make the prediction
        prediction = gbr.predict(input_data_normalized)
        return prediction[0]
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def main():
    st.title("Air Quality Index (AQI) Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Predict AQI Based on Pollutant Levels</h2>
    </div>
    <br><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.header("Input Pollutant Levels")
    pm2_5 = st.sidebar.number_input('PM2.5 (ug/m3)', min_value=0.0, max_value=1000.0)
    pm10 = st.sidebar.number_input('PM10 (ug/m3)', min_value=0.0, max_value=1000.0)
    no = st.sidebar.number_input('NO (ug/m3)', min_value=0.0, max_value=1000.0)
    no2 = st.sidebar.number_input('NO2 (ug/m3)', min_value=0.0, max_value=1000.0)
    nox = st.sidebar.number_input('NOx (ug/m3)', min_value=0.0, max_value=1000.0)
    nh3 = st.sidebar.number_input('NH3 (ug/m3)', min_value=0.0, max_value=1000.0)
    co = st.sidebar.number_input('CO (ug/m3)', min_value=0.0, max_value=1000.0)
    so2 = st.sidebar.number_input('SO2 (ug/m3)', min_value=0.0, max_value=1000.0)

    result = ""
    if st.button("Predict"):
        result = predict_aqi(pm2_5, pm10, no, no2, nox, nh3, co, so2)
        st.success(f'The predicted AQI is {result:.2f}')
    
    st.write("")

    if st.button("About"):
        st.header("About This App")
        st.write("""
        This app predicts the Air Quality Index (AQI) based on pollutant levels like PM2.5, PM10, NO2, SO2, and more.
        The model was trained on historical air quality data and uses Gradient Boosting for prediction.
        """)
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()
