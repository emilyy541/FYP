import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the saved Random Forest models for current prediction
with open('random_forest_model.pkl', 'rb') as file:
    rf_models = pickle.load(file)

# Load the LSTM model for time series prediction
lstm_model = tf.keras.models.load_model('lstm_model.h5')

# Streamlit app interface
st.title('Early Detection of Nutrient Pollution in Gulf of Alaska')

st.write("""
    ### Enter the values for the following variables:
""")

# Inputs for features based on the scale of the variables
feature_1 = st.number_input('Orthophosphate (mg/L)', format="%.3f")
feature_2 = st.number_input('Ammonium (mg/L)', format="%.3f")
feature_3 = st.number_input('Nitrite/Nitrate (mg/L)', format="%.3f")
feature_4 = st.number_input('Chlorophyll (µg/L)', format="%.3f")
feature_5 = st.number_input('Temperature (°C)', step=0.1)
feature_6 = st.number_input('Salinity (Sal)', format="%.3f")
feature_7 = st.number_input('Dissolved Oxygen (mg/L)', step=0.1)
feature_8 = st.number_input('Depth (m)', format="%.3f")
feature_9 = st.number_input('pH', step=0.1)
feature_10 = st.number_input('Turbidity (NTU)', format="%.3f")
feature_11 = st.number_input('Chlorophyll Fluorescence', format="%.3f")

# Pollution Classification Logic
def classify_pollution(value):
    """Classify pollution level based on predicted value."""
    if value > 1.0:
        return "Heavy"
    elif 0.5 < value <= 1.0:
        return "Moderate"
    else:
        return "Light"

# Button to make prediction
if st.button('Predict Current Levels'):
    # Convert inputs into a 2D numpy array for the model
    input_features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11]])
    
    # Use Random Forest to predict current pollution levels for each variable
    st.subheader('Predicted Nutrient Pollution Levels')

    for location in ['Homer', 'Seldovia']:
        st.write(f"**Location: {location}**")
        for target in ['orthophosphate', 'ammonium', 'nitrite_nitrate', 'chlorophyll']:
            prediction = rf_models[target].predict(input_features)[0]
            pollution_level = classify_pollution(prediction)  # Classify the pollution level
            st.write(f"Predicted {target.capitalize()} (mg/L): {prediction:.3f} - {pollution_level} Pollution")

# Time Series Prediction using LSTM
if st.button('Prediction of Nutrient Pollution Levels in Next 3 Years'):
    st.subheader('Time Series Predictions')

    # Prepare the input for LSTM (reshape as required by LSTM input)
    lstm_input = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

    # Predict the next 3 years using LSTM
    lstm_predictions = lstm_model.predict(lstm_input)

    # Generate years for x-axis
    years = np.arange(2024, 2027)
    
    # Plot the time series predictions
    fig, ax = plt.subplots()
    ax.plot(years, lstm_predictions.flatten(), marker='o', label='Predicted Pollution Level')
    ax.set_xlabel('Year')
    ax.set_ylabel('Nutrient Pollution Level (mg/L)')
    ax.set_title('Predicted Pollution Levels Over the Next 3 Years')
    st.pyplot(fig)
