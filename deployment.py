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

# Inputs without the increment buttons (step removed for free typing)
feature_1 = st.text_input('Orthophosphate (mg/L)', value="0.00")
feature_2 = st.text_input('Ammonium (mg/L)', value="0.00")
feature_3 = st.text_input('Nitrite/Nitrate (mg/L)', value="0.0")
feature_4 = st.text_input('Chlorophyll (µg/L)', value="0.0")
feature_5 = st.text_input('Salinity (Sal)', value="0.0")
feature_6 = st.text_input('Dissolved Oxygen (mg/L)', value="0.0")
feature_7 = st.text_input('Depth (m)', value="0.0")
feature_8 = st.text_input('pH', value="0.0")
feature_9 = st.text_input('Turbidity (NTU)', value="0.0")
feature_10 = st.text_input('Chlorophyll Fluorescence', value="0.0")

# Define the threshold values
thresholds = {
    'orthophosphate': 0.5,
    'ammonium': 0.5,
    'nitrite_nitrate': 1.0,
    'chlorophyll': 10.0
}

# Create input_features array
input_features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11]])

# Pollution Classification Logic
def classify_overall_pollution(predictions):
    """Classify the overall pollution based on nutrient levels."""
    status = {
        'orthophosphate': "Light" if predictions['orthophosphate'] <= thresholds['orthophosphate'] else ("Moderate" if predictions['orthophosphate'] <= 1.0 else "Heavy"),
        'ammonium': "Light" if predictions['ammonium'] <= thresholds['ammonium'] else ("Moderate" if predictions['ammonium'] <= 1.0 else "Heavy"),
        'nitrite_nitrate': "Light" if predictions['nitrite_nitrate'] <= thresholds['nitrite_nitrate'] else ("Moderate" if predictions['nitrite_nitrate'] <= 2.0 else "Heavy"),
        'chlorophyll': "Light" if predictions['chlorophyll'] <= thresholds['chlorophyll'] else ("Moderate" if predictions['chlorophyll'] <= 20.0 else "Heavy")
    }
    
    pollution_levels = list(status.values())
    
    if "Heavy" in pollution_levels:
        return "Heavy"
    elif "Moderate" in pollution_levels:
        return "Moderate"
    else:
        return "Light"

# Choose location
location = st.selectbox('Select Location', ['Homer', 'Seldovia'])

# Button to make current pollution prediction
if st.button('Predict Current Levels'):
    st.subheader(f'Predicted Nutrient Pollution Levels for {location}:')

    st.write(f"**Orthophosphate (mg/L):** {feature_1}")
    st.write(f"**Ammonium (mg/L):** {feature_2}")
    st.write(f"**Nitrite/Nitrate (mg/L):** {feature_3}")
    st.write(f"**Chlorophyll (µg/L):** {feature_4}")

    # Predict current nutrient pollution levels
    predictions = {}
    for target in ['orthophosphate', 'ammonium', 'nitrite_nitrate', 'chlorophyll']:
        predictions[target] = rf_models[target].predict(input_features)[0]
    
    overall_pollution = classify_overall_pollution(predictions)
    
    for nutrient, predicted_value in predictions.items():
        threshold = thresholds[nutrient]
        status = "Under Threshold" if predicted_value <= threshold else "Over Threshold"
        st.write(f"Predicted {nutrient.capitalize()} (mg/L): {predicted_value:.3f} - {status}")

    st.write(f"**Overall Nutrient Pollution: {overall_pollution}**")

    # Graphical representation
    fig, ax = plt.subplots()
    nutrients = ['Orthophosphate', 'Ammonium', 'Nitrite/Nitrate', 'Chlorophyll']
    values = [predictions['orthophosphate'], predictions['ammonium'], predictions['nitrite_nitrate'], predictions['chlorophyll']]
    thresholds_list = [thresholds['orthophosphate'], thresholds['ammonium'], thresholds['nitrite_nitrate'], thresholds['chlorophyll']]
    
    ax.bar(nutrients, values, color='blue', label='Predicted Values')
    ax.axhline(y=thresholds['orthophosphate'], color='red', linestyle='--', label='Orthophosphate Threshold')
    ax.axhline(y=thresholds['ammonium'], color='green', linestyle='--', label='Ammonium Threshold')
    ax.axhline(y=thresholds['nitrite_nitrate'], color='orange', linestyle='--', label='Nitrite/Nitrate Threshold')
    ax.axhline(y=thresholds['chlorophyll'], color='purple', linestyle='--', label='Chlorophyll Threshold')
    
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title(f'Nutrient Pollution Levels vs Thresholds for {location}')
    ax.legend()
    
    st.pyplot(fig)

# Time Series Prediction using LSTM
if st.button('Prediction of Nutrient Pollution Levels in Next 3 Years'):
    st.subheader(f'Time Series Predictions for {location}')

    # Prepare the input for LSTM (reshape as required by LSTM input)
    lstm_input = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

    lstm_predictions = lstm_predictions[:3]
    years = np.arange(2024, 2027) 
    fig, ax = plt.subplots()
    ax.plot(years, lstm_predictions.flatten(), marker='o', label='Predicted Pollution Level')
    ax.set_xlabel('Year')
    ax.set_ylabel('Nutrient Pollution Level (mg/L)')
    ax.set_title(f'Predicted Pollution Levels Over the Next 3 Years for {location}')
    st.pyplot(fig)
