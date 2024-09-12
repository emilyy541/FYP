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
    ### Enter the values for the following 11 input features:
    1. **Orthophosphate** (mg/L)
    2. **Ammonium** (mg/L)
    3. **Nitrite/Nitrate** (mg/L)
    4. **Chlorophyll** (µg/L)
    5. **Temperature** (°C)
    6. **Salinity**
    7. **Dissolved Oxygen Content** (mg/L)
    8. **Depth** (m)
    9. **pH**
    10. **Turbidity**
    11. **Chlorophyll Fluorescence**
""")

# Inputs for the 11 features
feature_1 = st.number_input('Orthophosphate (mg/L)', value=0.0)
feature_2 = st.number_input('Ammonium (mg/L)', value=0.0)
feature_3 = st.number_input('Nitrite/Nitrate (mg/L)', value=0.0)
feature_4 = st.number_input('Chlorophyll (µg/L)', value=0.0)
feature_5 = st.number_input('Temperature (°C)', value=0.0)
feature_6 = st.number_input('Salinity', value=0.0)
feature_7 = st.number_input('Dissolved Oxygen Content (mg/L)', value=0.0)
feature_8 = st.number_input('Depth (m)', value=0.0)
feature_9 = st.number_input('pH', value=7.0)
feature_10 = st.number_input('Turbidity', value=0.0)
feature_11 = st.number_input('Chlorophyll Fluorescence', value=0.0)

# Button to make prediction
if st.button('Predict Current Levels'):
    # Convert inputs into a 2D numpy array for the model (11 features)
    input_features = np.array([[feature_1, feature_2, feature_3, feature_4, 
                                feature_5, feature_6, feature_7, feature_8, 
                                feature_9, feature_10, feature_11]])

    # Display the input nutrient levels
    st.write(f"**Orthophosphate (mg/L):** {feature_1}")
    st.write(f"**Ammonium (mg/L):** {feature_2}")
    st.write(f"**Nitrite/Nitrate (mg/L):** {feature_3}")
    st.write(f"**Chlorophyll (µg/L):** {feature_4}")

    # Use Random Forest to predict current pollution levels for each target variable
    st.subheader('Predicted Nutrient Pollution Levels')

    for location in ['Homer', 'Seldovia']:
        st.write(f"**Location: {location}**")
        for target in ['orthophosphate', 'Ammonium', 'Nitrite_Nitrate', 'Chlorophyll']:
            prediction = rf_models[target].predict(input_features)
            st.write(f"Predicted {target.capitalize()} (mg/L): {prediction[0]}")

# Time Series Prediction using LSTM
if st.button('Prediction of Nutrient Pollution Levels in Next 3 Years'):
    st.subheader('Time Series Predictions')

    # Prepare the input for LSTM (reshape as required by LSTM input)
    lstm_input = input_features.reshape((1, 1, input_features.shape[1]))

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
