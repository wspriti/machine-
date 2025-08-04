import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("road_accident_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚦 Road Accident Severity Predictor")

# Example input fields (update based on your model's actual features)
feature1 = st.number_input("Enter Feature 1 (e.g., Speed)", value=50.0)
feature2 = st.number_input("Enter Feature 2 (e.g., Age)", value=30.0)
feature3 = st.number_input("Enter Feature 3", value=1.0)
feature4 = st.number_input("Enter Feature 4", value=1.0)

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Class: {int(prediction[0])}")
