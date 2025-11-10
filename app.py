import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and encoder
model = pickle.load(open(r"disease_model.pkl", "rb"))
le = pickle.load(open(r"label_encoder.pkl", "rb"))

st.title("Disease Prediction App")

# Load your original dataset to get symptom names
df = pd.read_csv(r"Training.csv")
symptoms = df.drop('prognosis', axis=1).columns.tolist()

# User selects symptoms
selected = st.multiselect("Select the symptoms you have:", symptoms)

# Create input vector
input_data = np.zeros(len(symptoms))
for symptom in selected:
    input_data[symptoms.index(symptom)] = 1

if st.button("Predict Disease"):
    pred = model.predict([input_data])[0]
    disease = le.inverse_transform([pred])[0]
    st.success(f" Predicted Disease: **{disease}**")
