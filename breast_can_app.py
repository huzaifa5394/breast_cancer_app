import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)
# Data description
st.title("Breast Cancer Prediction Dashboard")
st.header("Data overview")
dataFrame = pd.read_csv("data.csv")
st.write(dataFrame.head())
if st.checkbox("Show Data Description"):
    st.write(dataFrame.describe())
#heat map
st.header("Correlation Heatmap")
dataFrame["diagnosis"]=dataFrame["diagnosis"].apply(lambda x: 1 if x=="M" else 0 )
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(dataFrame.corr(), cmap="coolwarm", annot=False, fmt=".2f", ax=ax)
st.pyplot(fig)


st.header("Enter Feature Values")
radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")
smoothness_mean = st.number_input("Smoothness Mean")
compactness_mean = st.number_input("Compactness Mean")
concavity_mean = st.number_input("Concavity Mean")
concave_points_mean = st.number_input("Concave Points Mean")
symmetry_mean = st.number_input("Symmetry Mean")
fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

input_data = [
    radius_mean, texture_mean, perimeter_mean, area_mean,
    smoothness_mean, compactness_mean, concavity_mean,
    concave_points_mean, symmetry_mean, fractal_dimension_mean
]

def predict_dignose(input_data):
    data=np.array(input_data).reshape(1,-1)
    prediction=model.predict(data)
    return "Malignant" if prediction[0]==1 else "Benign"

if st.button("Predict"):
    dignosis=predict_dignose(input_data)
    st.write(f"The model predict that the patient is {dignosis}")
