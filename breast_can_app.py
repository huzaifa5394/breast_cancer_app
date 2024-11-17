import streamlit as st
import numpy as np
import pandas as pd
#import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
#import os

#model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
#with open(model_path, "rb") as file:
    #model = pickle.load(file)
Breast_Cancer_data=pd.read_csv("C:/Users/ROSHAN/Desktop/breast cancer project/my_app/brest_can_app/data.csv")
Breast_Cancer_data["diagnosis"]=Breast_Cancer_data["diagnosis"].apply(lambda x: 1 if x=="M" else 0 )
feature=Breast_Cancer_data.drop("diagnosis",axis=1).to_numpy()
target=Breast_Cancer_data["diagnosis"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(feature,target, test_size=0.2, random_state=42)
clf=LogisticRegression()
param={"penalty":["l1","l2","elasticnet"],"C":[1,2,3,4,5,6,8,10,20,30,40,50]}
classifier=GridSearchCV(clf,param_grid=param,scoring='accuracy',cv=5)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
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
    prediction=classifier.predict(data)
    return "Malignant" if prediction[0]==1 else "Benign"

if st.button("Predict"):
    dignosis=predict_dignose(input_data)
    st.write(f"The model predict that the patient is {dignosis}")
