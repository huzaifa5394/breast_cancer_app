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
Breast_Cancer_data=pd.read_csv("data.csv")
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
st.title("🩺Breast Cancer Prediction Dashboard")
st.header("📊Data overview")
dataFrame = pd.read_csv("data.csv")
st.write(dataFrame.head())
if st.checkbox("Show Data Description"):
    st.write(dataFrame.describe())
#heat map
st.header("📈Correlation Heatmap")
dataFrame["diagnosis"]=dataFrame["diagnosis"].apply(lambda x: 1 if x=="M" else 0 )
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(dataFrame.corr(), cmap="coolwarm", annot=False, fmt=".2f", ax=ax)
st.pyplot(fig)
#___________________________________
#Custom CSS for styling
# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Change background color of the main app container */
        .stApp {
            background-color: #242928;
          
        }

        /* Input and button styling */
        input[type="number"], .stButton button {
            border-radius: 10px;
            padding: 10px;
            background-color: #5a9; /* Light green */
            color: white;
            font-weight: bold;
            border: none;
        }

        /* Text and header styling */
        h1, h2, h3 {
            color: #fff;
            font-weight:bold;
            letter-spacing:2px; /* Dark blue for headers */
        }
        p{
            font-weight:bold;
            color:white;
        
        }
        
        /* Button specific styles */
        .stButton>button {
            color: white;
            border: solid 1px  linear-gradient(90deg, rgba(63,94,251,1) 16%, rgba(252,70,107,1) 82%);
            border-radius: 10px;
            padding: 0.5em 1em;
            font-weight: bold;
            font-size:15px;
        }

        /* Align content in the center */
        .block-container {
            padding: 2rem;
            font-weight:bold;
            font-color:white;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            background: rgb(34,195,134);
            background: linear-gradient(0deg, rgba(34,195,134,0.7987570028011204) 30%, rgba(160,45,253,1) 80%);
        }

    </style>
    """,
    unsafe_allow_html=True
)
#-----------------------------------------------------------------

st.header("✍️Enter Feature Values")
radius_mean = st.number_input("Radius Mean(enter number 6 to 28)")
texture_mean = st.number_input("Texture Mean(enter number 9 to 40)")
perimeter_mean = st.number_input("Perimeter Mean (enter number 43 to 190)")
area_mean = st.number_input("Area Mean (enter number 143 to 2500)")
smoothness_mean = st.number_input("Smoothness Mean (enter number 0.05 to 0.1)")
compactness_mean = st.number_input("Compactness Mean (enter number 0.01 to 0.3)")
concavity_mean = st.number_input("Concavity Mean (enter number 0 to 0.4)")
concave_points_mean = st.number_input("Concave Points Mean (enter number 0 to 0.2)")
symmetry_mean = st.number_input("Symmetry Mean (enter number 0.1 to 0.3)")
fractal_dimension_mean = st.number_input("Fractal Dimension Mean (enter number 0.05 to 0.09)")

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

# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align:center;">
        <p>Developed by Huzaifa | Breast Cancer Prediction App</p>
    </footer>
    """,
    unsafe_allow_html=True
)
