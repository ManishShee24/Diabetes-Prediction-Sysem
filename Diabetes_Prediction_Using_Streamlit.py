import numpy as np
import pandas as pd
import pickle
import streamlit as st

f = open('C:/Users/hp/Python_Projects/Resume_Projects/trained_model.sav', 'rb')
load_model = pickle.load(f)

# creating a function for diabetes prediction
def diabetes_prediction(input_data):
    input_data_array = np.asarray(input_data)
    input_data_reshaped = input_data_array.reshape(1, -1)

    prediction = load_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 1):
        print('The Patient has diabetes')
    else:
        print('The Patient has no diabetes')

# giving a title
st.title('Diabetes Prediction Web App')

# getting the input data from user
Pregnancies = st.text_input('Number of Pregnancy')
Glucose = st.text_input('Glucose Level')
BloodPressure = st.text_input('Blood Pressure Value')
SkinThickness = st.text_input('Skin Thickness Value')
Insulin = st.text_input('Insulin Level')
BMI = st.text_input('BMI Value')
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
Age = st.text_input('Age of the Person')

# code for prediction
diagnosis = ''

# creating a button for Prediction
if st.button('Diabetes Test Result'):
    diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
                                     DiabetesPedigreeFunction, Age])

    st.success(diagnosis)