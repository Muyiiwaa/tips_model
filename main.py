# import the needed libraries

import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import plotly.express as px

main_data = px.data.tips()
cat_cols = ['sex','smoker','day', 'time']

# load the prediction objects
label_encoders = joblib.load(filename="label_encoders.pkl")
scaler = joblib.load(filename="scaler.pkl")
model = joblib.load(filename="model_v1.pkl")

st.title('TIPS PREDICTION SYSTEM')

column1, column2 = st.columns(2)

with column1:
    total_bill = st.number_input(label= 'Total Bill') 
    sex = st.selectbox(label='Gender', options= ['Male','Female'])
    smoker = st.selectbox(label='Smoker', options= ['Yes','No'])
with column2:
    day = st.selectbox(label='Day', options= list(main_data['day'].unique()))
    time = st.selectbox(label='Time', options= list(main_data['time'].unique()))
    size = st.number_input(label= 'Size')
    
if st.button(label='predict', type='primary'):
    st.divider()
    sample_dict = {
        'total_bill': [total_bill], 'sex':[sex], 'smoker':[smoker],
        'day':[day], 'time':[time], 'size':[size]}
    data = pd.DataFrame(sample_dict)
    # encode the categorical columns
    for col in cat_cols:
        encoder = label_encoders[col]
        data[col] = encoder.transform(data[col])
    # scale the dataset
    data = scaler.transform(data)
    # get model prediction
    prediction = model.predict(data)
    st.success(f'This user is expected to tip around ${round(prediction[0], 3)}')
    
    
    