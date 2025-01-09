#  streamlit run laptop_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Define the relative paths to the files
pipe_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
data_path = os.path.join(os.path.dirname(__file__), 'laptop_data.pkl')

# Load the pickle files
pipe = pickle.load(open(pipe_path, 'rb'))
data = pickle.load(open(data_path, 'rb'))


st.title("Laptop Price Predictor")

# brand
company= st.selectbox('Brand',data['Company'].unique())

# type of laptop
type= st.selectbox('Type',data['TypeName'].unique())

# Ram
ram= st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight= st.number_input('Weight of Laptop')

# Touchscreen
touchscreen= st.selectbox('TouchScreen',['No','Yes'])

# IPS
ips= st.selectbox('IPS',['No','Yes'])

# screen size
screen_size= st.number_input('Screen Size')

# resolution
resolution= st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
cpu= st.selectbox('CPU',data['cpu_brand'].unique())

hdd= st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd= st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu= st.selectbox('GPU',data['Gpu_brand'].unique())

os= st.selectbox('OS',data['OS'].unique())

if st.button('Predict Price'):
    ppi= None
    if touchscreen== 'Yes':
        touchscreen= 1
    else:
        touchscreen= 0
    if ips== 'Yes':
        ips= 1
    else:
        ips= 0

    x_res= int(resolution.split('x')[0])
    y_res= int(resolution.split('x')[1])
    ppi= ((x_res**2) + (y_res**2))**0.5/screen_size
    query= np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query= query.reshape(1,12)
    st.title("The Predicted Price of this Configuration is "+str(int(np.exp(pipe.predict(query)[0]))))