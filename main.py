import streamlit as st
from loading import predict_and_view

st.title('Predict the food name by using image')
buffer = st.file_uploader('upload food image file here', type=['jpg', 'png', 'jpeg'])

if buffer is not None:
  st.write('loading')
  image, name, accuracy = predict_and_view(buffer, './acc_model.h5')
  st.image(image, caption=f'prediction is {name} with accuracy {accuracy}')
