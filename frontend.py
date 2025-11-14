import streamlit as st
import io
import PIL
import requests
import base64

backend_url = "http://127.0.0.1:8000/postai"

uploaded_file = st.file_uploader("Choose a image ...", type=['jpg' , 'png'])

if uploaded_file is not None :
    if st.button("Process"):
        files = {"file": uploaded_file}
        response = requests.post(backend_url, files = files)
        data = response.json()
        image_base64 = data.get("image_base64")
        decodebase64 = base64.b64decode(image_base64)
        st.image(decodebase64 , caption = "Processed Image")
        st.write(response.status_code)
        st.write("Prediction:", data.get("prediction"))