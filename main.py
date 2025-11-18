
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import io
import base64
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st


from gradcam_visual import grad_cam
from gradcam_visual import over_heatmap

#save it in cache for better performance
@st.cache_resource
def load_model_cache():
    return load_model("final_pneumonia_model.h5")


def main():
    model = load_model_cache()
    st.set_page_config(page_title="Pneumonia Detection", layout="wide", page_icon="🩺")
    st.title("🩺 Pneumonia Detection with Grad-CAM")
    uploaded_file = st.file_uploader("Choose a image ...", type=['jpg' , 'png'])
    col1, col2, col3 = st.columns([1, 3, 1])
    if uploaded_file is not None :
            with col2:
                if st.button("🩺 Process Image", use_container_width=True):
                    spin_left, spin_center, spin_right = st.columns([1, 1, 1])
                    with spin_center:
                        with st.spinner("Processing..."):
                            bytes_data = uploaded_file.getvalue()
                            img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                            img = img.resize((224, 224))
                            img_array = np.array(img)
                            img_array = np.expand_dims(img_array, axis=0) 
                            img_array = preprocess_input(img_array)
                            predictions = model.predict(img_array)
                            heatmap, class_idx = grad_cam(model , img_array , 'conv5_block16_concat')
                            superimposed_image = over_heatmap(img , heatmap)
                            cv2.imwrite("heatmap.jpg", superimposed_image)

                            #save image into memory as .jpg
                            succes , encoded_image = cv2.imencode('.jpg' , superimposed_image)
                            #encode to bytes
                            image_bytes = encoded_image.tobytes()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                            decodebase64 = base64.b64decode(image_base64)
                            #display image with gradcam
                            left, right = st.columns(2)
                            with left:
                                st.image(decodebase64 , caption = "Processed Image")
                            #display normal image
                            with right:
                                st.image(img , caption = "Original Image")
                            pred = predictions[0]
                            st.write(f"Prediction Pneumonia : {pred[0] * 100:.2f} %")

main()