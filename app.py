import numpy as np
import io
import base64
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
import cv2

#local imports
from gradcam_visual import grad_cam
from loading_model import load_model_cache

def preprocessing_for_model(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array)
    return img_array   


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
                            img_array = preprocessing_for_model(img)
                            predictions = model.predict(img_array)
                            superimposed_image = grad_cam(model , img_array , 'conv5_block16_concat' , img)
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