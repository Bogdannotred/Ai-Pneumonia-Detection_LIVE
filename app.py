import numpy as np
import io
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
import cv2
import time

from gradcam_visual import grad_cam
from loading_model import load_model_cache


st.set_page_config(
    page_title="AI Pneumonia Detection", 
    layout="wide", 
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa; 
        }
        h1 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        div.stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

def preprocessing_for_model(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array)
    return img_array   

def main():
    local_css()
    model = load_model_cache()
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
        st.title("Control Panel")
        st.info(
            """
            **Pneumonia Detection AI**
            
            This app utilizes **MobileNetV2** and **Grad-CAM** to identify lung areas affected by pneumonia.
            """
        )
        st.markdown("---")
        st.warning(
            "⚠️ **Medical Disclaimer**\n\n"
            "This tool is strictly for educational/demonstrative purposes. "
            "It should not be used as a substitute for professional medical diagnosis"
        )

    #Main Area
    st.title("🫁 AI-Assisted Pneumonia Detection")
    st.markdown("##### Upload a chest X-ray for automatic analysis using MobileNetV2 and Grad-CAM.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Drag and drop an image here or click to upload", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        #Read image
        bytes_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        col_preview, col_action = st.columns([1, 2])
        
        with col_preview:
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
        with col_action:
            st.write("### The image is ready.")
            st.write("Press the button below to run the detection algorithm and generate the heatmap (Grad-CAM).")
            
            if st.button("🔍 Analyze X-ray", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preprocessing image...")
                progress_bar.progress(20)
                img_array = preprocessing_for_model(img)
                
                status_text.text("Running MobileNetV2 model...")
                progress_bar.progress(50)
                predictions = model.predict(img_array)
                pred_probability = predictions[0][0]
                
                status_text.text("Generating Grad-CAM Heatmap...")
                progress_bar.progress(80)
                superimposed_image = grad_cam(model, img_array, 'conv5_block16_concat', img)
                
                # Color conversion
                superimposed_image_rgb = cv2.cvtColor(superimposed_image, cv2.COLOR_BGR2RGB)
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                #Display Results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Container Metrics
                c1, c2, c3 = st.columns([1, 1, 2])
                
                is_pneumonia = pred_probability > 0.5
                confidence = pred_probability * 100 if is_pneumonia else (1 - pred_probability) * 100
                label = "PNEUMONIA" if is_pneumonia else "NORMAL"
                color = "red" if is_pneumonia else "green"

                with c1:
                    st.metric("Predicted Diagnosis", label, delta_color="off")
                with c2:
                    st.metric("Confidence Level", f"{confidence:.2f}%")
                with c3:
                    if is_pneumonia:
                        st.error(f"The model detected signs of pneumonia with a probability of {pred_probability:.2%}.")
                    else:
                        st.success(f"The model indicates a normal X-ray (Pneumonia probability: {pred_probability:.2%}).")

                # Visual Display
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(img, caption="Original Image", use_container_width=True)
                with res_col2:
                    st.image(superimposed_image_rgb, caption="Grad-CAM Analysis (Areas of Interest)", use_container_width=True) 
main()