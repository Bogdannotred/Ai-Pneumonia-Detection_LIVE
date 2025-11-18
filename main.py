
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


#save it in cache for better performance
@st.cache_resource
def load_model_cache():
    return load_model("final_pneumonia_model.h5")


def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_value = predictions[0]
        class_idx = 1 if pred_value > 0.5 else 0

    grads = tape.gradient(pred_value, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), class_idx


def main():
    model =load_model_cache()
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
                            heatmap = np.uint8(255 * heatmap)
                            heatmap = cv2.applyColorMap(heatmap , cv2.COLORMAP_JET)

                            #transform original image into a cv2 img
                            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            #take the values from oringinal image and resize it for heatmap
                            heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0])) 
                            #take the heatmap and superimpose it on original image
                            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

                            cv2.imwrite("heatmap.jpg", superimposed_img)

                            #save image into memory as .jpg
                            succes , encoded_image = cv2.imencode('.jpg' , superimposed_img)
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