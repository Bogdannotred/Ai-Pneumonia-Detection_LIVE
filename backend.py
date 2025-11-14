from fastapi import FastAPI , UploadFile, File
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



app = FastAPI()

# Incarca modelul
model = load_model("best_model.h5")

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






@app.post("/postai")
async def postai(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # force 3 channels
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    heatmap, class_idx = grad_cam(model , img_array , 'block5_conv3')
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap , cv2.COLORMAP_JET)




    return {"prediction": predictions.tolist()}
