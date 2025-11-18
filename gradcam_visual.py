import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

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

def over_heatmap(img, heatmap):
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap , cv2.COLORMAP_JET)
    #transform original image into a cv2 img
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  
    #take the values from oringinal image and resize it for heatmap   
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))     
    #take the heatmap and superimpose it on original image   
    superimposed_image = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    return superimposed_image

