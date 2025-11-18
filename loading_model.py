import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf


#save it in cache for better performance
@st.cache_resource
def load_model_cache():
    return load_model("final_pneumonia_model.keras")
