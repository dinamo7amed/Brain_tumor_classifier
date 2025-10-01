import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import time
from streamlit_lottie import st_lottie
import requests
import os

# -------------------------------
# Load Lottie Animation
# -------------------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_x62chJ.json")

# -------------------------------
# Page Background & Title
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
    }
    h1 {
        color: white;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Brain Tumor Classification")

# -------------------------------
# Download Model if not exists
# -------------------------------
model_url = "https://huggingface.co/dinamo7amed/brain_tumor_model/resolve/main/brain_tumor_classification.h5"
model_path = "brain_tumor_classification.h5"

if not os.path.exists(model_path):
    st.info("Downloading model... Please wait ⏳")
    r = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    st.success("Model downloaded ✅")

# -------------------------------
# Load Model
# -------------------------------
model = load_model(model_path)

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224,224)).convert("RGB")
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Display Lottie Animation
    st_lottie(lottie_animation, speed=1, width=300, height=300, key="brain")

    # Fake Progress Bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)

    # Predict
    prediction = model.predict(img_array)

    # Assuming 4 classes: adjust as per your model
    classes = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    pred_class = classes[np.argmax(prediction)]

    st.markdown(f"### Prediction: **{pred_class}**")
