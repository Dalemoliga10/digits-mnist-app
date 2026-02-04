import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Fashion MNIST Predictor")

st.title("Fashion MNIST Predictor")
st.write("Sube una imagen y el modelo hará la predicción")

# Cargar el modelo (una sola vez)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dataset_Mnist_Ejercicio.keras")

model = load_model()

def reshape_img(image):
    img = np.array(image.convert("L"))     # escala de grises
    img = cv2.resize(img, (28, 28))        # tamaño MNIST
    img = img / 255.0
    img = np.expand_dims(img, 0)           # (1, 28, 28)
    return img

uploaded_file = st.file_uploader(
    "Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", width=150)

    img = reshape_img(image)
    prediction = model.predict(img)
    predicted_class = prediction.argmax()

    st.success(f"Predicción: {predicted_class}")
