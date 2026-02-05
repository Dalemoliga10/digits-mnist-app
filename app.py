import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Prueba streamlit mnist numeros", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dataset_Mnist_Ejercicio.keras")

model = load_model()

def reshape_img(image):
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img_flat = img.reshape(1, 784)
    return img, img_flat

uploaded_file = st.file_uploader(
    "Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Imagen subida: ")
    st.image(image, width=150)

    img_28, img_flat = reshape_img(image)

    prediction = model.predict(img_flat)
    probs = prediction[0]
    predicted_class = int(np.argmax(probs))

    st.subheader("Predicción")
    st.success(f"Clase predicha: {predicted_class}")

    st.subheader("Probabilidades por clase")
    for i, p in enumerate(probs):
        st.write(f"Clase {i}: {p:.6f}")

