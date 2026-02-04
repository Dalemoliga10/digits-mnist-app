import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="MNIST / Fashion MNIST Predictor", layout="centered")

st.title("MNIST / Fashion MNIST Predictor")
st.write("Sube una imagen tipo MNIST (fondo negro, número blanco) y el modelo hará la predicción")

# =========================
# Cargar modelo
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fashion_mnist.keras")

model = load_model()

# =========================
# Preprocesado de imagen
# =========================
def reshape_img(image):
    img = np.array(image.convert("L"))      # Escala de grises
    img = cv2.resize(img, (28, 28))          # Tamaño MNIST
    img = img / 255.0                        # Normalizar
    img = 1.0 - img                          # Invertir colores (CLAVE)
    img = (img > 0.5).astype(np.float32)     # Umbralizar
    img_flat = img.reshape(1, 784)           # Aplanar para Dense
    return img, img_flat

# =========================
# Subida de imagen
# =========================
uploaded_file = st.file_uploader(
    "Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Imagen original")
    st.image(image, width=150)

    img_28, img_flat = reshape_img(image)

    st.subheader("Imagen que ve el modelo (28x28)")
    st.image(img_28, clamp=True)

    prediction = model.predict(img_flat)
    probs = prediction[0]
    predicted_class = int(np.argmax(probs))

    st.subheader("Predicción")
    st.success(f"Clase predicha: {predicted_class}")

    st.subheader("Probabilidades por clase")
    for i, p in enumerate(probs):
        st.write(f"Clase {i}: {p:.6f}")

    # Rechazo simple si la confianza es baja
    if probs[predicted_class] < 0.7:
        st.warning("⚠️ Baja confianza: la imagen puede no ser válida para MNIST")

st.markdown("---")
st.caption("Modelo entrenado sobre MNIST/Fashion MNIST · Streamlit + TensorFlow")
