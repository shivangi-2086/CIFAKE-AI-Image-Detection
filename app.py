import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Debug information
# -----------------------------
st.write("TensorFlow Version:", tf.__version__)
st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir("."))

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(
    "cifake_model.h5",
    compile=False
)

# -----------------------------
# App Title
# -----------------------------
st.title("🖼️ CIFAKE - AI Image Detector")

st.write("Upload an image to determine whether it is Real or AI Generated.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # Preprocess
    # -----------------------------
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    st.write("Image Shape:", img.shape)
    st.write("Minimum Pixel:", float(np.min(img)))
    st.write("Maximum Pixel:", float(np.max(img)))

    img = np.expand_dims(img, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    pred = model.predict(img, verbose=0)

    confidence = float(pred[0][0])

    st.write("Raw Model Output:", confidence)

    if confidence >= 0.5:
        st.success("Prediction: REAL")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.error("Prediction: FAKE")
        st.write(f"Confidence: {(1-confidence)*100:.2f}%")
