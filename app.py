import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "cifake_model.h5",
    compile=False
)

st.title("🖼️ CIFAKE - AI Image Detector")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)   
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    confidence = float(pred[0][0])

    st.write("Raw model output:", confidence)

    if confidence > 0.5:
        st.success("Prediction: Real")
        st.write(f"Confidence: {confidence * 100:.2f}%")
    else:
        st.success("Prediction: Fake")
        st.write(f"Confidence: {(1 - confidence) * 100:.2f}%")
