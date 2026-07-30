import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    "cifake_model.h5",
    compile=False
)

# App title
st.title("🖼️ CIFAKE - AI Image Detector")
st.write("Upload an image to determine whether it is Real or AI Generated.")

# Upload image
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

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)
    confidence = float(pred[0][0])

    # Display result
    if confidence >= 0.5:
        st.success("Prediction: REAL")
        st.write(f"Confidence: {confidence * 100:.2f}%")
    else:
        st.error("Prediction: FAKE")
        st.write(f"Confidence: {(1 - confidence) * 100:.2f}%")
