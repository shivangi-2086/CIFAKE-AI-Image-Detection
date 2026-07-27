import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("model/cifake_model.keras")

# Input image
img_path = input("Enter image path: ")

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert to NumPy array
img = image.img_to_array(img)

# Normalize
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

confidence = float(prediction[0][0])

if confidence > 0.5:
    print("\nPrediction: AI Generated")
    print(f"Confidence: {confidence*100:.2f}%")
else:
    print("\nPrediction: Real")
    print(f"Confidence: {(1-confidence)*100:.2f}%")