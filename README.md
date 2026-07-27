# CIFAKE – AI Generated Image Detection

## Overview
CIFAKE is a deep learning project that classifies images as Real or AI-Generated using a Convolutional Neural Network (CNN). The project is built using TensorFlow and Keras and provides an easy-to-use interface for image prediction.

## Features
- Detects Real and AI-Generated images
- CNN-based image classification
- Streamlit web interface
- TensorFlow & Keras implementation
- Easy image prediction using uploaded files

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit

## Project Structure
```
CIFAKE/
│── app.py
│── predict.py
│── gradcam.py
│── README.md
│── requirements.txt
│
├── model/
│   └── cifake_model.keras
│
├── test_images/
│
└── outputs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

Terminal Version

```bash
python predict.py
```

Web App

```bash
streamlit run app.py
```

## Sample Output

Prediction: Real

Confidence: 89.86%

<img src="blob:chrome-untrusted://media-app/19deaeea-dcb4-4ac1-ab86-0610442bb29e" alt="![Prediction Result](prediction.png)/><img width="558" height="696" alt="image" src="https://github.com/user-attachments/assets/17a6ddf0-e3d4-409c-b19d-8c992abbd291" />


## Future Improvements

- Grad-CAM heatmap visualization
- Support for multiple AI image generators
- Improved CNN architecture
- Web deployment
