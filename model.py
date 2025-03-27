import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the pre-trained model
MODEL_PATH = 'model/pomegranate_disease_model.h5'
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Healthy', 'Disease1', 'Disease2']  # Replace with actual labels

def predict_disease(image_path):
    """
    Predict the disease from the given image using the pre-trained model.
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Adjust target size as per your model
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Return the class label
    return CLASS_LABELS[predicted_class]