from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model (runs when Gunicorn starts)
model = tf.keras.models.load_model('plant_disease_model.keras')
print("Model loaded successfully")

# Preprocess image
def preprocess_image(image):
    img = Image.open(image).resize((224, 224))  # Resize to 224x224
    img = img.convert('RGB')  # Convert to RGB
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def home():
    return "It's very nice to meet, Welcome to the Plant Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        result = np.argmax(prediction, axis=1)[0]  # Get predicted class index
        return jsonify({'prediction': int(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # For local testing