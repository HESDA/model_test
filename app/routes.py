from flask import Blueprint, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io
import os

main = Blueprint('main', __name__)

# Load the .h5 model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model.h5'))

# Labels and descriptions
foods = [
    {"label": "bakso", "description": "Deskripsi Bakso"},
    {"label": "bebek_betutu", "description": "Deskripsi Bebek Betutu"},
    {"label": "gado_gado", "description": "Deskripsi Gado-Gado"},
    {"label": "nasi_goreng", "description": "Deskripsi Nasi Goreng"},
    {"label": "pempek", "description": "Deskripsi Pempek"},
    {"label": "rawon", "description": "Deskripsi Rawon"},
    {"label": "rendang", "description": "Deskripsi Rendang"},
    {"label": "sate", "description": "Deskripsi Sate"},
    {"label": "soto", "description": "Deskripsi Soto"}
]

labels = [food['label'] for food in foods]
descriptions = {food['label']: food['description'] for food in foods}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

@main.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    input_data = preprocess_image(image)
    
    predictions = model.predict(input_data)
    predicted_label = labels[np.argmax(predictions)]
    description = descriptions[predicted_label]
    
    response = {
        "label": predicted_label,
        "description": description
    }
    
    return jsonify(response)

@main.route('/list', methods=['GET'])
def list_foods():
    return jsonify(foods)
