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
labels = ["bakso", "bebek_betutu", "gado_gado", "nasi_goreng", "pempek", "rawon", "rendang", "sate", "soto"]
descriptions = {
    "bakso": "Deskripsi Bakso",
    "bebek_betutu": "Deskripsi Bebek Betutu",
    "gado_gado": "Deskripsi Bebek Betutu",
    "nasi_goreng": "Deskripsi Bebek Betutu",
    "pempek": "Deskripsi Bebek Betutu",
    "rawon": "Deskripsi Bebek Betutu",
    "rendang": "Deskripsi Bebek Betutu",
    "sate": "Deskripsi Bebek Betutu",
    "soto": "Deskripsi Bebek Betutu",
    # ... tambahkan deskripsi lainnya
}

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
