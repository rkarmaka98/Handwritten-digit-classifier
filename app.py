from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
model = load_model('digit_classifier_model.keras')  # Load your pre-trained model

@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img).argmax()
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
