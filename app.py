from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import io

app = Flask(__name__)
model = load_model('digit_classifier_model.keras')  # Load your pre-trained model

# Create a new model that outputs the activations of each layer
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.inputs, outputs=layer_outputs)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0 # Normalize the image
    activations = activation_model.predict(img)
    predictions = model.predict(img).argmax()
    
    # Convert activations to a serializable format
    activations_serializable = [activation.tolist() for activation in activations]
    
    return jsonify({'prediction': int(predictions), 'activations': activations_serializable})

if __name__ == '__main__':
    app.run(debug=True)
