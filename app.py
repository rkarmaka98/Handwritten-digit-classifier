from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import base64
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('digit_classifier_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    data = request.form['imageData']
    if isinstance(data, list):
        data = ''.join(data)
    data = data.split(',')
    image_data = base64.b64decode(data[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = ImageOps.invert(image)
    image.save('drawing.png')
    return 'Image saved successfully!'

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form['imageData']
    if isinstance(data, list):
        data = data
    data = data.split(',')
    image_data = base64.b64decode(data[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image=ImageOps.invert(image)
    # Resize the image to 28x28 pixels while maintaining aspect ratio
    image = ImageOps.fit(image, (56, 56), method=Image.LANCZOS)
    image.show()
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 56, 56, 1)
    
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
