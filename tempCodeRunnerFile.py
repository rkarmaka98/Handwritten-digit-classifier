from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import base64
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import deserialize as clone_layer

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
    image = ImageOps.fit(image, (28, 28), method=Image.LANCZOS)
    # image.show()
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.form['imageData']
    if isinstance(data, list):
        data = ''.join(data)
    data = data.split(',')
    image_data = base64.b64decode(data[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # Invert the image so the ink is white and the background is black
    inverted_image = ImageOps.invert(image)
    
    # Resize the image to 56x56 pixels while maintaining aspect ratio
    resized_image = ImageOps.fit(inverted_image, (28, 28), method=Image.LANCZOS)
    
    image_array = np.array(resized_image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)


    # Create an activation model using a new input layer
    input_layer = Input(shape=(28, 28, 1), name='unique_input_layer')
    x = input_layer
    layer_outputs = []

    # Rebuild the model using the loaded model's layers, cloning each layer
    for layer in model.layers:
        cloned_layer = clone_layer(tf.keras.layers.serialize(layer))  # Clone the layer
        x = cloned_layer(x)
        if 'conv' in layer.name.lower():
            layer_outputs.append(x)
    activation_model = Model(inputs=input_layer, outputs=layer_outputs)

    # Get the activations for the input image
    activations = activation_model.predict(image_array)
    
    # Convert activations to base64 images
    activation_images = []
    for layer_index, activation in enumerate(activations):
        num_filters = activation.shape[-1]
        height, width = activation.shape[1:3]

        # Create a grid to plot the activations
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        grid = np.zeros((grid_size * height, grid_size * width))
        
        for i in range(grid_size):
            for j in range(grid_size):
                filter_index = i * grid_size + j
                if filter_index < num_filters:
                    filter_activation = activation[0, :, :, filter_index]
                    grid[i*height:(i+1)*height, j*width:(j+1)*width] = filter_activation
        
        # Normalize the grid to be between 0 and 255
        grid = (grid - grid.min()) / (grid.max() - grid.min()) * 255
        grid = grid.astype(np.uint8)
        
        # Convert the grid to an image
        grid_image = Image.fromarray(grid)
        buffered = io.BytesIO()
        grid_image.save(buffered, format="PNG")
        grid_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        activation_images.append(grid_image_base64)
    
    return jsonify({'activations': activation_images})

if __name__ == '__main__':
    app.run(debug=True)
