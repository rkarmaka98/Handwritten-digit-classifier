# Hand-written digit classifier

This repository contains code for training a convolutional neural network (CNN) that classifies handwritten digits from the MNIST dataset. The model is implemented using TensorFlow and Keras, and is designed to recognize digits from 0 to 9 with high accuracy.

## Overview

The MNIST dataset is a classic dataset of handwritten digits that is widely used for training and testing image processing systems. In this repository, we build a more complex CNN model that can classify these digits. The model uses multiple convolutional layers, max pooling layers, fully connected layers, and dropout layers to achieve better accuracy.

<img width="1261" alt="image" src="https://github.com/user-attachments/assets/a6e61827-9d5c-4ba6-a0ab-244dcb6d3098">

## Features

- **Input Size**: The model accepts grayscale images of size 28x28.
- **Layers**: The model includes several convolutional layers, max pooling layers, fully connected layers, and dropout layers to minimize overfitting.
- **Model Training**: The model is trained on the MNIST dataset for digit recognition.
- **Save and Load Model**: The trained model can be saved and loaded for future use.

## Model Architecture

The model has the following architecture:

1. **Input Layer**: Accepts images of shape (28, 28, 1).
2. **Convolutional Layers**: Multiple convolutional layers with ReLU activation and `same` padding.
3. **Max Pooling Layers**: Reduce the spatial dimensions to help extract features more effectively.
4. **Fully Connected Layers**: Dense layers for combining features and learning complex relationships.
5. **Dropout Layers**: Used to reduce overfitting by randomly dropping nodes during training.
6. **Output Layer**: A softmax layer with 10 units, corresponding to the 10 digit classes (0-9).

## Installation

To run this repository, you need to have Python installed along with the necessary dependencies. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Training the Model**:
   Run the Python script to train the model:
   ```bash
   python train.py
   ```

2. **Saving the Model**:
   The trained model is saved as `digit_classifier_model.keras` for later use.

3. **Inference**:
   You can use the saved model to make predictions on new handwritten digit images.

## Code Walkthrough

### 1. Load and Preprocess Data

The MNIST dataset is loaded using `tf.keras.datasets.mnist`. The data is then resized to 28x28 pixels and normalized to values between 0 and 1.

### 2. Model Definition

The model is defined using the `tf.keras.models.Sequential` API. The model includes convolutional, max pooling, flatten, dense, and dropout layers to efficiently learn features from the input images.

### 3. Compilation and Training

The model is compiled using the Adam optimizer, sparse categorical crossentropy as the loss function, and accuracy as the metric. It is then trained on the training set for 5 epochs, with validation on the test set.

### 4. Save the Model

After training, the model is saved to a file (`digit_classifier_model.keras`) for future inference.

## Example

You can visualize the digit classification by using a Flask-based web interface where users can draw digits on a canvas, and the model will classify them.

## Requirements

- Python 3.6+
- TensorFlow
- Keras
- NumPy
- Flask (for web interface visualization)

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to fork this repository and contribute by adding more features, improving the model, or enhancing the web visualization.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun and is widely used for benchmarking machine learning algorithms.
- TensorFlow and Keras are used for building and training the neural network.


