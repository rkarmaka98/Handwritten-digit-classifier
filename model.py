import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import keras

# Load the MNIST dataset
# The MNIST dataset contains images of handwritten digits (0-9).
# It returns two sets of data: training data and testing data.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Resize images to 28x28 pixels
# Convert each image to 28x28 size, even though they are already 28x28,
# we use tf.image.resize to make sure they are in the correct format.
train_images = np.array([tf.image.resize(image[..., np.newaxis], (28, 28)).numpy() for image in train_images])
test_images = np.array([tf.image.resize(image[..., np.newaxis], (28, 28)).numpy() for image in test_images])

# Normalize pixel values to be between 0 and 1
# Pixel values in the images range from 0 to 255, we normalize them by dividing by 255.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Ensure the images have the correct shape (batch_size, height, width, channels)
# Expand dimensions to add a channel dimension, as the model expects a 4D input (batch, height, width, channels).
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Define the model
# Sequential model with more complex layers for digit classification.
model = models.Sequential([
    layers.Input(shape=(28, 28, 1), name='input_layer'),  # Input layer for grayscale 28x28 images with a unique name.
    layers.Conv2D(32, (3, 3), activation='relu', name='conv1', padding='same'),  # Convolutional layer with 32 filters and ReLU activation.
    layers.Conv2D(32, (3, 3), activation='relu', name='conv2', padding='same'),  # Additional convolutional layer with 32 filters.
    layers.MaxPooling2D((2, 2), name='maxpool1'),  # Max pooling layer to reduce spatial dimensions.
    layers.Conv2D(64, (3, 3), activation='relu', name='conv3', padding='same'),  # Convolutional layer with 64 filters.
    layers.Conv2D(64, (3, 3), activation='relu', name='conv4', padding='same'),  # Additional convolutional layer with 64 filters.
    layers.MaxPooling2D((2, 2), name='maxpool2'),  # Max pooling layer to further reduce dimensions.
    layers.Conv2D(128, (3, 3), activation='relu', name='conv5', padding='same'),  # Convolutional layer with 128 filters.
    layers.Conv2D(128, (3, 3), activation='relu', name='conv6', padding='same'),  # Additional convolutional layer with 128 filters.
    layers.MaxPooling2D((2, 2), name='maxpool3'),  # Max pooling layer to further reduce dimensions.
    layers.Flatten(name='flatten'),  # Flatten the 3D output to 1D for fully connected layers.
    layers.Dense(256, activation='relu', name='dense1'),  # Fully connected layer with 256 units.
    layers.Dropout(0.5, name='dropout1'),  # Dropout layer to reduce overfitting.
    layers.Dense(128, activation='relu', name='dense2'),  # Fully connected layer with 128 units.
    layers.Dropout(0.5, name='dropout2'),  # Dropout layer to reduce overfitting.
    layers.Dense(10, activation='softmax', name='output_layer')  # Output layer with 10 units (one for each digit) and softmax activation.
])

# Compile the model
# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# Train the model on the training data for 5 epochs, using the test set for validation.
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Save the model
# Save the trained model to a file for later use.
keras.saving.save_model(model, filepath='digit_classifier_model.keras')