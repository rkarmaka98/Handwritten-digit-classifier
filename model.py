import tensorflow as tf
import numpy as np
import json

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy data
x_train = np.random.rand(100, 4)
y_train = np.random.randint(3, size=100)

# Train the model and capture the weights
class WeightHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = [layer.get_weights() for layer in self.model.layers]
        weights_as_lists = [[w.tolist() for w in layer_weights] for layer_weights in weights]
        with open(f'weights_epoch_{epoch+1}.json', 'w') as f:
            json.dump(weights_as_lists, f)

history = WeightHistory()
model.fit(x_train, y_train, epochs=5, callbacks=[history])
