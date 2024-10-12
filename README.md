# Hand-written digit classifier

This project is a simple web application that allows users to draw digits on a canvas, classify the drawn digits using a pre-trained MNIST model, and save the drawings. The application is built using Flask for the backend and HTML5 Canvas for the frontend.
<img width="810" alt="image" src="https://github.com/user-attachments/assets/ce3fe6f3-ef9d-4deb-8f0d-2243fe25943c">
## Features

- Draw digits on a canvas.
- Classify the drawn digits using a pre-trained MNIST model.
- Save the drawings as images.
- Clear the canvas to draw new digits.

## Requirements

- Python 3.x
- Flask
- Pillow
- TensorFlow

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/digit-classifier.git
    cd digit-classifier
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install Flask Pillow tensorflow
    ```

4. Download or train the MNIST model and save it as `digit-classifer-model.keras` in the project directory.

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Draw a digit on the canvas, then click "Classify Drawing" to see the predicted digit. You can also save the drawing or clear the canvas to draw a new digit.


