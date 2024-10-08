from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training_history')
def get_training_history():
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)
