from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/weights')
def get_weights():
    with open('weights_epoch_5.json', 'r') as f:
        weights = json.load(f)
    return jsonify(weights)

if __name__ == '__main__':
    app.run(debug=True)
