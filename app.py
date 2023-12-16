from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('ml_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route("/predict", methods=["POST"])

def predict():
    nitrogen = request.form.get('nitrogen')
    fosfor = request.form.get('fosfor')
    kalium = request.form.get('kalium')
    temperature = request.form.get('suhu')
    humidity = request.form.get('kelembapan')
    ph = request.form.get('ph')

    input_query = np.array([[nitrogen, fosfor, kalium, temperature, humidity, ph]])

    recommendation = model.predict(input_query)[0]

    return jsonify({"Plant":str(recommendation)})

if __name__ == '__main__':
    app.run(debug=True)
