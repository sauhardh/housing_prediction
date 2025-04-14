# from django.shortcuts import render
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import pickle as pkl

app = Flask(__name__)

# load the model
forestmodel = pkl.load(open("housingmodel.pkl", "rb"))
scaler = pkl.load(open("scaling.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    fmt_data = np.array(list(data.values())).reshape(1, -1)
    final_input = scaler.transform(fmt_data)
    output = forestmodel.predict(final_input)
    return jsonify(output[0])


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    output = forestmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=output)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
