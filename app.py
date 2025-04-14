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
    print("got data: ", data)
    fmt_data = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(fmt_data)
    output = forestmodel.predict(new_data)
    print("predicted output: ", output)
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
