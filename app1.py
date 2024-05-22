import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@flask_app.route("/")
def home():
    return render_template("dex.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("dex.html", prediction_text="The weather condition is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True,port= 3000)