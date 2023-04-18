from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pickle
from keras.utils import load_img
import os

app = Flask(__name__)

textModel = pickle.load(open("naive_bayes_classifier_new.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer_new.pkl", "rb"))

imageModel = load_model("image_model (1).h5")


@app.route("/test", methods=["GET"])
def hello():
    return "Test"


@app.route("/predictText", methods=["POST"])
def predictText():
    data = request.get_json()
    txt = data["text"]
    bagOfWords = vectorizer.transform([txt])
    res = textModel.predict(bagOfWords)
    response = dict()
    response["prediction"] = str(res[0])
    return jsonify(response)


@app.route("/predictImage", methods=["POST"])
def predictImage():
    f = request.files["file"]
    f.save(f.filename)
    image = load_img(f.filename, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    probability = imageModel.predict(img)
    response = dict()
    response["prediction"] = str(0 if probability[0][0] <= 0.50 else 1)
    os.remove(f.filename)
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
