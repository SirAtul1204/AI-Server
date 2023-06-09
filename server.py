from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pickle
from keras.utils import load_img
import os
from io import BytesIO
from PIL import Image
import requests
import random

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
    data = request.get_json()
    x = requests.get(data["url"]).content
    fileName = str(random.randint(1, 1000)) + "." + data["url"].split(".")[-1]
    f = open(fileName, "wb")
    f.write(x)
    f.close()
    image = load_img(fileName, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    probability = imageModel.predict(img)
    response = dict()
    response["prediction"] = str(0 if probability[0][0] <= 0.50 else 1)
    os.remove(fileName)
    return jsonify(response)


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
