from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions


model = None


def read_image(file):
    pil_image = Image.open(BytesIO(file))
    return pil_image


def load_model():
    model = tf.keras.applications.ResNet101V2(weights="imagenet")
    return model


def predict(image: np.ndarray):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    result = decode_predictions(model.predict(image), 5)[0]

    response = []
    for res in result:
        current_response = {"class": res[1], "confidence": round(float(res[2] * 100), 2)}
        response.append(current_response)

    return response
