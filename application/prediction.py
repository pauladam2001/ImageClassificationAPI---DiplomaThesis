from PIL import Image   # Python's built-in lib for image manipulation
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions


model = None


def read_image(file):
    pil_image = Image.open(BytesIO(file))
    return pil_image


def load_model():
    model = tf.keras.applications.ResNet101V2(weights="imagenet")   # loads a ResNet101V2 neural network (resnet architecture and 101 layers) trained on
    return model                                                    # the imagenet dataset (1.000.000 images and 1000 classes)


def predict(image: np.ndarray):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]   # resize the image to (224,224) bc this is the input size of the resnet architecture
    image = np.expand_dims(image, 0)    # adds another dimension to the image (incorporates it into a list) because the predict function only works with batches of images
    image = image / 127.5 - 1.0         # normalizes pixel values, it brings them to the same scale. If you have pixel values ranging from 0-255
                                        # this /127.5 - 1.0 will bring them all proportionally into the range (-1.0, 1.0). NNs train faster and better when
                                        # most neuron values inside of them are subunit. This is how it is trained, will not work otherwise

    result = decode_predictions(model.predict(image), 5)[0]  # return only the first 5 predictions
                                                             # the [0] comes from the batches, as predict takes a list of images it also outputs a list
                                                             # of predictions for each image. We are interested only in the first one
    response = []
    for res in result:
        current_response = {"class": res[1], "confidence": round(float(res[2] * 100), 2)}   # compute the result (all classes that were returned and the associated confidence)
        response.append(current_response)

    return response
