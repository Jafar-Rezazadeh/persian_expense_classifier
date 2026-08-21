from pathlib import Path

from colorama import Fore
from tensorflow.keras.models import Model
import tensorflow as tf


def save_keras_model(model: Model, path: Path):
    try:
        model_path = path / "cnn_persian_expense_classifier.keras"
        model.save(model_path)
        print(Fore.GREEN, "successfully saved the model in {model_path}")
    except Exception as e:
        print(Fore.RED, "error while saving model:", e)


def save_tflite_model(model: Model, path: Path):

    path = path / "cnn_persian_expense_classifier.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(path, "wb") as file:
        file.write(tflite_model)
