from pathlib import Path

from tensorflow.keras.models import Model
import tensorflow as tf


def save_keras_model(model: Model, path: Path):
    model.save(path)


def save_tflite_model(model: Model, path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(path, "wb") as file:
        file.write(tflite_model)
