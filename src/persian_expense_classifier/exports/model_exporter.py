from pathlib import Path

from colorama import Fore
from tensorflow.keras.models import Model
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]


def save_keras_model(model: Model, path: Path, model_name: str):
    try:
        model_path = path / f"{model_name}.keras"
        model.save(ROOT / model_path)
        print(Fore.GREEN, "successfully saved the model in {model_path}")
    except Exception as e:
        print(Fore.RED, "error while saving model:", e)


def save_tflite_model(model: Model, path: Path, model_name: str):

    path = path / f"{model_name}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(ROOT / path, "wb") as file:
        file.write(tflite_model)
