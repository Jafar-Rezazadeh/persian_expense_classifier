import keras.layers as keras_layers
import tensorflow as tf
from persian_expense_classifier.utils.load_config import load_train_config
from persian_expense_classifier.preprocessing.input_standardize import STANDARDIZERS
from keras.layers import (
    TextVectorization,
    Conv1D,
    MaxPool1D,
    GlobalAveragePooling1D,
    Dense,
    Embedding,
    Input,
)


class CnnExpenseClassifierModel:
    def __init__(self):
        return

    def vectorizer(self) -> TextVectorization:

        config = load_train_config()
        vectorizer_config = config["text_vectorization"]

        standardizer = STANDARDIZERS[vectorizer_config["standardize"]]

        return TextVectorization(
            max_tokens=vectorizer_config["max_tokens"],
            output_sequence_length=vectorizer_config["output_sequence_length"],
            output_mode=vectorizer_config["output_mode"],
            standardize=standardizer,  # type: ignore
        )

    def create_model(
        self, text_vectorizer: keras_layers.TextVectorization
    ) -> tf.keras.Model:

        # TODO: get the hyperParameters from configs
        input = keras_layers.Input(shape=(1,), dtype=tf.string)
        x = text_vectorizer(input)
        x = keras_layers.Embedding(
            input_dim=text_vectorizer.vocabulary_size(), output_dim=128
        )(x)

        x = keras_layers.Conv1D(
            filters=64, kernel_size=4, padding="same", activation="relu"
        )(x)

        x = keras_layers.GlobalAveragePooling1D()(x)

        x = keras_layers.Dense(64, activation="relu")(x)

        x = keras_layers.Dense(6, activation="softmax")(x)

        return tf.keras.Model(input, x)
