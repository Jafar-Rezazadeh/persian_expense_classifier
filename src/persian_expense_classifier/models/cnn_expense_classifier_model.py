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
        self.train_config = load_train_config()

    def vectorizer(self) -> TextVectorization:

        vectorizer_config = self.train_config["text_vectorization"]

        standardizer = STANDARDIZERS[vectorizer_config["standardize"]]

        return TextVectorization(
            max_tokens=vectorizer_config["max_tokens"],
            output_sequence_length=vectorizer_config["output_sequence_length"],
            output_mode=vectorizer_config["output_mode"],
            standardize=standardizer,  # type: ignore
        )

    def build_model(self, text_vectorizer: TextVectorization) -> tf.keras.Model:

        cnn_conf = self.train_config["cnn_model"]

        input = Input(
            shape=(self.train_config["text_vectorization"]["output_sequence_length"],),
            dtype=tf.int32,
        )

        x = Embedding(
            input_dim=text_vectorizer.vocabulary_size(),
            output_dim=cnn_conf["embedding"]["output_dim"],
        )(input)

        x = Conv1D(
            filters=cnn_conf["conv1D"]["filters"],
            kernel_size=cnn_conf["conv1D"]["kernel_size"],
            padding=cnn_conf["conv1D"]["padding"],
            activation=cnn_conf["conv1D"]["activation"],
        )(x)

        x = GlobalAveragePooling1D()(x)

        x = Dense(
            cnn_conf["dense1"]["units"],
            activation=cnn_conf["dense1"]["activation"],
        )(x)

        output = Dense(
            cnn_conf["dense2"]["units"],
            activation=cnn_conf["dense2"]["activation"],
        )(x)

        return tf.keras.Model(input, output)
