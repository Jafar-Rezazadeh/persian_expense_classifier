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
