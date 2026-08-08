import tensorflow as tf
from keras.layers import (
    TextVectorization,
    Conv1D,
    MaxPool1D,
    GlobalAveragePooling1D,
    Dense,
    Embedding,
    Input,
)


# TODO: Implement CNN model
class CnnExpenseClassifierModel:
    def __init__(self):
        return

    def vectorizer(self) -> TextVectorization:
        return TextVectorization()
