from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf


class ExpanseClassifier:
    def __init__(self, num_words, maxLen, num_classes):

        model = Sequential(
            [
                Embedding(
                    input_dim=num_words,
                    output_dim=32,
                    input_length=maxLen,
                ),
                GlobalAveragePooling1D(),
                Dense(16, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
        )

        model.summary()

        self.model = model

    def fit(self, xTrainPad, yTrainEnc, validation_data, epochs=7, batch_size=24):
        self.model.fit(
            xTrainPad,
            yTrainEnc,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
        )

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def saveModelAsTfLite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Optional optimization for mobile
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open("build/expense_classifier_model.tflite", "wb") as f:
            f.write(tflite_model)
            f.close()
