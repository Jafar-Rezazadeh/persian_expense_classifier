from typing import cast

from keras.callbacks import History

from persian_expense_classifier.data.data_loader import load_data
from sklearn.model_selection import train_test_split
from keras import Model
from keras.losses import SparseCategoricalCrossentropy
from persian_expense_classifier.utils.load_config import load_train_config


class Trainer:

    def __init__(self, model: Model):
        config = load_train_config()
        self.train_config = config["train"]
        self.model: Model = model

    def _loadTrainTestSplitData(self):

        X, Y = load_data()
        return train_test_split(X, Y, test_size=0.2, shuffle=True)

    def _compileModel(self):
        self.model.compile(
            optimizer=self.train_config["optimizer"],
            loss=self.train_config["loss"],
            metrics=self.train_config["metrics"],
        )
        print(self.model.summary())

    def _fit(self, x_train, x_test, y_train, y_test, epochs):

        return self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
        )

    def train(self, epochs: int = 20):

        x_train, x_test, y_train, y_test = self._loadTrainTestSplitData()

        self._compileModel()

        history = self._fit(x_train, x_test, y_train, y_test, epochs)

        return history
