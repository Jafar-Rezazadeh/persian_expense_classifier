from persian_expense_classifier.data.data_loader import loadData
from sklearn.model_selection import train_test_split
from keras import Model
from keras.losses import SparseCategoricalCrossentropy
from numpy.typing import NDArray


class Trainer:

    def __init__(self, model: Model):
        self.model: Model = model

    def _loadTrainTestSplitData(self):

        X, Y = loadData()
        return train_test_split(X, Y, test_size=0.2, shuffle=True)

    def _compileModel(self):
        self.model.compile(
            optimizer="Adam",
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        print(self.model.summary())

    def _fit(self, x_train, x_test, y_train, y_test):

        return self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=20,
        )

    def train(self):

        x_train, x_test, y_train, y_test = self._loadTrainTestSplitData()

        self._compileModel()

        history = self._fit(x_train, x_test, y_train, y_test)

        return history
