import tensorflow as tf


from persian_expense_classifier.utils.load_config import load_train_config


class Trainer:

    def __init__(self, model: tf.keras.Model):
        config = load_train_config()
        self.train_config = config["train"]
        self.model = model

    def _compileModel(self):
        self.model.compile(
            optimizer=self.train_config["optimizer"],
            loss=self.train_config["loss"],
            metrics=self.train_config["metrics"],
        )
        print(self.model.summary())

    def train(
        self, x_train, x_test, y_train, y_test, epochs: int = 20
    ) -> tf.keras.callbacks.History:

        self._compileModel()

        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
        )

        return history
