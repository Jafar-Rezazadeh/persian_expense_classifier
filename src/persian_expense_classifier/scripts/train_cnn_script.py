from pathlib import Path

from persian_expense_classifier.data.data_loader import DataLoader
from persian_expense_classifier.models.cnn_expense_classifier_model import (
    CnnExpenseClassifierModel,
)
from persian_expense_classifier.training.trainer import Trainer
from persian_expense_classifier.training.visualization import plot_accuracy, plot_loss
from persian_expense_classifier.exports.model_exporter import (
    save_keras_model,
    save_tflite_model,
)
from sklearn.model_selection import train_test_split
from persian_expense_classifier.preprocessing.custom_label_encoder import (
    CustomLabelEncoder,
)
import numpy as np

dataLoader = DataLoader()
labelEncoder = CustomLabelEncoder()
model = CnnExpenseClassifierModel()
vectorizer = model.vectorizer()

X, y = dataLoader.load_data()


x_train, x_test, y_train, y_test = train_test_split(X, y)


# preprocessing the label
y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.transform(y_test)


# creating vocabulary using data
vectorizer.adapt(x_train)

model = model.build_model(vectorizer)

trainer = Trainer(model)


# training
history = trainer.train(
    x_train,
    x_test,
    y_train,
    y_test,
)

# visualization
plot_loss(history)
plot_accuracy(history)


# saving the trained model
path = Path("artifacts/models")
save_keras_model(model, path)
save_tflite_model(model, path)
