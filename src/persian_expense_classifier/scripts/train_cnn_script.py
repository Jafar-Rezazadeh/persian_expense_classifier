from pathlib import Path
from typing import Any, cast

import numpy as np

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
from persian_expense_classifier.exports.vectorization_exporter import (
    export_text_vectorization,
)

dataLoader = DataLoader()
labelEncoder = CustomLabelEncoder()
model = CnnExpenseClassifierModel()
vectorizer = model.vectorizer()

X, y = dataLoader.load_data()


x_train, x_test, y_train, y_test = train_test_split(X, y)

# creating vocabulary using data
vectorizer.adapt(x_train)


# preprocessing the label

x_train = vectorizer(x_train)
x_test = vectorizer(x_test)

y_train = labelEncoder.fit_transform(y_train)
y_test = labelEncoder.transform(y_test)


model = model.build_model(vectorizer)

trainer = Trainer(model)


# training
history = trainer.train(
    x_train,
    x_test,
    y_train,
    y_test,
    # TODO: change the epoch on real training
    epochs=1,
)

# evaluation
eval_result = model.evaluate(x_test, np.asarray(y_test), return_dict=True)
eval_result = cast(dict[str, Any], eval_result)
print("eval_loss:", eval_result["loss"])
print("eval_accuracy:", eval_result["accuracy"])


# visualization
plot_loss(history)
plot_accuracy(history)


# saving the trained model

save_root_path = Path("artifacts/models/cnn")
model_name = "cnn_persian_expense_classifier"

save_keras_model(model, save_root_path, model_name)
save_tflite_model(model, save_root_path, model_name)


# saving the vectorizer

export_text_vectorization(vectorizer, save_root_path)
