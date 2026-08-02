# %% imports
import os
import json
import re
from sklearn.model_selection import train_test_split
from persian_expense_classifier.data.data_loader import loadData
from persian_expense_classifier.preprocessing.tokenizer_and_padder import (
    CustomTokenizerAndPadded,
)
from persian_expense_classifier.preprocessing.custom_label_encoder import (
    CustomLabelEncoder,
)
from persian_expense_classifier.models.simple_expense_classifier_model import (
    SimpleExpanseClassifierModel,
)

maxLen = 20
num_words = 1000

# %% loadData
x, y = loadData()

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(xTrain.shape)


# %% normalizing input
def normalize_text(text: str) -> str:
    text = text.lower()

    # replace numbers
    text = re.sub(r"[0-9۰-۹]+", "NUM", text)

    # normalize currency
    text = text.replace("تومن", "تومان")

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


xTrain = [normalize_text(x) for x in xTrain]
xTest = [normalize_text(x) for x in xTest]

print(xTrain[0])


# %% prepare input

textPreparer = CustomTokenizerAndPadded(
    num_words=num_words, maxLen=maxLen, texts=xTrain
)

xTrainPad = textPreparer.convertTextsToPaddedSequences(xTrain)
xTestPad = textPreparer.convertTextsToPaddedSequences(xTest)


print(xTrainPad.shape)
print(xTestPad.shape)

# %% encoding labels
customLabelEncoder = CustomLabelEncoder()

yTrainEnc = customLabelEncoder.fit_transform(yTrain)
yTestEnc = customLabelEncoder.transform(yTest)


# %% building and training

num_classes = len(customLabelEncoder.classes_)

model = SimpleExpanseClassifierModel(
    num_words=num_words, maxLen=maxLen, num_classes=num_classes
)

model.fit(
    xTrainPad,
    yTrainEnc,
    validation_data=(xTestPad, yTestEnc),
    epochs=7,
    batch_size=24,
)

# %% evaluating
loss, acc = model.evaluate(xTestPad, yTestEnc)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# %% predicting on new test

sample_texts = ["400 کرایه تاکسی", "250 شام", "3 میلیون اجاره"]

preparedInput = textPreparer.convertTextsToPaddedSequences(sample_texts)

pred_probs = model.predict(preparedInput)
pred_ids = pred_probs.argmax(axis=1)

pred_labels = customLabelEncoder.inverse_transform(pred_ids)

print("realTest \n")
for text, label in zip(sample_texts, pred_labels):
    print(text, "->", label)
print("\n-----------")


# %% exporting needed stuff

os.makedirs("build", exist_ok=True)

model.saveModelAsTfLite()
textPreparer.saveTokenizerAsJson()


customLabelEncoder.saveLabelEncoderAsJson()


# exporting config
config = {"max_len": maxLen, "num_words": num_words}

with open("build/config.json", "w") as f:
    json.dump(config, f)
