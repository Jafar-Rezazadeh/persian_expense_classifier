from sklearn.preprocessing import LabelEncoder
from numpy.typing import ArrayLike
import numpy as np
import json


class CustomLabelEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit_transform(self, y: ArrayLike) -> ArrayLike:
        return self.label_encoder.fit_transform(y)

    def transform(self, y: ArrayLike) -> ArrayLike:
        return self.label_encoder.transform(y)

    def inverse_transform(self, ids: ArrayLike) -> np.ndarray:
        return self.label_encoder.inverse_transform(ids)

    @property
    def classes_(self):
        return self.label_encoder.classes_

    def saveLabelEncoderAsJson(self):

        with open("build/labels.json", "w", encoding="utf-8") as f:
            json.dump(
                {i: value for i, value in enumerate(self.label_encoder.classes_)},
                f,
                ensure_ascii=False,
            )
            f.close()
