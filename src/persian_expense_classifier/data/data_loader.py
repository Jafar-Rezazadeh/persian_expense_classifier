from pathlib import Path

import pandas as pd
import numpy as np
import json
from typing import cast


class DataLoader:
    def __init__(self) -> None:
        ROOT = Path(__file__).resolve().parents[3]
        self.ROOT = ROOT

        with open(ROOT / "data/labels.json") as file:
            labels = cast(dict, json.load(file))

            self.labels = {int(key): value for key, value in labels.items()}

    def load_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """returns `X` , `y`"""

        df1 = pd.read_csv(self.ROOT / "data/raw/expense_dataset_5000.csv")
        df2 = pd.read_csv(self.ROOT / "data/raw/expense_dataset_10000.csv")
        df3 = pd.read_csv(self.ROOT / "data/raw/expense_dataset_10000_20words.csv")
        df4 = pd.read_csv(
            self.ROOT / "data/raw/expenses_1000_food_included_of_5000.csv"
        )
        df5 = pd.read_csv(self.ROOT / "data/raw/expenses_dataset_50000.csv")

        df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

        self.convert_missing_or_int_labels_to_string_labels(df)

        x = df.drop(columns=["id", "label"])
        y = df["label"]

        x = x.astype(str).values.flatten()
        y = y.astype(str).values.flatten()

        return x, y

    def convert_missing_or_int_labels_to_string_labels(self, df):

        def convert_label(x):
            if pd.isna(x):
                return self.labels[4]

            try:
                return self.labels[int(x)]
            except:
                return x

        df["label"] = df["label"].apply(convert_label)
