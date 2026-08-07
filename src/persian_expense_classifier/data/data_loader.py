from pathlib import Path

import pandas as pd
import numpy as np
import json
from typing import cast
import random


def loadData() -> tuple[np.ndarray, np.ndarray]:

    ROOT = Path(__file__).resolve().parents[3]

    df1 = pd.read_csv(ROOT / "data/raw/expense_dataset_5000.csv")
    df2 = pd.read_csv(ROOT / "data/raw/expense_dataset_10000.csv")
    df3 = pd.read_csv(ROOT / "data/raw/expense_dataset_10000_20words.csv")
    df4 = pd.read_csv(ROOT / "data/raw/expenses_1000_food_included_of_5000.csv")
    df5 = pd.read_csv(ROOT / "data/raw/expenses_dataset_50000.csv")

    with open(ROOT / "data/labels.json") as file:
        labels = cast(dict, json.load(file))

    labels = {int(key): value for key, value in labels.items()}

    if pd.api.types.is_numeric_dtype(df4["label"]):

        df4["label"] = df4["label"].map(
            lambda x: labels.get(int(x)) if pd.notna(x) else labels[4]
        )

    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    x = df.drop(columns=["id", "label"])
    y = df["label"]

    x = x.astype(str).values.flatten()
    y = y.astype(str).values.flatten()

    return x, y
