# %% imports
import pandas as pd
import numpy as np
import json
from typing import cast
import random


# %%
def loadData() -> tuple[np.ndarray, np.ndarray]:
    # %%
    df1 = pd.read_csv("data/raw/expense_dataset_5000.csv")
    df2 = pd.read_csv("data/raw/expense_dataset_10000.csv")
    df3 = pd.read_csv("data/raw/expense_dataset_10000_20words.csv")
    df4 = pd.read_csv("data/raw/expenses_dataset_50000.csv")

    # %% normalizing label of dfs
    with open("data/labels.json") as file:
        labels = cast(dict, json.load(file))

    labels = {int(key): value for key, value in labels.items()}

    if pd.api.types.is_numeric_dtype(df4["label"]):

        df4["label"] = df4["label"].map(
            lambda x: labels.get(int(x)) if pd.notna(x) else labels[4]
        )

    # %% concat the dfs
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    # print(df.describe(include="all"))

    # %% splitting
    x = df.drop(columns=["id", "label"])
    y = df["label"]

    x = x.astype(str).values.flatten()
    y = y.astype(str).values.flatten()

    randIndex = random.randint(0, len(x))

    # print(x.shape, y.shape, x[randIndex], y[randIndex])

    # %%
    return x, y
