import json
from pathlib import Path
import pytest
from pytest_mock import MockerFixture, MockType
from persian_expense_classifier.data.data_loader import load_data
import pandas as pd


def test_load_expected_csv_files(mock_read_csv: MockType, fake_df: pd.DataFrame):
    # act
    load_data()

    # assert
    paths = [call.args[0] for call in mock_read_csv.call_args_list]

    assert Path(r"data/raw/expense_dataset_5000.csv").resolve() in paths
    assert Path(r"data/raw/expense_dataset_10000_20words.csv").resolve() in paths
    assert Path(r"data/raw/expense_dataset_10000.csv").resolve() in paths
    assert Path(r"data/raw/expenses_1000_food_included_of_5000.csv").resolve() in paths
    assert Path(r"data/raw/expenses_dataset_50000.csv").resolve() in paths


def test_should_convert_int_label_to_string(mocker: MockerFixture):
    # arrange
    df_fake = pd.DataFrame({"id": [0], "label": [4], "text": ["a simple text"]})

    mocker.patch(
        "persian_expense_classifier.data.data_loader.pd.read_csv",
        return_value=df_fake,
    )

    # act
    load_data()

    # assert
    assert type(df_fake["label"].iloc[0]) == str


def test_should_concat_all_loaded_dfs(mocker: MockerFixture):
    # arrange
    df1 = pd.DataFrame({"id": [0], "label": [4], "text": ["df1"]})
    df2 = pd.DataFrame({"id": [1], "label": [6], "text": ["df2"]})
    df3 = pd.DataFrame({"id": [2], "label": [3], "text": ["df3"]})
    df4 = pd.DataFrame({"id": [3], "label": [2], "text": ["df4"]})
    df5 = pd.DataFrame({"id": [4], "label": [5], "text": ["df5"]})

    mocker.patch(
        "persian_expense_classifier.data.data_loader.pd.read_csv",
        side_effect=[df1, df2, df3, df4, df5],
    )

    # act
    X, y = load_data()

    # assert
    expected_dfs_text = ["df1", "df2", "df3", "df4", "df5"]
    assert X.tolist() == expected_dfs_text


def test_return_expected_result(mocker: MockerFixture, fake_df: pd.DataFrame):
    # act
    X, y = load_data()

    # assert
    assert X[0] == fake_df["text"][0]
    assert y[0] == fake_df["label"][0]
