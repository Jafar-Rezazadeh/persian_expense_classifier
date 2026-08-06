import json
from pathlib import Path
import pytest
from pytest_mock import MockerFixture, MockType
from persian_expense_classifier.data.data_loader import loadData
import pandas as pd


def test_loading_data_correct_file(mock_read_csv: MockType, fake_df: pd.DataFrame):
    # act
    loadData()

    # assert
    mock_read_csv.assert_called()


def test_should_convert_int_label_to_string(mocker: MockerFixture):
    # arrange
    df_fake = pd.DataFrame({"id": [0], "label": [4], "text": ["a simple text"]})

    mocker.patch(
        "persian_expense_classifier.data.data_loader.pd.read_csv",
        return_value=df_fake,
    )

    # act
    loadData()

    # assert
    assert type(df_fake["label"].iloc[0]) == str


def test_should_concat_all_loaded_dfs(mocker: MockerFixture):
    # arrange
    df1 = pd.DataFrame({"id": [0], "label": [4], "text": ["df1"]})
    df2 = pd.DataFrame({"id": [1], "label": [6], "text": ["df2"]})
    df3 = pd.DataFrame({"id": [2], "label": [3], "text": ["df3"]})
    df4 = pd.DataFrame({"id": [3], "label": [2], "text": ["df4"]})

    mocker.patch(
        "persian_expense_classifier.data.data_loader.pd.read_csv",
        side_effect=[df1, df2, df3, df4],
    )

    # act
    X, y = loadData()

    # assert
    expected_dfs_text = ["df1", "df2", "df3", "df4"]
    assert X.tolist() == expected_dfs_text


def test_return_expected_result(mocker: MockerFixture, fake_df: pd.DataFrame):
    # act
    X, y = loadData()

    # assert
    assert X[0] == fake_df["text"][0]
    assert y[0] == fake_df["label"][0]
