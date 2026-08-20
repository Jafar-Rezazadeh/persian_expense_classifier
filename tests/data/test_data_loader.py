import json
from pathlib import Path
import pytest
from pytest_mock import MockerFixture, MockType
from persian_expense_classifier.data.data_loader import DataLoader
import pandas as pd

fake_labels = {
    "0": "entertainment",
    "1": "food",
    "2": "health",
    "4": "test",
}


@pytest.fixture
def fake_df():
    return pd.DataFrame(
        {
            "id": [0],
            "label": ["food"],
            "text": ["this is a test 500"],
        }
    )


@pytest.fixture(autouse=True)
def mock_read_csv(mocker: MockerFixture, fake_df: pd.DataFrame):
    return mocker.patch(
        "persian_expense_classifier.data.data_loader.pd.read_csv",
        return_value=fake_df,
    )


@pytest.fixture(autouse=True)
def mock_load_labels(mocker: MockerFixture):
    return mocker.patch(
        "persian_expense_classifier.data.data_loader.open",
        mocker.mock_open(read_data=json.dumps(fake_labels)),
    )


@pytest.fixture
def dataLoader():
    dataLoader = DataLoader()
    return dataLoader


# --------


def test_should_load_labels_expected_path(mock_load_labels):

    # arrange

    # act
    DataLoader()

    # assert
    mock_load_labels.assert_called_once()
    arg = mock_load_labels.call_args_list[0].args[0]

    assert "labels.json" in Path(arg).name


def test_load_expected_csv_files(mock_read_csv: MockType, dataLoader):
    # act
    dataLoader.load_data()

    # assert
    paths = [call.args[0] for call in mock_read_csv.call_args_list]

    assert Path(r"data/raw/expense_dataset_5000.csv").resolve() in paths
    assert Path(r"data/raw/expense_dataset_10000_20words.csv").resolve() in paths
    assert Path(r"data/raw/expense_dataset_10000.csv").resolve() in paths
    assert Path(r"data/raw/expenses_1000_food_included_of_5000.csv").resolve() in paths
    assert Path(r"data/raw/expenses_dataset_50000.csv").resolve() in paths


def test_should_concat_all_loaded_dfs(mocker: MockerFixture, dataLoader):
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
    X, y = dataLoader.load_data()

    # assert
    expected_dfs_text = ["df1", "df2", "df3", "df4", "df5"]
    assert X.tolist() == expected_dfs_text


def test_should_convert_missing_or_int_labels_to_string_labels(
    mocker: MockerFixture, dataLoader
):
    # arrange
    df_fake = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "label": [2, "1", float("nan")],
            "text": ["a simple text", "b simple text", "c simple text"],
        },
    )

    mocker.patch(
        "persian_expense_classifier.data.data_loader.pd.read_csv",
        return_value=df_fake,
    )

    # act
    X, y = dataLoader.load_data()

    # assert
    assert y[0] == fake_labels["2"]
    assert y[1] == fake_labels["1"]
    assert y[2] == fake_labels["4"]


def test_return_expected_result(dataLoader, fake_df: pd.DataFrame):
    # act
    X, y = dataLoader.load_data()

    # assert
    assert X[0] == fake_df["text"][0]
    assert y[0] == fake_df["label"][0]
