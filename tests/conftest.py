import pandas as pd
import pytest
from pytest_mock import MockerFixture


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
