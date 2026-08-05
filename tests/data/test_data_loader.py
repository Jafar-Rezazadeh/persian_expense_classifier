import pytest
from persian_expense_classifier.data.data_loader import loadData

# TODO: implement tests


def test_loading_data_from_file():
    X, y = loadData()

    print(X.shape)
    assert len(X) != 0
