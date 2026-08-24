from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from persian_expense_classifier.exports.vectorization_exporter import (
    export_text_vectorization,
)

fake_vocab = [
    "",
    "[UNK]",
    "NUMBER",
    "مبلغ",
    "تومن",
    "هزار",
    "هزینه",
    "شد",
    "به",
]
fake_config = {
    "dtype": "string",
    "max_tokens": 10000,
    "split": "whitespace",
    "output_mode": "int",
    "output_sequence_length": 20,
}


@pytest.fixture
def mock_vectorizer():
    mock_vectorizer = MagicMock()
    mock_vectorizer.get_config = MagicMock()
    mock_vectorizer.get_config.return_value = fake_config

    mock_vectorizer.get_vocabulary = MagicMock()
    mock_vectorizer.get_vocabulary.return_value = fake_vocab

    return mock_vectorizer


@pytest.fixture(autouse=True)
def mock_open(mocker: MockerFixture):
    return mocker.patch(
        "persian_expense_classifier.exports.vectorization_exporter.open",
        mocker.mock_open(),
    )


@pytest.fixture(autouse=True)
def mock_json_dumps(mocker: MockerFixture):
    return mocker.patch(
        "persian_expense_classifier.exports.vectorization_exporter.json.dumps"
    )


def test_should_call_expected_methods_of_vectorizer_to_get_needed_stuff(
    mock_vectorizer,
):
    # arrange

    # act
    export_text_vectorization(mock_vectorizer)

    # assert
    mock_vectorizer.get_config.assert_called_once()
    mock_vectorizer.get_vocabulary.assert_called_once()


def test_pop_the_expected_values_from_config(mock_vectorizer, mock_json_dumps):
    # arrange

    # act
    export_text_vectorization(mock_vectorizer)

    # assert
    fake_config: dict = mock_vectorizer.get_config.return_value
    assert "standardize " not in fake_config.keys()


def test_call_json_dumps_with_expected_args(mock_vectorizer, mock_json_dumps):
    # arrange

    # act
    export_text_vectorization(mock_vectorizer)

    # assert
    call_obj = mock_json_dumps.call_args_list[0].args[0]
    call_ensure_ascii = mock_json_dumps.call_args.kwargs["ensure_ascii"]
    call_indent = mock_json_dumps.call_args.kwargs["indent"]
    assert call_obj == (fake_config | {"vocabulary_list": fake_vocab})
    assert call_ensure_ascii == False
    assert call_indent == 4


def test_call_open_with_expected_args(mock_open, mock_vectorizer):
    # arrange
    path = Path("test")
    name = "text_vectorization_config.json"

    # act
    export_text_vectorization(mock_vectorizer, path)

    # assert
    call_path = mock_open.call_args_list[0].args[0]
    call_mode = mock_open.call_args_list[0].args[1]
    call_encoding = mock_open.call_args.kwargs["encoding"]

    assert str(call_path).endswith(str(path / name))
    assert call_mode == "w"
    assert call_encoding == "utf-8"
