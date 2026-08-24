from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from persian_expense_classifier.exports.model_exporter import (
    save_keras_model,
    save_tflite_model,
)


@pytest.fixture(autouse=True)
def fake_model():
    fake_model = MagicMock()
    fake_model.save = MagicMock()
    return fake_model


class TestSaveKerasModel:

    def test_should_call_expected_functionality_with_expected_arg(self, fake_model):
        # arrange
        path = Path("test_path")
        model_name = "test"

        # act
        save_keras_model(fake_model, path, model_name)

        # assert
        path_arg = fake_model.save.call_args_list[0].args[0]
        assert str(path_arg).endswith(str(path / f"{model_name}.keras"))


class TestSaveTfLiteModel:

    @pytest.fixture(autouse=True)
    def fake_converter(self, mocker: MockerFixture):

        fake_tflite_model = b"this is a test"

        fake_converter = MagicMock()
        fake_converter.convert = MagicMock()
        fake_converter.convert.return_value = fake_tflite_model

        mocker.patch(
            "persian_expense_classifier.exports.model_exporter.tf.lite.TFLiteConverter.from_keras_model",
            return_value=fake_converter,
        )
        return fake_converter

    @pytest.fixture(autouse=True)
    def fake_open(self, mocker: MockerFixture):

        mock_open = mocker.patch(
            "persian_expense_classifier.exports.model_exporter.open",
            mocker.mock_open(),
        )
        return mock_open

    def test_should_create_tflite_converter_using_given_model(
        self, fake_model, fake_converter
    ):
        # arrange
        path = Path("test/path")
        model_name = "test"

        # act
        save_tflite_model(fake_model, path, model_name)

        # assert
        fake_converter.convert.assert_called_once()

    def test_should_open_the_tflite_file(self, fake_model, fake_open):
        # arrange
        path = Path("test")
        model_name = "test"

        # act
        save_tflite_model(fake_model, path, model_name)

        # assert

        path_arg = fake_open.call_args_list[0].args[0]
        mode_arg = fake_open.call_args_list[0].args[1]
        assert str(path_arg).endswith(str(path / f"{model_name}.tflite"))
        assert mode_arg == "wb"
