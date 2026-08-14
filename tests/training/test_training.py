from typing import cast
from unittest.mock import ANY, MagicMock
import pytest
from pytest_mock import MockerFixture
from persian_expense_classifier.training.trainer import Trainer
from tests import fake_test_data
from keras.callbacks import History

fake_training_config = fake_test_data.train_config_dic


@pytest.fixture(autouse=True)
def mock_model():
    mock_model = MagicMock()
    mock_model.compile = MagicMock()
    mock_model.fit = MagicMock()
    mock_model.fit.return_value = History()
    mock_model.summary = MagicMock()

    return mock_model


@pytest.fixture(autouse=True)
def mock_load_train_config(mocker: MockerFixture):
    mock = mocker.patch(
        "persian_expense_classifier.training.trainer.load_train_config",
        return_value=fake_training_config,
    )
    return mock


@pytest.fixture(autouse=True)
def mock_load_data(mocker: MockerFixture):
    mock = mocker.patch(
        "persian_expense_classifier.training.trainer.load_data",
        return_value=fake_test_data.load_data,
    )
    return mock


# ------------
def test_should_load_training_config_on_init(
    mock_load_train_config, mock_model, mocker: MockerFixture
):
    # arrange

    # act
    Trainer(mock_model)

    # assert
    mock_load_train_config.assert_called_once()


class TestTrainMethod:

    def test_should_called_load_data(
        self, mock_model, mock_load_data, mocker: MockerFixture
    ):
        # arrange
        trainer = Trainer(mock_model)

        # act
        trainer.train()

        # assert
        mock_load_data.assert_called_once()

    def test_should_call_model_compile_with_train_config_params_then_summary(
        self, mock_model, mocker: MockerFixture
    ):
        # arrange
        trainer = Trainer(mock_model)

        # act
        trainer.train()

        # assert
        mock_compile = cast(MagicMock, mock_model.compile)
        mock_summary = cast(MagicMock, mock_model.summary)

        mock_compile.assert_called_once_with(
            optimizer=fake_training_config["train"]["optimizer"],
            loss=fake_training_config["train"]["loss"],
            metrics=fake_training_config["train"]["metrics"],
        )
        mock_summary.assert_called_once()

    def test_should_call_fit_with_expected_args(
        self, mock_model, mocker: MockerFixture
    ):
        # arrange
        epochs = 580
        trainer = Trainer(mock_model)

        # act
        trainer.train(epochs)

        # assert
        mock_fit = cast(MagicMock, mock_model.fit)
        args, kwargs = mock_fit.call_args
        assert kwargs.get("epochs") == epochs

    def test_should_return_history(self, mock_model, mocker: MockerFixture):
        # arrange
        trainer = Trainer(mock_model)

        # act
        result = trainer.train()

        # assert
        assert isinstance(result, History)
