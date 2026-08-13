from unittest.mock import MagicMock

import keras.layers as keras_layers
from persian_expense_classifier.preprocessing.input_standardize import STANDARDIZERS
import pytest
from pytest_mock import MockerFixture
import tensorflow as tf
import fake_test_data

from persian_expense_classifier.models.cnn_expense_classifier_model import (
    CnnExpenseClassifierModel,
)


class TestTextVectorizationLayer:

    @pytest.fixture(autouse=True)
    def mock_load_train_config(self, mocker: MockerFixture):
        fake_config = fake_test_data.fake_train_config_dic
        return mocker.patch(
            "persian_expense_classifier.models.cnn_expense_classifier_model.load_train_config",
            return_value=fake_config,
        )

    def test_should_return_expected_layer_type(
        self, mocker: MockerFixture, mock_load_train_config: MagicMock
    ):
        # act
        result = CnnExpenseClassifierModel().vectorizer()

        # assert
        assert isinstance(result, tf.keras.layers.Layer)

    def test_call_expected_module_to_load_config_with_expected_arg(
        self, mocker: MockerFixture, mock_load_train_config: MagicMock
    ):

        # act
        CnnExpenseClassifierModel().vectorizer()

        # assert
        mock_load_train_config.assert_called_once()

    def test_has_expected_parameters_based_on_train_config(self, mocker: MockerFixture):
        fake_config = fake_test_data.fake_train_config_dic

        mocker.patch(
            "persian_expense_classifier.models.cnn_expense_classifier_model.load_train_config",
            return_value=fake_config,
        )

        # act
        vectorizer = CnnExpenseClassifierModel().vectorizer()

        # assert
        vectorizer_conf = fake_config["text_vectorization"]

        assert vectorizer._max_tokens == vectorizer_conf["max_tokens"]
        assert (
            vectorizer._output_sequence_length
            == vectorizer_conf["output_sequence_length"]
        )
        assert vectorizer._output_mode == vectorizer_conf["output_mode"]
        assert vectorizer._standardize == STANDARDIZERS[vectorizer_conf["standardize"]]


class TestCreateModel:

    @pytest.fixture(autouse=True)
    def fake_vectorizer(self) -> keras_layers.TextVectorization:
        vectorizer = keras_layers.TextVectorization(
            vocabulary=["this", "is", "a", "test"],
            output_sequence_length=20,
            max_tokens=1000,
        )
        return vectorizer

    def test_should_return_keras_model_object(
        self, fake_vectorizer, mocker: MockerFixture
    ):
        # arrange

        # act
        result = CnnExpenseClassifierModel().create_model(fake_vectorizer)

        # assert
        assert isinstance(result, tf.keras.Model)

    def test_has_expected_layers(self, fake_vectorizer, mocker: MockerFixture):
        # arrange

        # act
        model = CnnExpenseClassifierModel().create_model(fake_vectorizer)

        # assert
        layers = model.layers
        assert [type(x) for x in layers] == [
            keras_layers.InputLayer,
            keras_layers.TextVectorization,
            keras_layers.Embedding,
            #
            keras_layers.Conv1D,
            #
            keras_layers.GlobalAveragePooling1D,
            #
            keras_layers.Dense,
            keras_layers.Dense,
        ]
