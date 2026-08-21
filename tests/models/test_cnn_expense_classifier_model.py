from typing import cast
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


@pytest.fixture(autouse=True)
def mock_load_train_config(mocker: MockerFixture):
    fake_config = fake_test_data.train_config_dic
    return mocker.patch(
        "persian_expense_classifier.models.cnn_expense_classifier_model.load_train_config",
        return_value=fake_config,
    )


class TestTextVectorizationLayer:

    def test_should_return_expected_layer_type(self):
        # act
        result = CnnExpenseClassifierModel().vectorizer()

        # assert
        assert isinstance(result, tf.keras.layers.Layer)

    def test_call_expected_module_to_load_config_with_expected_arg(
        self, mock_load_train_config: MagicMock
    ):

        # act
        CnnExpenseClassifierModel().vectorizer()

        # assert
        mock_load_train_config.assert_called_once()

    def test_has_expected_parameters_based_on_train_config(self, mocker: MockerFixture):
        fake_config = fake_test_data.train_config_dic

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


class TestBuildModel:

    @pytest.fixture(autouse=True)
    def fake_vectorizer(self) -> keras_layers.TextVectorization:
        vectorizer = keras_layers.TextVectorization(
            vocabulary=["this", "is", "a", "test"],
            output_sequence_length=20,
            max_tokens=1000,
        )
        return vectorizer

    def test_should_return_keras_model_object(self, fake_vectorizer):
        # arrange

        # act
        result = CnnExpenseClassifierModel().build_model(fake_vectorizer)

        # assert
        assert isinstance(result, tf.keras.Model)

    def test_has_expected_layers(self, fake_vectorizer):
        # arrange

        # act
        model = CnnExpenseClassifierModel().build_model(fake_vectorizer)

        # assert
        layers = model.layers
        assert [type(x) for x in layers] == [
            keras_layers.InputLayer,
            keras_layers.Embedding,
            #
            keras_layers.Conv1D,
            #
            keras_layers.GlobalAveragePooling1D,
            #
            keras_layers.Dense,
            keras_layers.Dense,
        ]

    def test_should_each_layer_has_expected_hyper_params_based_on_train_config(
        self, fake_vectorizer, mocker: MockerFixture
    ):
        # arrange
        fake_config = fake_test_data.train_config_dic
        mocker.patch(
            "persian_expense_classifier.models.cnn_expense_classifier_model.load_train_config",
            return_value=fake_config,
        )

        # act
        model = CnnExpenseClassifierModel().build_model(fake_vectorizer)

        # assert
        cnn_conf = fake_config["cnn_model"]

        input_layer = model.layers[0]

        assert input_layer.get_config()["batch_input_shape"] == (
            None,
            fake_config["text_vectorization"]["output_sequence_length"],
        )

        embedding = next(
            x for x in model.layers if isinstance(x, keras_layers.Embedding)
        )
        embedding_conf = embedding.get_config()
        assert embedding_conf["output_dim"] == cnn_conf["embedding"]["output_dim"]
        assert embedding_conf["input_dim"] == fake_vectorizer.vocabulary_size()

        # conv1d
        conv = next(x for x in model.layers if isinstance(x, keras_layers.Conv1D))
        conv_conf = conv.get_config()
        conv_expected = cnn_conf["conv1D"]
        assert conv_conf["filters"] == conv_expected["filters"]
        assert conv_conf["kernel_size"] == conv_expected["kernel_size"]
        assert conv_conf["padding"] == conv_expected["padding"]
        assert conv_conf["activation"] == conv_expected["activation"]

        # dense layers
        # first dense
        dense1 = next(
            x
            for x in model.layers
            if isinstance(x, keras_layers.Dense)
            and x.get_config()["units"] == cnn_conf["dense1"]["units"]
        )
        dense1_conf = dense1.get_config()
        assert dense1_conf["units"] == cnn_conf["dense1"]["units"]
        assert dense1_conf["activation"] == cnn_conf["dense1"]["activation"]

        # final dense
        dense2 = next(
            x
            for x in model.layers
            if isinstance(x, keras_layers.Dense)
            and x.get_config()["units"] == cnn_conf["dense2"]["units"]
        )
        dense2_conf = dense2.get_config()
        assert dense2_conf["units"] == cnn_conf["dense2"]["units"]
        assert dense2_conf["activation"] == cnn_conf["dense2"]["activation"]
