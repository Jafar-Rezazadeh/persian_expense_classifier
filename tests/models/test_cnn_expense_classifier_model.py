from pytest_mock import MockerFixture
import tensorflow as tf

from persian_expense_classifier.models.cnn_expense_classifier_model import (
    CnnExpenseClassifierModel,
)


class TestTextVectorizationLayer:

    def test_should_return_expected_layer_type(self, mocker: MockerFixture):

        # act
        result = CnnExpenseClassifierModel().vectorizer()

        # assert
        assert isinstance(result, tf.keras.layers.Layer)

    def test_load_config_from_expected_file(self, mocker: MockerFixture):
        # arrange
        # TODO: test the expected load config called

        # act

        # assert
        pass

    def test_has_expected_parameters(self, mocker: MockerFixture):

        # act
        result = CnnExpenseClassifierModel().vectorizer()

        # assert
        print(result._output_sequence_length)
        # assert result.output_shape == (20,)
