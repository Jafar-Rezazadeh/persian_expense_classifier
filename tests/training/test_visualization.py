from unittest.mock import MagicMock, call
from keras.callbacks import History
import pytest
from pytest_mock import MockerFixture
from persian_expense_classifier.training.visualization import (
    plot_loss,
    plot_accuracy,
)

fake_history = History()
fake_history.history = {
    "loss": [0.8, 0.5, 0.4],
    "val_loss": [0.8, 0.5, 0.4],
    "accuracy": [0.1, 0.2, 0.3],
    "val_accuracy": [0.1, 0.2, 0.3],
}


@pytest.fixture(autouse=True)
def fake_plt(mocker: MockerFixture):
    mock_plt = MagicMock()
    mock_plt.figure = MagicMock()
    mock_plt.plot = MagicMock()
    mock_plt.xlabel = MagicMock()
    mock_plt.ylabel = MagicMock()
    mock_plt.legend = MagicMock()
    mock_plt.show = MagicMock()

    return mocker.patch(
        "persian_expense_classifier.training.visualization.plt",
        return_value=mock_plt,
    )


class TestPlotLoss:

    def test_should_call_expected_method_of_pyplot(self, fake_plt):
        # arrange

        # act
        plot_loss(fake_history)

        # assert
        fake_plt.figure.assert_called_once()
        fake_plt.plot.assert_has_calls(
            [
                call(fake_history.history["loss"], label="loss"),
                call(fake_history.history["val_loss"], label="val_loss"),
            ]
        )
        fake_plt.xlabel.assert_called_once_with("epochs")
        fake_plt.ylabel.assert_called_once_with("loss")
        fake_plt.legend.assert_called_once()
        fake_plt.show.assert_called_once()


class TestPlotAccuracy:

    def test_should_call_expected_plt_methods_with_expected_args(self, fake_plt):
        # arrange

        # act
        plot_accuracy(fake_history)

        # assert
        fake_plt.figure.assert_called_once()
        fake_plt.plot.assert_has_calls(
            [
                call(fake_history.history["accuracy"], label="accuracy"),
                call(fake_history.history["val_accuracy"], label="val_accuracy"),
            ]
        )
        fake_plt.xlabel.assert_called_once_with("epochs")
        fake_plt.ylabel.assert_called_once_with("accuracy")
        fake_plt.legend.assert_called_once()
        fake_plt.show.assert_called_once()
