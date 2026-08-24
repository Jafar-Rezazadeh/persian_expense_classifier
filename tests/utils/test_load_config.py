from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from persian_expense_classifier.utils.load_config import load_train_config


class TestLoadTrainConfig:

    @pytest.fixture(autouse=True)
    def mock_open(self, mocker):

        fake_file = mocker.mock_open(
            read_data="""
                        training:
                            batch_size: 32
                            epochs: 10
                        """,
        )

        mock = mocker.patch(
            "persian_expense_classifier.utils.load_config.open",
            fake_file,
        )
        return mock

    def test_should_call_open_with_expected_args(
        self, mocker: MockerFixture, mock_open: MagicMock
    ):
        # arrange

        # act
        load_train_config()

        # assert
        mock_open.assert_called_once()
        assert str(mock_open.call_args.args[0]).endswith(
            str(Path("config/train_config.yml").resolve())
        )
        assert mock_open.call_args.args[1] == "r"
        assert mock_open.call_args.kwargs["encoding"] == "utf-8"

    def test_should_call_expected_module_to_load_data(self, mocker: MockerFixture):
        # arrange
        mockResult = mocker.patch(
            "persian_expense_classifier.utils.load_config.yaml.safe_load",
            return_value={},
        )

        # act
        load_train_config()

        # assert
        mockResult.assert_called_once()

    def test_should_return_expected_result(self, mocker: MockerFixture):
        # arrange
        fake_file = mocker.mock_open(read_data="""
            
            test:
              field1: value1
              field2: value2
              field3: value3
            
            """)

        mocker.patch("persian_expense_classifier.utils.load_config.open", fake_file)

        # act
        result = load_train_config()

        # assert
        assert type(result) == dict
        assert result["test"]["field1"] == "value1"
