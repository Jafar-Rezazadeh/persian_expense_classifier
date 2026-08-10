from pytest_mock import MockerFixture
from persian_expense_classifier.utils.load_config import load_yml_config


class TestLoadYmlConfig:
    def mockOpen(self, mocker):

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

    def test_should_call_open_with_expected_args(self, mocker: MockerFixture):
        # arrange
        path_to_call = "path.yaml"

        mock = self.mockOpen(mocker)

        # act
        load_yml_config(path_to_call)

        # assert
        mock.assert_called_once()
        assert str(mock.call_args.args[0]).endswith(path_to_call)
        assert mock.call_args.args[1] == "r"
        assert mock.call_args.kwargs["encoding"] == "utf-8"

    def test_should_call_expected_module_to_load_data(self, mocker: MockerFixture):
        # arrange
        self.mockOpen(mocker)

        mockResult = mocker.patch(
            "persian_expense_classifier.utils.load_config.yaml.safe_load",
            return_value={},
        )

        # act
        load_yml_config("path.yaml")

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
        result = load_yml_config("path.yaml")

        # assert
        assert type(result) == dict
        assert result["test"]["field1"] == "value1"
