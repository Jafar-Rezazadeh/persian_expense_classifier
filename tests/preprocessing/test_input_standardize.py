from pytest_mock import MockerFixture
from persian_expense_classifier.preprocessing.input_standardize import input_standardize


def test_should_lower_the_text(mocker: MockerFixture):
    # arrange
    input = "THIS is a Test"

    # act
    result = input_standardize(input)

    # assert
    assert bytes(result.numpy()).decode() == input.lower()


def test_all_punctuation_should_be_removed(mocker: MockerFixture):
    # arrange
    input = "اشتراک ١٢ ماهه، ۵۰٪ تخفیف!"

    # act
    result = input_standardize(input)

    # assert
    assert bytes(result.numpy()).decode() == "اشتراک NUMBER ماهه NUMBER تخفیف"


def test_replace_any_number_with_number_word(mocker: MockerFixture):
    # arrange
    input = "food for 3000 dollar"

    # act
    result = input_standardize(input)

    # assert
    assert bytes(result.numpy()).decode() == "food for NUMBER dollar"


def test_should_replace_persian_number(mocker: MockerFixture):
    # arrange
    input = "پرداخت ۱۲۳۴ تومان."

    # act
    result = input_standardize(input)

    # assert
    assert bytes(result.numpy()).decode() == "پرداخت NUMBER تومان"


def test_replace_number_which_has_punctuations(mocker: MockerFixture):
    # arrange
    input = "Rent: 1,250.50 USD"

    # act
    result = input_standardize(input)

    # assert
    assert bytes(result.numpy()).decode() == "rent NUMBER usd"
