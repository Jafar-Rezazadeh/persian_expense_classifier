from collections.abc import Callable

from tensorflow import Tensor, strings


def input_standardize(text: Tensor) -> Tensor:

    text = strings.lower(text)

    # Remove Persian/Arabic punctuation
    text = strings.regex_replace(text, r"[،؛؟٪٫٬«»…]", "")

    # Remove ASCII punctuation
    text = strings.regex_replace(text, r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]", "")

    # Replace numbers with NUMBER
    text = strings.regex_replace(text, r"[0-9۰-۹٠-٩]+", " NUMBER ")

    # Collapse whitespace
    text = strings.regex_replace(text, r"\s+", " ")

    return text


STANDARDIZERS = {
    "expense_standardize": input_standardize,
}
