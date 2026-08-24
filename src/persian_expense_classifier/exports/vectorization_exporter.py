from pathlib import Path

from keras.layers import TextVectorization
import json


def export_text_vectorization(vectorizer: TextVectorization, path: Path = Path("")):

    ROOT = Path(__file__).parents[3]
    save_path = (ROOT / path / "text_vectorization_config.json").resolve()

    config = vectorizer.get_config()

    if "standardize" in config.keys():
        config.pop("standardize")

    vocabulary = vectorizer.get_vocabulary()

    full_config = config | {"vocabulary_list": vocabulary}

    jsonString = json.dumps(
        full_config,
        ensure_ascii=False,
        indent=4,
    )

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(jsonString)
