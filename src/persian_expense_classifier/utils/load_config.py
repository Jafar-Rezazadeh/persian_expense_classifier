from pathlib import Path

import yaml


def load_yml_config(path: str):
    ROOT = Path(__file__).resolve().parents[3]

    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
