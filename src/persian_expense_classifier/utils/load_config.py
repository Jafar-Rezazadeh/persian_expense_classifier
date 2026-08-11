from pathlib import Path

import yaml


def load_train_config():
    ROOT = Path(__file__).resolve().parents[3]

    with open(ROOT / "config/train_config.yml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
