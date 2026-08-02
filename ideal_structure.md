## ideal project structure

```persian_expense_classifier/
│
├── pyproject.toml
├── environment.yml
├── README.md
├── .gitignore
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── labels.json
│
├── configs/
│ └── train.yaml
│
├── models/
│ ├── best.keras
│ └── final.keras
│
├── scripts/
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── src/
│ └── persian_expense_classifier/
│ ├── **init**.py
│ │
│ ├── data/
│ │ ├── **init**.py
│ │ ├── loader.py
│ │ ├── preprocessing.py
│ │ └── dataset.py
│ │
│ ├── models/
│ │ ├── **init**.py
│ │ └── classifier.py
│ │
│ ├── training/
│ │ ├── **init**.py
│ │ ├── trainer.py
│ │ ├── callbacks.py
│ │ └── metrics.py
│ │
│ ├── inference/
│ │ ├── **init**.py
│ │ └── predictor.py
│ │
│ └── utils/
│ ├── **init**.py
│ ├── io.py
│ ├── logger.py
│ └── seed.py
│
└── tests/
```
