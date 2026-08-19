import numpy as np

train_config_dic = {
    "text_vectorization": {
        "max_tokens": 7000,
        "output_sequence_length": 17,
        "output_mode": "int",
        "standardize": "expense_standardize",
    },
    "cnn_model": {
        "embedding": {"output_dim": 2},
        "conv1D": {
            "filters": 3,
            "kernel_size": (4,),
            "padding": "same",
            "activation": "relu",
        },
        "dense1": {
            "units": 5,
            "activation": "relu",
        },
        "dense2": {
            "units": 6,
            "activation": "relu",
        },  # number of classes
    },
    "train": {
        "test_size": 0.2,
        "shuffle": True,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
    },
}

load_data = [(["hello how are you"], ["food"]) for _ in range(50)]
