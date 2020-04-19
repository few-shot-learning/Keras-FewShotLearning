import os
from pathlib import Path

import pandas as pd
from tensorflow.keras.utils import get_file

BASE_PATH = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"


def load_dataframe(dataset_name):
    dataset_path = get_file(
        f"{dataset_name}.zip",
        origin=f"{BASE_PATH}/{dataset_name}.zip",
        extract=True,
        cache_subdir=Path("datasets") / "omniglot",
    )
    dataset_dir = os.path.splitext(dataset_path)[0]
    dataset = pd.DataFrame(columns=["image_name", "alphabet", "label"])
    for root, _, files in os.walk(dataset_dir):
        if files:
            alphabet, label = Path(root).relative_to(dataset_dir).parts
            root = Path(root)
            image_names = [root / file for file in files]
            dataset = dataset.append(
                pd.DataFrame({"image_name": image_names, "alphabet": alphabet, "label": label}), ignore_index=True,
            )

    return dataset


def load_data():
    train_set = load_dataframe("images_background")
    test_set = load_dataframe("images_evaluation")
    return train_set, test_set
