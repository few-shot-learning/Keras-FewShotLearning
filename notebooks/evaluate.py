from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import metrics

#%% Init
k_shot = 4
n_way = 16
n_episodes = 100
base_dir = Path()

test_set = (
    pd.read_csv(base_dir / "annotations" / "all_annotations.csv")
    .loc[lambda df: df.split == "test"]
    .groupby("label")
    .filter(lambda group: len(group) > k_shot)
    .assign(label_code=lambda df: df.label.astype("category").cat.codes)
    .reset_index(drop=True)
)

model = load_model("siamese_nets_classifier/1")
encoder, support_layer = model.layers

#%% Compute test_set embeddings
test_dataset = (
    test_set.assign(crop_window=lambda df: df[["crop_y", "crop_x", "crop_height", "crop_width"]].values.tolist())
    .pipe(lambda df: tf.data.Dataset.from_tensor_slices(df.to_dict("list")))
    .map(
        lambda annotation: tf.io.decode_and_crop_jpeg(
            contents=tf.io.read_file(annotation["image_name"]), crop_window=annotation["crop_window"], channels=3,
        )
    )
    .map(lambda x: model.signatures["preprocessing"](x)["output_0"])
    .batch(64)
)
embeddings = encoder.predict(test_dataset, verbose=1)

#%% Generate random k_shot n_way task and compute performance
test_labels = test_set.label.unique()
np.random.seed(0)
scores = []
for _ in range(n_episodes):
    selected_labels = np.random.choice(test_labels, size=n_way, replace=False)
    support_set = (
        test_set.loc[lambda df: df.label.isin(selected_labels)]
        .groupby("label")
        .apply(lambda group: group.sample(k_shot))
        .reset_index("label", drop=True)
    )
    query_set = test_set.loc[lambda df: df.label.isin(selected_labels)].loc[lambda df: ~df.index.isin(support_set.index)]
    support_set_embeddings = tf.convert_to_tensor(embeddings[support_set.index])
    support_set_labels = tf.cast(pd.get_dummies(support_set.label).values, tf.float32)
    support_layer.set_support_set([support_set_embeddings, support_set_labels])
    y_true = pd.get_dummies(query_set.label)
    y_pred = support_layer(embeddings[query_set.index])
    scores += [tf.reduce_mean(metrics.categorical_accuracy(y_true.values, y_pred)).numpy()]
