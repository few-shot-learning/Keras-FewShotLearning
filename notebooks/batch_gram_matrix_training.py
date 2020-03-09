from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
import yaml
from tensorflow.keras import applications as keras_applications
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keras_fsl.dataframe.operators import ToKShotDataset
from keras_fsl.models import SiameseNets
from keras_fsl.models.layers import Classification, GramMatrix
from keras_fsl.sequences import prediction
from keras_fsl.losses import (
    binary_crossentropy,
    accuracy,
    mean_score_classification_loss,
    min_eigenvalue,
)
from keras_fsl.utils import compose

# tf.config.experimental_run_functions_eagerly(True)

#%% Init data
output_folder = Path("logs") / "batch_gram_training" / datetime.today().strftime("%Y%m%d-%H%M%S")
output_folder.mkdir(parents=True, exist_ok=True)
try:
    shutil.copy(__file__, output_folder / "training_pipeline.py")
except (FileNotFoundError, NameError):
    pass

all_annotations = (
    pd.read_csv("data/annotations/cropped_images.csv")
    .assign(
        day=lambda df: df.image_name.str.slice(3, 11),
        image_name=lambda df: "data/images/cropped_images/" + df.image_name,
        label_one_hot=lambda df: pd.get_dummies(df.label).values.tolist(),
        crop_y=lambda df: df.y1,
        crop_x=lambda df: df.x1,
        crop_height=lambda df: df.y2 - df.y1,
        crop_width=lambda df: df.x2 - df.x1,
        crop_window=lambda df: df[["crop_y", "crop_x", "crop_height", "crop_width"]].values.tolist(),
    )
    .filter(items=["day", "image_name", "crop_window", "label", "label_one_hot"])
)
train_val_test_split = yaml.safe_load(open("data/annotations/cropped_images_split.yaml"))
train_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split["train_set_dates"])]
val_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split["val_set_dates"])]
test_set = all_annotations.loc[lambda df: df.day.isin(train_val_test_split["test_set_dates"])].reset_index(drop=True)

#%% Init model
branch_model_name = "MobileNet"
siamese_nets = SiameseNets(
    branch_model={
        "name": branch_model_name,
        "init": {"include_top": False, "input_shape": (224, 224, 3), "pooling": "avg"},
    },
    head_model={
        "name": "MixedNorms",
        "init": {
            "norms": [
                lambda x: 1 - tf.nn.l2_normalize(x[0]) * tf.nn.l2_normalize(x[1]),
                lambda x: tf.math.abs(x[0] - x[1]),
                lambda x: tf.nn.softmax(tf.math.abs(x[0] - x[1])),
                lambda x: tf.square(x[0] - x[1]),
            ],
            "use_bias": False,
        },
    },
)

model = Sequential([siamese_nets.get_layer("branch_model"), GramMatrix(kernel=siamese_nets.get_layer("head_model"))])

#%% Init training
preprocessing = compose(
    partial(tf.cast, dtype=tf.float32),
    tf.image.random_flip_left_right,
    tf.image.random_flip_up_down,
    partial(tf.image.resize_with_crop_or_pad, target_height=224, target_width=224),
    partial(getattr(keras_applications, branch_model_name.lower()).preprocess_input, data_format="channels_last"),
)

callbacks = [
    TensorBoard(output_folder, write_images=True, histogram_freq=1),
    ModelCheckpoint(
        str(output_folder / "kernel_loss_best_loss_weights.h5"), save_best_only=True, save_weights_only=True,
    ),
    ModelCheckpoint(
        str(output_folder / "kernel_loss_best_accuracy_weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        monitor="val__accuracy",
    ),
    ReduceLROnPlateau(),
]
train_dataset = (
    train_set
    .pipe(ToKShotDataset(k_shot=8))
    .map(lambda annotation: (preprocessing(annotation['image']), tf.cast(annotation['label_one_hot'], tf.float32)))
)
val_dataset = (
    val_set
    .pipe(ToKShotDataset(k_shot=8))
    .map(lambda annotation: (preprocessing(annotation['image']), tf.cast(annotation['label_one_hot'], tf.float32)))
)

#%% Train model with loss on kernel
siamese_nets.get_layer("branch_model").trainable = False
optimizer = Adam(lr=1e-4)
margin = 0.05
batch_size = 64
model.compile(
    optimizer=optimizer,
    loss=binary_crossentropy(margin),
    metrics=[binary_crossentropy(0.0), accuracy(margin), mean_score_classification_loss, min_eigenvalue],
)
model.fit(
    train_dataset.batch(batch_size).repeat(),
    steps_per_epoch=len(train_set) // batch_size,
    validation_data=val_dataset.batch(batch_size).repeat(),
    validation_steps=len(val_set) // batch_size,
    initial_epoch=0,
    epochs=5,
    callbacks=callbacks,
)

siamese_nets.get_layer("branch_model").trainable = True
optimizer = Adam(lr=1e-5)
model.compile(
    optimizer=optimizer,
    loss=binary_crossentropy(margin),
    metrics=[binary_crossentropy(0.0), accuracy(margin), mean_score_classification_loss, min_eigenvalue],
)
model.fit(
    train_dataset.batch(batch_size).repeat(),
    steps_per_epoch=len(train_set) // batch_size,
    validation_data=val_dataset.batch(batch_size).repeat(),
    validation_steps=len(val_set) // batch_size,
    initial_epoch=5,
    epochs=20,
    callbacks=callbacks,
)

model.save(output_folder / "final_model.h5")

#%% Eval on test set
k_shot = 1
n_way = 5
n_episode = 100
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_set.to_dict('list'))
    .map(lambda annotation: tf.io.decode_and_crop_jpeg(
        contents=tf.io.read_file(annotation['image_name']),
        crop_window=annotation['crop_window'],
        channels=3,
    ))
    .map(preprocessing)
    .batch(64)
)

embeddings = siamese_nets.get_layer("branch_model").predict(test_dataset, verbose=1)

scores = []
for _ in range(n_episode):
    selected_labels = np.random.choice(test_set.label.unique(), size=n_way, replace=True)
    support_set = (
        test_set
        .loc[lambda df: df.label.isin(selected_labels)]
        .groupby("label")
        .apply(lambda group: group.sample(k_shot))
        .reset_index("label", drop=True)
    )
    query_set = (
        test_set
        .loc[lambda df: df.label.isin(selected_labels)]
        .loc[lambda df: ~df.index.isin(support_set.index)]
    )
    support_set_embeddings = embeddings[support_set.index]
    query_set_embeddings = embeddings[query_set.index]
    test_sequence = prediction.pairs.ProductSequence(
        support_images_array=support_set_embeddings,
        query_images_array=query_set_embeddings,
        support_labels=support_set.label.values,
        query_labels=query_set.label.values,
    )
    scores += [
        (
            test_sequence.pairs_indexes.assign(
                score=siamese_nets.get_layer("head_model").predict_generator(test_sequence, verbose=1)
            )
            .groupby("query_index")
            .apply(
                lambda group: (
                    group.sort_values("score", ascending=False)
                    .assign(
                        average_precision=lambda df: df.target.expanding().mean(),
                        good_prediction=lambda df: df.target.iloc[0],
                    )
                    .loc[lambda df: df.target]
                    .agg("mean")
                )
            )
            .agg("mean")
        )
    ]

scores = pd.DataFrame(scores)[["score", "average_precision", "good_prediction"]]
plt.clf()
scores.boxplot()
plt.savefig(output_folder / "scores_boxplot.png")
plt.clf()
scores.good_prediction.hist()
plt.savefig(output_folder / "scores_good_predictions.png")
scores.to_csv(output_folder / "scores.csv", index=False)
