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
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keras_fsl.dataframe.operators import ToKShotDataset
from keras_fsl.models import SiameseNets
from keras_fsl.models.layers import GramMatrix
from keras_fsl.losses import (
    binary_crossentropy,
    accuracy,
    mean_score_classification_loss,
    min_eigenvalue,
)
from keras_fsl.utils import compose

# tf.config.experimental_run_functions_eagerly(True)

#%% Init model
branch_model_name = "MobileNet"
siamese_nets = SiameseNets(
    branch_model={"name": branch_model_name, "init": {"include_top": False, "input_shape": (224, 224, 3), "pooling": "avg"},},
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
output_folder = Path("logs") / "batch_gram_training" / datetime.today().strftime("%Y%m%d-%H%M%S")
output_folder.mkdir(parents=True, exist_ok=True)
try:
    shutil.copy(__file__, output_folder / "training_pipeline.py")
except (FileNotFoundError, NameError):
    pass

callbacks = [
    TensorBoard(output_folder, write_images=True, histogram_freq=1),
    ModelCheckpoint(str(output_folder / "kernel_loss_best_loss_weights.h5"), save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(),
]


#%% Init data
@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8),))
def preprocessing(input_tensor):
    return compose(
        partial(tf.cast, dtype=tf.float32),
        partial(tf.image.resize_with_pad, target_height=224, target_width=224),
        partial(getattr(keras_applications, branch_model_name.lower()).preprocess_input, data_format="channels_last"),
    )(input_tensor)


data_augmentation = compose(
    tf.image.random_flip_left_right, tf.image.random_flip_up_down, partial(tf.image.random_brightness, max_delta=0.25),
)

train_val_test_split = {
    day: key for key, days in yaml.safe_load(open("data/annotations/train_val_test_split.yaml")).items() for day in days
}
all_annotations = (
    pd.read_csv("data/annotations/cropped_images.csv")
    .assign(
        day=lambda df: df.image_name.str.slice(3, 11),
        image_name=lambda df: "data/images/cropped_images/" + df.image_name,
        crop_y=lambda df: df.y1,
        crop_x=lambda df: df.x1,
        crop_height=lambda df: df.y2 - df.y1,
        crop_width=lambda df: df.x2 - df.x1,
        split=lambda df: df.day.map(train_val_test_split),
    )
    .filter(items=["day", "image_name", "crop_x", "crop_y", "crop_height", "crop_width", "label", "split"])
)

#%% Train model with loss on kernel
siamese_nets.get_layer("branch_model").trainable = False
optimizer = Adam(lr=1e-4)
margin = 0.05
batch_size = 64
datasets = all_annotations.groupby("split").apply(
    lambda group: (
        group.pipe(ToKShotDataset(k_shot=8, preprocessing=compose(preprocessing, data_augmentation))).batch(batch_size).repeat()
    )
)
model.compile(
    optimizer=optimizer,
    loss=binary_crossentropy(margin),
    metrics=[binary_crossentropy(), accuracy(margin), mean_score_classification_loss, min_eigenvalue],
)
model.fit(
    datasets["train"],
    steps_per_epoch=all_annotations.split.value_counts()["train"] // batch_size,
    validation_data=datasets["val"],
    validation_steps=all_annotations.split.value_counts()["val"] // batch_size,
    initial_epoch=0,
    epochs=5,
    callbacks=callbacks,
)

siamese_nets.get_layer("branch_model").trainable = True
optimizer = Adam(lr=1e-5)
model.compile(
    optimizer=optimizer,
    loss=binary_crossentropy(margin),
    metrics=[binary_crossentropy(), accuracy(margin), mean_score_classification_loss, min_eigenvalue],
)
model.fit(
    datasets["train"],
    steps_per_epoch=all_annotations.split.value_counts()["train"] // batch_size,
    validation_data=datasets["val"],
    validation_steps=all_annotations.split.value_counts()["val"] // batch_size,
    initial_epoch=5,
    epochs=20,
    callbacks=callbacks,
)

model.save(output_folder / "final_model.h5")

#%% Export artifacts
siamese_nets.save(output_folder / "final_model.h5")

model.load_weights(str(output_folder / "best_loss.h5"))
siamese_nets.get_layer("branch_model").save(str(output_folder / "branch_model_best_loss.h5"))
siamese_nets.get_layer("head_model").save(str(output_folder / "head_model_best_loss.h5"))


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string), tf.TensorSpec(shape=[None, 4], dtype=tf.int32)))
def decode_and_crop_and_serve(image_name, crop_window):
    # currently not working on GPU, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_and_crop_jpeg(contents=tf.io.read_file(x[0]), crop_window=x[1], channels=3)),
            (image_name, crop_window),
            dtype=tf.float32,
        )
    return siamese_nets.get_layer("branch_model")(input_tensor)


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string),))
def decode_and_serve(image_name):
    # currently not working on GPU, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_jpeg(contents=tf.io.read_file(x), channels=3)), image_name, dtype=tf.float32,
        )
    return siamese_nets.get_layer("branch_model")(input_tensor)


tf.saved_model.save(
    siamese_nets.get_layer("branch_model"),
    export_dir=str(output_folder / "branch_model"),
    signatures={"serving_default": decode_and_crop_and_serve, "from_crop": decode_and_serve, "preprocessing": preprocessing},
)

#%% Eval on test set
test_set = (
    all_annotations.loc[lambda df: df.split == "test"]
    .loc[lambda df: df.split == "test"]
    .reset_index(drop=True)
    .assign(crop_window=lambda df: df[["crop_y", "crop_x", "crop_height", "crop_width"]].values.tolist())
)

#%% Compute test_set embeddings
test_dataset = (
    test_set.filter(items=["image_name", "crop_window"])
    .pipe(lambda df: tf.data.Dataset.from_tensor_slices(df.to_dict("list")))
    .map(
        lambda annotation: tf.io.decode_and_crop_jpeg(
            contents=tf.io.read_file(annotation["image_name"]), crop_window=annotation["crop_window"], channels=3,
        )
    )
    .map(lambda image: preprocessing(image))
    .batch(64)
)

embeddings = tf.convert_to_tensor(siamese_nets.get_layer("branch_model").predict(test_dataset, verbose=1))

#%% Generate random k_shot n_way task and compute performance
k_shot = 10
n_way = 30
n_episode = 100
random_state = np.random.RandomState(0)
scores = []
wrong_support = []
wrong_query = []
allowed_labels = test_set.label.value_counts().loc[lambda c: c > k_shot].index
for _ in range(n_episode):
    selected_labels = random_state.choice(allowed_labels, size=n_way, replace=False)
    support_set = (
        test_set.loc[lambda df: df.label.isin(selected_labels)]
        .groupby("label")
        .apply(lambda group: group.sample(k_shot, random_state=random_state))
        .reset_index("label", drop=True)
    )
    query_set = test_set.loc[lambda df: df.label.isin(selected_labels)].loc[lambda df: ~df.index.isin(support_set.index)]
    query_set_indexes = np.repeat(query_set.index.values, len(support_set.index)).astype(np.int64)
    support_set_indexes = np.tile(support_set.index.values, len(query_set.index)).astype(np.int64)
    predicted_indexes = np.argmax(
        np.reshape(
            siamese_nets.get_layer("head_model").predict(
                tf.data.Dataset.from_tensor_slices((query_set_indexes, support_set_indexes))
                .map(
                    lambda query_index, support_index: (
                        {
                            siamese_nets.get_layer("head_model").input_names[0]: embeddings[query_index],
                            siamese_nets.get_layer("head_model").input_names[1]: embeddings[support_index],
                        }
                    )
                )
                .batch(64),
                verbose=1,
            ),
            (len(query_set.index), len(support_set.index)),
        ),
        axis=1,
    )
    accuracy = support_set.loc[support_set_indexes[predicted_indexes]].label.values == query_set.label.values
    wrong_support += support_set_indexes[predicted_indexes][np.logical_not(accuracy)].tolist()
    wrong_query += query_set.index.values[np.logical_not(accuracy)].tolist()
    scores += [np.mean(accuracy)]

#%% Save artifacts
scores = pd.DataFrame({"accuracy": scores})
scores.to_csv(output_folder / "scores.csv", index=False)
confusions = pd.concat(
    [
        test_set.loc[wrong_support].add_suffix("_left").reset_index(drop=True),
        test_set.loc[wrong_query].add_suffix("_right").reset_index(drop=True),
    ],
    axis=1,
)
confusions.to_csv(output_folder / "wrong_pairs.csv", index=False)
pd.crosstab(confusions.label_left, confusions.label_right, margins=True).to_csv(output_folder / "errors_confusion_matrix.csv")
plt.clf()
scores.boxplot()
plt.savefig(output_folder / "scores_boxplot.png")
plt.clf()
scores.accuracy.hist()
plt.savefig(output_folder / "accuracy_hist.png")
scores.agg("mean").to_csv(output_folder / "metrics.csv")
