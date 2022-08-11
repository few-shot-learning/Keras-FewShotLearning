"""
This notebooks borrows from https://www.tensorflow.org/addons/tutorials/losses_triplet and is intended to compare tf.addons triplet loss
implementation against this one. It is also aimed at benchmarking the impact of the distance function.
"""
from pathlib import Path
from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Dropout, GlobalMaxPooling2D, Input, Flatten, MaxPooling2D, Lambda
from keras.models import Sequential

from keras_fsl.layers import GramMatrix
from keras_fsl.losses.gram_matrix_losses import BinaryCrossentropy, class_consistency_loss, TripletLoss
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy
from keras_fsl.utils.tensors import get_dummies


#%% Build datasets
train_dataset, val_dataset, test_dataset = tfds.load(
    name="cifar10", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True
)
train_dataset = train_dataset.shuffle(1024).batch(64, drop_remainder=True)
val_dataset = val_dataset.shuffle(1024).batch(64, drop_remainder=True)
test_dataset = test_dataset.batch(64, drop_remainder=True)

train_labels = [labels.numpy().tolist() for _, labels in train_dataset]
val_labels = [labels.numpy().tolist() for _, labels in val_dataset]
test_labels = [labels.numpy().tolist() for _, labels in test_dataset]
train_steps = len(train_labels)
val_steps = len(val_labels)
test_steps = len(test_labels)

print(
    pd.concat(
        [
            pd.DataFrame({"label": tf.nest.flatten(train_labels)}).assign(split="train"),
            pd.DataFrame({"label": tf.nest.flatten(val_labels)}).assign(split="val"),
            pd.DataFrame({"label": tf.nest.flatten(test_labels)}).assign(split="test"),
        ]
    )
    .groupby("split")
    .apply(lambda group: pd.get_dummies(group.label).agg("sum"))
)

output_dir = Path("logs") / "triplet_loss_cifar10"
output_dir.mkdir(exist_ok=True, parents=True)
results = []

#%% Save test labels for later visualization in projector https://projector.tensorflow.org/
np.savetxt(output_dir / "meta.tsv", tf.nest.flatten(test_labels), fmt="%0d")

#%% Build model
encoder = Sequential(
    [
        Input(train_dataset.element_spec[0].shape[1:]),
        Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        GlobalMaxPooling2D(),
        Flatten(),
    ]
)
encoder.save_weights(str(output_dir / "initial_encoder.h5"))

#%% Train encoder with usual cross entropy
encoder.load_weights(str(output_dir / "initial_encoder.h5"))
classifier = Sequential([encoder, Dense(10, activation="softmax")])
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
classifier.fit(
    train_dataset.map(
        lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat(),
    epochs=100,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(
        lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard(str(output_dir / "sparse_categorical_crossentropy"))],
)
loss, accuracy = classifier.evaluate(
    test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y)), steps=test_steps
)
results += [{"experiment": "classifier", "loss": loss, "top_score_classification_accuracy": accuracy}]
embeddings = encoder.predict(test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y)), steps=test_steps)
np.savetxt(str(output_dir / "classifier_embeddings.tsv"), embeddings, delimiter="\t")

#%% Train
experiments = [
    {
        "name": "l2_triplet_loss",
        "kernel": Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=1)),
        "loss": TripletLoss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "l1_triplet_loss",
        "kernel": Lambda(lambda x: tf.reduce_sum(tf.abs(x[0] - x[1]), axis=1)),
        "loss": TripletLoss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "cosine_similarity_triplet_loss",
        "kernel": Lambda(
            lambda x: 1 - tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)
        ),
        "loss": TripletLoss(0.1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "cosine_similarity_crossentropy_loss",
        "kernel": Lambda(
            lambda x: (1 + tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)) / 2
        ),
        "loss": BinaryCrossentropy(),
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, BinaryCrossentropy()],
    },
    {
        "name": "cosine_similarity_consistency_loss",
        "kernel": Lambda(
            lambda x: (1 + tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)) / 2
        ),
        "loss": class_consistency_loss,
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, BinaryCrossentropy()],
    },
    {
        "name": "mixed_norms_triplet_loss",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "relu", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": TripletLoss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "learnt_norms_triplet_loss",
        "kernel": {"name": "LearntNorms", "init": {"activation": "relu"}},
        "loss": TripletLoss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "mixed_similarity_triplet_loss",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "sigmoid", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": TripletLoss(0.1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "learnt_similarity_triplet_loss",
        "kernel": {"name": "LearntNorms", "init": {"activation": "sigmoid"}},
        "loss": TripletLoss(0.1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "mixed_similarity_crossentropy_loss",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "sigmoid", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": BinaryCrossentropy(),
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, BinaryCrossentropy()],
    },
    {
        "name": "learnt_similarity_crossentropy_loss",
        "kernel": {"name": "LearntNorms", "init": {"activation": "sigmoid"}},
        "loss": BinaryCrossentropy(),
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, BinaryCrossentropy()],
    },
    {
        "name": "mixed_similarity_consistency_loss",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "sigmoid", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": class_consistency_loss,
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, BinaryCrossentropy()],
    },
    {
        "name": "learnt_similarity_consistency_loss",
        "kernel": {"name": "LearntNorms", "init": {"activation": "sigmoid"}},
        "loss": class_consistency_loss,
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, BinaryCrossentropy()],
    },
]
projectors = [
    {"name": "", "projector": []},
    {"name": "_l2_normalize", "projector": [Lambda(lambda x: tf.math.l2_normalize(x, axis=1))]},
    {"name": "_dense_10", "projector": [Dense(10)]},
    {"name": "_dense_128", "projector": [Dense(128)]},
]
for experiment, projector in itertools.product(experiments, projectors):
    pprint(experiment)
    pprint(projector)
    for i in range(10):
        encoder.load_weights(str(output_dir / "initial_encoder.h5"))
        model = Sequential([encoder, *projector["projector"], GramMatrix(kernel=experiment["kernel"])])
        model.compile(
            optimizer="adam", loss=experiment["loss"], metrics=experiment["metrics"],
        )
        model.fit(
            train_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), get_dummies(y)[0])).repeat(),
            epochs=100,
            steps_per_epoch=train_steps,
            validation_data=val_dataset.map(
                lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), get_dummies(y)[0])
            ).repeat(),
            validation_steps=val_steps,
            callbacks=[TensorBoard(str(output_dir / f"{experiment['name']}{projector['name']}_{i}"))],
        )
        results += [
            {
                "experiment": experiment["name"],
                "projector": projector["name"],
                "iteration": i,
                **dict(
                    zip(
                        model.metrics_names,
                        model.evaluate(
                            test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), get_dummies(y)[0])),
                            steps=test_steps,
                        ),
                    )
                ),
            }
        ]
        embeddings = encoder.predict(
            test_dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), get_dummies(y)[0])), steps=test_steps
        )
        np.savetxt(str(output_dir / f"{experiment['name']}{projector['name']}_{i}.tsv"), embeddings, delimiter="\t")

#%% Export final stats
pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)

#%% Plot results
results = pd.read_csv(output_dir / "results.csv")
baseline = results[results.experiment == "classifier"].dropna(axis=1)
results = (
    results.loc[lambda df: df.experiment != "classifier"]
    .pipe(
        lambda df: pd.concat(
            [
                df.fillna({"projector": "raw"}).filter(items=["top_score_classification_accuracy", "projector"]),
                df.experiment.str.extract(
                    r"(?P<similarity>l1|l2|cosine_similarity|mixed_norms|mixed_similarity|learnt_norms|learnt_similarity)_"
                    r"(?P<loss_name>triplet_loss|consistency_loss|crossentropy_loss)"
                ),
            ],
            axis=1,
        )
    )
    .assign(projector=lambda df: df.projector.str.strip("_"))
)
chart = sns.catplot(
    x="similarity",
    y="top_score_classification_accuracy",
    col="projector",
    row="loss_name",
    data=results,
    legend=True,
    legend_out=True,
)
chart.set_xticklabels(rotation=90)
[ax.axhline(y=baseline.top_score_classification_accuracy[0]) for ax in chart.axes.flatten()]
plt.tight_layout()
plt.savefig(output_dir / "all_losses.png")
plt.show()
