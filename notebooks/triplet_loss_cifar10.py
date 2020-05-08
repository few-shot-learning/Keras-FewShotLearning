"""
This notebooks borrows from https://www.tensorflow.org/addons/tutorials/losses_triplet and is intended to compare tf.addons triplet loss
implementation against this one. It is also aimed at benchmarking the impact of the distance function.
"""
from pathlib import Path
from pprint import pprint

import io
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential

from keras_fsl.layers import GramMatrix
from keras_fsl.losses.gram_matrix_losses import binary_crossentropy, class_consistency_loss, triplet_loss
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy
from keras_fsl.utils.tensors import get_dummies


#%% Build datasets
def preprocessing(input_tensor):
    return tf.cast(input_tensor, tf.float32) / 255


train_dataset, val_dataset, test_dataset = tfds.load(
    name="cifar10", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True
)
train_dataset = train_dataset.shuffle(1024).batch(64, drop_remainder=True)
val_dataset = val_dataset.shuffle(1024).batch(64, drop_remainder=True)
test_dataset = test_dataset.batch(64, drop_remainder=True)

train_labels = [batch[1].numpy().tolist() for batch in train_dataset]
val_labels = [batch[1].numpy().tolist() for batch in val_dataset]
test_labels = [batch[1].numpy().tolist() for batch in test_dataset]
train_steps = len(train_labels)
val_steps = len(val_labels)
test_steps = len(test_labels)

input_shape = next(train_dataset.take(1).as_numpy_iterator())[0].shape

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
results = []

#%% Save test labels for later visualization in projector https://projector.tensorflow.org/
out_m = io.open(output_dir / "meta.tsv", "w", encoding="utf-8")
for img, labels in tfds.as_numpy(test_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()

#%% Build model
encoder = Sequential(
    [
        Input(input_shape[1:]),
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
    ]
)
encoder.save_weights(str(output_dir / "initial_encoder.h5"))

#%% Train encoder with usual cross entropy
encoder.load_weights(str(output_dir / "initial_encoder.h5"))
classifier = Sequential([encoder, Dense(10, activation="softmax")])
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]
)
classifier.fit(
    train_dataset.map(lambda x, y: (preprocessing(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(),
    epochs=50,
    steps_per_epoch=train_steps,
    validation_data=val_dataset.map(
        lambda x, y: (preprocessing(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat(),
    validation_steps=val_steps,
    callbacks=[TensorBoard(str(output_dir / "sparse_categorical_crossentropy"))],
)
loss, accuracy = classifier.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), y)), steps=test_steps)
results += [{"name": "classifier", "loss": loss, "accuracy": accuracy}]
embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), y)), steps=test_steps)
np.savetxt(str(output_dir / "classifier_embeddings.tsv"), embeddings, delimiter="\t")

#%% Train
experiments = [
    {
        "name": "l2_triplet_loss",
        "kernel": Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=1)),
        "loss": triplet_loss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "l1_triplet_loss",
        "kernel": Lambda(lambda x: tf.reduce_sum(tf.abs(x[0] - x[1]), axis=1)),
        "loss": triplet_loss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "mixed_norms_triplet_loss",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "relu", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": triplet_loss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "learnt_norms_triplet_loss",
        "kernel": {"name": "LearntNorms", "init": {"activation": "relu"}},
        "loss": triplet_loss(1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "cosine_similarity_triplet_loss",
        "kernel": Lambda(
            lambda x: 1 - tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)
        ),
        "loss": triplet_loss(0.1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "mixed_similarity_triplet_loss",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "sigmoid", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": triplet_loss(0.1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "learnt_similarity_triplet_loss",
        "kernel": {"name": "LearntNorms", "init": {"activation": "sigmoid"}},
        "loss": triplet_loss(0.1),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "mixed_crossentropy",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "sigmoid", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": binary_crossentropy(),
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, binary_crossentropy()],
    },
    {
        "name": "learnt_crossentropy",
        "kernel": {"name": "LearntNorms", "init": {"activation": "sigmoid"}},
        "loss": binary_crossentropy(),
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, binary_crossentropy()],
    },
    {
        "name": "mixed_consistency",
        "kernel": {
            "name": "MixedNorms",
            "init": {"activation": "sigmoid", "norms": [lambda x: tf.square(x[0] - x[1]), lambda x: tf.abs(x[0] - x[1])]},
        },
        "loss": class_consistency_loss,
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, binary_crossentropy()],
    },
    {
        "name": "learnt_consistency",
        "kernel": {"name": "LearntNorms", "init": {"activation": "sigmoid"}},
        "loss": class_consistency_loss,
        "metrics": [classification_accuracy(ascending=False), class_consistency_loss, binary_crossentropy()],
    },
]
for experiment in experiments:
    pprint(experiment)
    encoder.load_weights(str(output_dir / "initial_encoder.h5"))
    model = Sequential([encoder, GramMatrix(kernel=experiment["kernel"])])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001), loss=experiment["loss"], metrics=experiment["metrics"],
    )
    model.fit(
        train_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
        epochs=100,
        steps_per_epoch=train_steps,
        validation_data=val_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])).repeat(),
        validation_steps=val_steps,
        callbacks=[TensorBoard(str(output_dir / experiment["name"])), EarlyStopping(patience=10)],
    )
    results += [
        {
            "name": experiment["name"],
            **dict(
                zip(
                    model.metrics_names,
                    model.evaluate(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps),
                )
            ),
        }
    ]
    embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)
    np.savetxt(str(output_dir / f"{experiment['name']}.tsv"), embeddings, delimiter="\t")

#%% Export final stats
pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
