from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalMaxPooling2D, Input, Flatten, MaxPooling2D, Lambda
from tensorflow.keras.models import Sequential

from keras_fsl.layers import GramMatrix
from keras_fsl.losses.gram_matrix_losses import BinaryCrossentropy, ClassConsistencyLoss
from keras_fsl.metrics.gram_matrix_metrics import classification_accuracy
from keras_fsl.utils.tensors import get_dummies

#%% Init
output_dir = Path("logs") / "unsupervised_cifar10"
output_dir.mkdir(exist_ok=True, parents=True)
results = []


#%% Preprocessing
def data_augmentation(input_tensor):
    output_tensor = input_tensor
    output_tensor = tf.image.random_flip_left_right(output_tensor)
    output_tensor = tf.image.random_flip_up_down(output_tensor)
    output_tensor = tf.image.random_saturation(output_tensor, 0.5, 2)
    output_tensor = tf.image.random_brightness(output_tensor, max_delta=0.4)
    return output_tensor


def preprocessing(input_tensor):
    return tf.image.convert_image_dtype(input_tensor, dtype=tf.float32)


#%% Build datasets
k_shot = 4
n_way = 16
train_dataset = (
    tfds.load(name="cifar10", split="train[:90%]", shuffle_files=True)
    .shuffle(50000 * 9 // 10)
    .batch(n_way, drop_remainder=True)
    .map(
        lambda annotation: (
            tf.repeat(annotation["image"], k_shot, axis=0),
            tf.cast(get_dummies(tf.repeat(annotation["id"], k_shot))[0], tf.float32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
)

val_dataset = (
    tf.data.experimental.choose_from_datasets(
        datasets=[
            tfds.load(name="cifar100", split="train[90%:]").filter(lambda annotation: annotation["label"] == i)
            for i in range(100)
        ],
        choice_dataset=(
            tf.data.Dataset.range(100)
            .shuffle(buffer_size=100, reshuffle_each_iteration=True)
            .flat_map(lambda index: tf.data.Dataset.from_tensors(index).repeat(k_shot))
        ),
    )
    .batch(k_shot * n_way)
    .map(
        lambda annotation: (annotation["image"], tf.cast(get_dummies(annotation["label"])[0], tf.float32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
)

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
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
    ]
)
encoder.save_weights(str(output_dir / "initial_encoder.h5"))

#%% Supervised baseline with usual cross entropy
encoder.load_weights(str(output_dir / "initial_encoder.h5"))
classifier = Sequential([encoder, Dense(10, activation="softmax")])
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]
)
classifier.fit(
    (
        tfds.load(name="cifar10", split="train[:90%]", shuffle_files=True, as_supervised=True)
        .map(lambda x, y: (preprocessing(data_augmentation(x)), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(k_shot * n_way)
        .repeat()
    ),
    epochs=10,
    steps_per_epoch=train_steps,
    validation_data=(
        tfds.load(name="cifar10", split="train[90%:]", shuffle_files=True, as_supervised=True)
        .map(lambda x, y: (preprocessing(data_augmentation(x)), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(k_shot * n_way)
        .repeat()
    ),
    validation_steps=val_steps,
    callbacks=[TensorBoard(str(output_dir / "sparse_categorical_crossentropy"))],
)
results += [
    {
        "name": "classifier",
        **dict(
            zip(
                classifier.metrics_names,
                classifier.evaluate(
                    (
                        tfds.load(name="cifar10", split="test", as_supervised=True)
                        .map(
                            lambda x, y: (preprocessing(data_augmentation(x)), y),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        )
                        .batch(k_shot * n_way)
                        # .repeat()
                    ),
                    # steps=test_steps,
                ),
            )
        ),
    }
]

#%% Train
experiments = [
    {
        "name": "binary_supervised",
        "kernel": Lambda(
            lambda x: 1 - tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)
        ),
        "loss": BinaryCrossentropy(unsupervised=False),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "binary_unsupervised",
        "kernel": Lambda(
            lambda x: 1 - tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)
        ),
        "loss": BinaryCrossentropy(unsupervised=True),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "class_consistency_supervised",
        "kernel": Lambda(
            lambda x: 1 - tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)
        ),
        "loss": ClassConsistencyLoss(unsupervised=False),
        "metrics": [classification_accuracy(ascending=True)],
    },
    {
        "name": "class_consistency_unsupervised",
        "kernel": Lambda(
            lambda x: 1 - tf.reduce_sum(tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1), axis=1)
        ),
        "loss": ClassConsistencyLoss(unsupervised=True),
        "metrics": [classification_accuracy(ascending=True)],
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
