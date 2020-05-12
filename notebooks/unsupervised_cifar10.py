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
    tfds.load(name="cifar10", split="train[:90%]")
    .shuffle(50000 * 9 // 10)
    .batch(n_way, drop_remainder=True)
    .map(
        lambda annotation: (
            tf.repeat(annotation["image"], k_shot, axis=0),
            tf.cast(get_dummies(tf.repeat(annotation["id"], k_shot))[0], tf.float32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .unbatch()
    .map(lambda x, y: (preprocessing(data_augmentation(x)), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(k_shot * n_way)
)

val_dataset, test_dataset = (
    dataset.map(
        lambda x, y: (preprocessing(x), tf.one_hot(y, depth=10)), num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(128)
    for dataset in tfds.load(name="cifar10", split=["train[90%:]", "test"], as_supervised=True)
)

train_steps = len([_ for _ in train_dataset])  # Fixme: tf.data.experimental returns UNKNOWN_CARDINALITY
val_steps = len([_ for _ in val_dataset])  # Fixme: tf.data.experimental returns UNKNOWN_CARDINALITY
test_steps = len([_ for _ in test_dataset])  # Fixme: tf.data.experimental returns UNKNOWN_CARDINALITY

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
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
classifier.fit(
    (
        tfds.load(name="cifar10", split="train[:90%]", as_supervised=True)
        .map(
            lambda x, y: (preprocessing(data_augmentation(x)), tf.one_hot(y, depth=10)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(128)
        .repeat()
    ),
    epochs=10,
    steps_per_epoch=train_steps * n_way // 128,
    validation_data=val_dataset,
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
                        .repeat()
                    ),
                    steps=test_steps,
                ),
            )
        ),
    }
]

#%% Train
experiments = [
    {"name": "binary_supervised", "loss": BinaryCrossentropy(unsupervised=False)},
    {"name": "binary_unsupervised", "loss": BinaryCrossentropy(unsupervised=True)},
    {"name": "class_consistency_supervised", "loss": ClassConsistencyLoss(unsupervised=False)},
    {"name": "class_consistency_unsupervised", "loss": ClassConsistencyLoss(unsupervised=True)},
]
for experiment in experiments:
    pprint(experiment)
    encoder.load_weights(str(output_dir / "initial_encoder.h5"))
    model = Sequential([encoder, GramMatrix(kernel="LearntNorms")])
    model.compile(
        optimizer="adam", loss=experiment["loss"], metrics=[classification_accuracy(ascending=False)],
    )
    model.fit(
        train_dataset.repeat(),
        epochs=5,
        steps_per_epoch=train_steps,
        validation_data=val_dataset.repeat(),
        validation_steps=val_steps,
        callbacks=[TensorBoard(str(output_dir / experiment["name"])), EarlyStopping(patience=10)],
    )
    results += [
        {
            "name": experiment["name"],
            **dict(zip(model.metrics_names, model.evaluate(test_dataset.repeat(), steps=test_steps),)),
        }
    ]
    embeddings = encoder.predict(test_dataset.map(lambda x, y: (preprocessing(x), get_dummies(y)[0])), steps=test_steps)

#%% Export final stats
pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
