from pathlib import Path

import click
import pandas as pd
import tensorflow as tf
from tensorflow.keras import applications as keras_applications
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from keras_fsl.dataframe.operators import ToKShotDataset
from keras_fsl.layers import CentroidsMatrix
from keras_fsl.utils.tensors import get_dummies
from keras_fsl.utils.training import compose


#%% Toggle some config if required
# tf.config.experimental_run_functions_eagerly(True)
# tf.config.optimizer.set_jit(True)
# policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
# tf.keras.mixed_precision.experimental.set_policy(policy)


#%% CLI args
@click.option("--base_dir", help="Base directory for the training", type=Path, default="")
@click.command()
def train(base_dir):
    #%% Init model
    encoder = keras_applications.MobileNet(input_shape=(224, 224, 3), include_top=False, pooling="avg")
    support_layer = CentroidsMatrix(
        kernel={
            "name": "MixedNorms",
            "init": {
                "norms": [
                    lambda x: 1 - tf.nn.l2_normalize(x[0]) * tf.nn.l2_normalize(x[1]),
                    lambda x: tf.math.abs(x[0] - x[1]),
                    lambda x: tf.nn.softmax(tf.math.abs(x[0] - x[1])),
                    lambda x: tf.square(x[0] - x[1]),
                ],
                "use_bias": True,
            },
        },
        activation="linear",
    )

    #%% Init training
    callbacks = [
        TensorBoard(base_dir, write_images=True, histogram_freq=1),
        ModelCheckpoint(str(base_dir / "best_loss.h5"), save_best_only=True),
        ReduceLROnPlateau(),
    ]

    #%% Init data
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8),))
    def preprocessing(input_tensor):
        output_tensor = tf.cast(input_tensor, dtype=tf.float32)
        output_tensor = tf.image.resize_with_pad(output_tensor, target_height=224, target_width=224)
        output_tensor = keras_applications.mobilenet.preprocess_input(output_tensor, data_format="channels_last")
        return output_tensor

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),))
    def data_augmentation(input_tensor):
        output_tensor = tf.image.random_flip_left_right(input_tensor)
        output_tensor = tf.image.random_flip_up_down(output_tensor)
        output_tensor = tf.image.random_brightness(output_tensor, max_delta=0.25)
        return output_tensor

    all_annotations = pd.read_csv(base_dir / "annotations" / "all_annotations.csv").assign(
        label_code=lambda df: df.label.astype("category").cat.codes
    )
    class_count = all_annotations.groupby("split").apply(lambda group: group.label.value_counts())

    #%% Train model
    k_shot = 4
    cache = base_dir / "cache"
    datasets = all_annotations.groupby("split").apply(
        lambda group: (
            ToKShotDataset(
                k_shot=k_shot,
                preprocessing=compose(preprocessing, data_augmentation),
                cache=str(cache / group.name),
                reset_cache=True,
                dataset_mode="with_cache",
                label_column="label_code",
            )(group)
        )
    )

    y_true = Input(shape=(None,), name="y_true")
    output = support_layer([encoder.output, y_true])
    model = Model([encoder.inputs, y_true], output)

    batch_size = 64
    batched_datasets = datasets.map(
        lambda dataset: dataset.batch(batch_size, drop_remainder=True)
        .map(lambda x, y: (x, get_dummies(y)[0]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(lambda x, y: ((x, y), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .repeat()
    )

    encoder.trainable = False
    optimizer = Adam(lr=1e-4)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["categorical_accuracy", "categorical_crossentropy"]
    )
    model.fit(
        datasets["train"].batch(batch_size).repeat(),
        steps_per_epoch=len(class_count["train"]) * k_shot // batch_size * 150,
        validation_data=datasets["val"].batch(batch_size).repeat(),
        validation_steps=max(len(class_count["val"]) * k_shot // batch_size, 100),
        initial_epoch=0,
        epochs=3,
        callbacks=callbacks,
    )

    encoder.trainable = True
    optimizer = Adam(lr=1e-5)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["categorical_accuracy", "categorical_crossentropy"]
    )
    model.fit(
        datasets["train"].batch(batch_size).repeat(),
        steps_per_epoch=len(class_count["train"]) * k_shot // batch_size * 150,
        validation_data=datasets["val"].batch(batch_size).repeat(),
        validation_steps=max(len(class_count["val"]) * k_shot // batch_size, 100),
        initial_epoch=3,
        epochs=10,
        callbacks=callbacks,
    )

    #%% Evaluate on test set. Each batch is a k_shot, n_way=batch_size / k_shot task
    model.load_weights(str(base_dir / "best_loss.h5"))
    model.evaluate(batched_datasets["test"], steps=max(len(class_count["test"]) * k_shot // batch_size, 100))


#%% Run command
if __name__ == "__main__":
    train()
