import shutil
from functools import partial
from pathlib import Path

import click
import pandas as pd
import tensorflow as tf
from keras_fsl.dataframe.operators import ToKShotDataset
from keras_fsl.losses import binary_crossentropy, max_crossentropy, std_crossentropy
from keras_fsl.metrics import accuracy, same_image_score, top_score_classification_accuracy
from keras_fsl.models import SiameseNets
from keras_fsl.models.layers import GramMatrix
from keras_fsl.utils import compose
from tensorflow.keras import applications as keras_applications
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


#%% Toggle some config if required
# tf.config.experimental_run_functions_eagerly(True)
# tf.config.optimizer.set_jit(True)
# policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
# tf.keras.mixed_precision.experimental.set_policy(policy)


#%% CLI args
@click.option(
    "--base_dir", help="Base directory for the training", type=Path,
)
@click.command()
def train(base_dir):
    #%% Init model
    branch_model_name = "MobileNet"
    siamese_nets = SiameseNets(
        branch_model={"name": branch_model_name, "init": {"include_top": False, "input_shape": (224, 224, 3), "pooling": "avg"}},
        head_model={
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
    )

    model = Sequential([siamese_nets.get_layer("branch_model"), GramMatrix(kernel=siamese_nets.get_layer("head_model"))])

    #%% Init training
    callbacks = [
        TensorBoard(base_dir, write_images=True, histogram_freq=1),
        ModelCheckpoint(str(base_dir / "best_loss.h5"), save_best_only=True),
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
    all_annotations = pd.read_csv(base_dir / "annotations" / "all_annotations.csv")
    class_count = all_annotations.groupby("split").apply(lambda group: group.label.value_counts())

    #%% Train model
    margin = 0.05
    k_shot = 4
    cache = base_dir / "cache"
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir()
    datasets = all_annotations.groupby("split").apply(
        lambda group: (
            group.pipe(
                ToKShotDataset(
                    k_shot=k_shot,
                    preprocessing=compose(preprocessing, data_augmentation)
                    # k_shot=k_shot, preprocessing=compose(preprocessing, data_augmentation), cache=str(cache / group.name)
                )
            )
        )
    )

    batch_size = 64
    siamese_nets.get_layer("branch_model").trainable = False
    optimizer = Adam(lr=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy(margin),
        metrics=[
            accuracy(margin),
            binary_crossentropy(),
            max_crossentropy,
            std_crossentropy,
            same_image_score,
            top_score_classification_accuracy,
        ],
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

    siamese_nets.get_layer("branch_model").trainable = True
    optimizer = Adam(lr=1e-5)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy(margin),
        metrics=[
            accuracy(margin),
            binary_crossentropy(),
            max_crossentropy,
            std_crossentropy,
            same_image_score,
            top_score_classification_accuracy,
        ],
    )
    model.fit(
        datasets["train"].batch(batch_size).repeat(),
        steps_per_epoch=len(class_count["train"]) * k_shot // batch_size * 150,
        validation_data=datasets["val"].batch(batch_size).repeat(),
        validation_steps=max(len(class_count["val"]) * k_shot // batch_size, 100),
        initial_epoch=3,
        epochs=5,
        callbacks=callbacks,
    )

    #%% Evaluate on test set
    model.evaluate(datasets["test"].batch(batch_size).repeat(), steps=max(len(class_count["test"]) * k_shot // batch_size, 100))

    #%% Export artifacts
    siamese_nets.save(base_dir / "final_model.h5")

    model.load_weights(str(base_dir / "best_loss.h5"))
    siamese_nets.get_layer("branch_model").save(str(base_dir / "branch_model_best_loss.h5"))
    siamese_nets.get_layer("head_model").save(str(base_dir / "head_model_best_loss.h5"))

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
        export_dir=str(base_dir / "branch_model"),
        signatures={"serving_default": decode_and_crop_and_serve, "from_crop": decode_and_serve, "preprocessing": preprocessing,},
    )


#%% Run command
if __name__ == "__main__":
    train()
