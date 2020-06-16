from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from gpumonitor.callbacks.tf import TFGpuMonitorCallback
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, GlobalMaxPooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Sequential

from keras_fsl.utils.datasets import assign, cache, cache_with_tf_record, read_decode_and_crop_jpeg, transform

#%% Read Cifar10 dataset to dump images and create df
train_dataset = tfds.load(name="cifar10", split="train")
output_dir = Path("logs") / "benchmark_caching_performance"
(output_dir / "cifar10").mkdir(exist_ok=True, parents=True)
examples = []
for example in train_dataset:
    tf.io.write_file(str(output_dir / "cifar10" / example["id"].numpy().decode()), tf.io.encode_jpeg(example["image"]))
    examples += [{"id": example["id"].numpy().decode(), "label": example["label"].numpy()}]

#%% Create datasets
datasets = {
    key: pd.DataFrame(examples)
    .assign(filename=lambda df: str(output_dir / "cifar10") + "/" + df.id)
    .pipe(lambda df: tf.data.Dataset.from_tensor_slices(df.to_dict("list")))
    .map(assign(image=read_decode_and_crop_jpeg), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .apply(cache_func)
    .map(
        transform(image=partial(tf.image.convert_image_dtype, dtype=tf.float32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .map(
        lambda x: (
            tf.ensure_shape(x["image"], train_dataset.element_spec["image"].shape),
            tf.ensure_shape(x["label"], train_dataset.element_spec["label"].shape),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .batch(64)
    for key, cache_func in zip(
        ["tf_record_cache", "dataset_cache", "no_cache"],
        [cache_with_tf_record(output_dir / "tf_record_cache"), cache(output_dir / "dataset_cache"), lambda ds: ds],
    )
}

#%% Create model
model = Sequential(
    [
        Input(train_dataset.element_spec["image"].shape),
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
model.save_weights(str(output_dir / "initial_weights.h5"))

#%% Train
for key, dataset in datasets.items():
    model.load_weights(str(output_dir / "initial_weights.h5"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.fit(dataset, callbacks=[TFGpuMonitorCallback(delay=0.5)])
