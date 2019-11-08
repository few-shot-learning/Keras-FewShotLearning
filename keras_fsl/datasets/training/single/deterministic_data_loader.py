import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf


class DeterministicDataLoader:
    def __init__(self, annotations, batch_size, repeat_number=None, classes=None):

        if not isinstance(annotations, list):
            annotations = [annotations]
        self.annotations = [
            annotations_.assign(crop_coordinates=lambda df: df.get("crop_coordinates"))
            for annotations_ in annotations
        ]

        self.batch_size = batch_size
        self.targets = pd.get_dummies(self.annotations[0].label)
        if classes is not None:
            self.targets = self.targets.reindex(classes, axis=1, fill_value=0)

        self.repeat_number = repeat_number

    def load(self, shuffle=True, cache=True, augment=True):
        image_dataset = tf.data.Dataset.from_tensor_slices(list(self.annotations[0].image_name.values))
        image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        box_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(self.annotations[0][["x1", "x2", "y1", "y2"]].values, tf.int32))
        label_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(self.targets.values, tf.float32))

        crop_dataset = tf.data.Dataset.zip((image_dataset, box_dataset))
        crop_dataset = (
            crop_dataset.map(
                lambda image, box: tf.image.crop_to_bounding_box(
                    image, box[2], box[0], box[3] - box[2], box[1] - box[0]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(
                lambda image: tf.image.resize_with_crop_or_pad(image, 224, 224),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        )

        dataset = tf.data.Dataset.zip((crop_dataset, label_dataset))
        if cache:
            dataset = dataset.cache()

        if augment:
            dataset = (
                dataset.map(
                    lambda x, y: (tf.image.random_flip_left_right(x), y),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
                .map(
                    lambda x, y: (tf.image.random_flip_up_down(x), y),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
                .map(
                    lambda x, y: (tf.image.random_brightness(x, 0.1), y),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
                .map(
                    lambda x, y: (tf.image.random_contrast(x, 0.9, 1.1), y),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
                .map(
                    lambda x, y: (tf.image.random_saturation(x, 0.9, 1.1), y),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
            )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(count=self.repeat_number)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image
