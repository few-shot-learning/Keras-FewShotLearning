from pathlib import Path

import pandas as pd
import tensorflow as tf

from keras_fsl.dataframe.operators.abstract_operator import AbstractOperator


class ToKShotDataset(AbstractOperator):
    """
    Create tf.data.Dataset with random groups of k_shot consecutive images with the same label
    """

    def __init__(self, k_shot, preprocessing, label_column="label_one_hot", cache=""):
        """

        Args:
            k_shot (int): number of consecutive crops from the same class
            preprocessing (function): to be applied onto the image after opening
            label_column (str): either "label_one_hot" or "label" depending on the expected form of the network
            cache (Union[str, pathlib.Path]): cache directory to be passed to tf.data.Dataset.cache: each dataset (one per label) will be
                cached in Path(cache) / label. No cleaning is done, see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache.
        """
        self.k_shot = k_shot
        self.preprocessing = preprocessing
        self.label_column = label_column
        self.cache = Path(cache)
        self.cache.mkdir(exist_ok=True)

    @staticmethod
    def load_img(annotation):
        """
        Args:
            annotation (dict): with keys 'image_name': path to the image and 'crop_window' to be passed to tf.io.decode_and_crop_jpeg
        Returns:
            dict: the input dict with an extra image key.
        """
        return {
            "image": tf.io.decode_and_crop_jpeg(
                tf.io.read_file(annotation["image_name"]), crop_window=annotation["crop_window"], channels=3,
            ),
            **annotation,
        }

    def repeat_k_shot(self, index):
        return tf.data.Dataset.from_tensors(index).repeat(self.k_shot)

    def to_dataset(self, group):
        """
        Transform a pd.DataFrame into a tf.data.Dataset and load images
        """
        return (
            tf.data.Dataset.from_tensor_slices(group.to_dict("list"))
            .map(self.load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .cache(str(self.cache / group.name))
            .shuffle(buffer_size=len(group), reshuffle_each_iteration=True)
            .repeat()
        )

    def __call__(self, input_dataframe):
        return tf.data.experimental.choose_from_datasets(
            datasets=(
                input_dataframe.assign(
                    label_one_hot=lambda df: pd.get_dummies(df.label).values.tolist(),
                    crop_window=lambda df: df[["crop_y", "crop_x", "crop_height", "crop_width"]].values.tolist(),
                )
                .groupby("label")
                .apply(self.to_dataset)
            ),
            choice_dataset=(
                tf.data.Dataset.range(len(input_dataframe.label.unique()))
                .shuffle(buffer_size=len(input_dataframe.label.unique()), reshuffle_each_iteration=True)
                .flat_map(self.repeat_k_shot)
            ),
        ).map(
            lambda annotation: (self.preprocessing(annotation["image"]), tf.cast(annotation[self.label_column], tf.float32)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
