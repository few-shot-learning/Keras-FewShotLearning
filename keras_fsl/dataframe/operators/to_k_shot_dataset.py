from pathlib import Path

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from keras_fsl.dataframe.operators.abstract_operator import AbstractOperator
from keras_fsl.utils import image_preprocessing as ip
from keras_fsl.utils.tfrecord_utils import infer_tfrecord_encoder_decoder_from_sample, clear_cache


class ToKShotDataset(AbstractOperator):
    """
    Create tf.data.Dataset with random groups of k_shot consecutive images with the same label
    """

    def __init__(
        self,
        k_shot,
        preprocessing,
        label_column="label_one_hot",
        cache=None,
        reset_cache=False,
        max_shuffle_buffer_size=100,
        dataset_mode="with_tf_record",
    ):
        """

        Args:
            k_shot (int): number of consecutive crops from the same class
            preprocessing (function): to be applied onto the image after opening
            label_column (str): either "label_one_hot" or "label" depending on the expected form of the network
            cache (Union[str, Path]): cache directory to be passed to tf.data.Dataset.cache.
                Each dataset, one per label, will be cached in Path(cache) / label.
            reset_cache (bool): should reset the cache
            max_shuffle_buffer_size (int): maximum buffer size for shuffle
            dataset_mode (str): one of {with_tf_record, with_cache, raw}
        """
        self.k_shot = k_shot
        self.preprocessing = preprocessing
        self.label_column = label_column
        self.cache = cache
        self._reset_cache = reset_cache
        self._max_shuffle_buffer_size = max_shuffle_buffer_size

        _transforms = {
            "with_tf_record": self.to_dataset_with_tf_record,
            "with_cache": self.to_dataset_with_cache,
            "raw": self.to_dataset_direct,
        }
        self._transform_group_to_dataset = _transforms[dataset_mode]
        if cache is not None:
            self.cache = Path(cache)
            self.cache.mkdir(exist_ok=True, parents=True)

    def repeat_k_shot(self, index):
        return tf.data.Dataset.from_tensors(index).repeat(self.k_shot)

    def transform_group_to_shuffled_dataset(self, group):
        return (
            self._transform_group_to_dataset(group)
            .shuffle(buffer_size=min(self._max_shuffle_buffer_size, len(group)), reshuffle_each_iteration=True)
            .repeat()
        )

    @staticmethod
    def to_dataset_direct(group):
        """
        Transform a pd.DataFrame into a tf.data.Dataset and load images
        """
        return tf.data.Dataset.from_tensor_slices(group.to_dict("list")).map(
            ip.add_field(ip.load_crop_as_uint8_tensor, "image"), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def to_dataset_with_cache(self, group):
        """
        Transform a pd.DataFrame into a tf.data.Dataset and load images
        """
        filename = self.cache / group.name
        if self._reset_cache:
            clear_cache(filename)
        dataset = self.to_dataset_direct(group).cache(str(filename))
        for _ in dataset:
            continue
        return dataset

    def to_dataset_with_tf_record(self, group):
        """
        Transform a pd.DataFrame into a tf.data.Dataset and load images
        """
        filename = self.cache / group.name
        original_dataset = (
            self.to_dataset_direct(group)
            .map(ip.transform_field(tf.image.encode_jpeg, "image"), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        first_sample = next(iter(original_dataset))
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(first_sample)
        if self._reset_cache:
            clear_cache(filename)
            with tf.io.TFRecordWriter(str(filename)) as writer:
                # TODO : use idiomatic writer.write(dataset) when Dataset.map(encoder) work in graph mode with tensors
                for sample in original_dataset:
                    writer.write(encoder(sample))
        return (
            tf.data.TFRecordDataset(str(filename), num_parallel_reads=tf.data.experimental.AUTOTUNE)
            .map(decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(ip.transform_field(tf.io.decode_jpeg, "image"), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )

    def __call__(self, input_dataframe):
        tqdm.pandas(desc=f"Building {self.__class__.__name__} at {self.cache}")
        return tf.data.experimental.choose_from_datasets(
            datasets=(
                input_dataframe.assign(
                    label_one_hot=lambda df: pd.get_dummies(df.label).values.tolist(),
                    crop_window=lambda df: df[["crop_y", "crop_x", "crop_height", "crop_width"]].values.tolist(),
                )
                .groupby("label")
                .progress_apply(self.transform_group_to_shuffled_dataset)
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
