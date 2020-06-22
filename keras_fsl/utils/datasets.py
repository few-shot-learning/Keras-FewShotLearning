import pathlib
from functools import partial
from pathlib import Path
from typing import Callable, List, Union

import tensorflow as tf

from keras_fsl.utils.types import TENSOR_MAP, TF_TENSOR


def assign(**kwargs: Callable[[TENSOR_MAP], TF_TENSOR]) -> Callable[[TENSOR_MAP], TENSOR_MAP]:
    """Wrap a tensor produce function to create a new field in the TENSOR_MAP"""

    def annotations_mapper(annotations: TENSOR_MAP) -> TENSOR_MAP:
        return {
            **annotations,
            **{key: produce(annotations) for key, produce in kwargs.items()},
        }

    return annotations_mapper


def transform(**kwargs: Callable[[TF_TENSOR], TF_TENSOR]) -> Callable[[TENSOR_MAP], TENSOR_MAP]:
    """Wrap a tensor transform function to apply it on a specific field of a TENSOR_MAP"""

    def annotations_mapper(annotations: TENSOR_MAP) -> TENSOR_MAP:
        return {**annotations, **{key: _transform(annotations[key]) for key, _transform in kwargs.items()}}

    return annotations_mapper


def filter_items(items: List[str]) -> Callable[[TENSOR_MAP], TENSOR_MAP]:
    """Filter keys like pandas.DataFrame.filter"""

    def annotations_mapper(annotations: TENSOR_MAP) -> TENSOR_MAP:
        return {key: value for key, value in annotations.items() if key in items}

    return annotations_mapper


def read_decode_and_crop_jpeg(annotation: TENSOR_MAP) -> TF_TENSOR:
    """
    Args:
        annotation (TENSOR_MAP): contains 'filename' (path to the image) and possibly 'crop_window' (box coordinates)
    Returns:
        TF_TENSOR: the crop described by annotations as a uint8 array
    """
    image_content = tf.io.read_file(filename=annotation["filename"])
    if "crop_window" in annotation:
        return tf.io.decode_and_crop_jpeg(image_content, crop_window=annotation["crop_window"], channels=3)

    return tf.io.decode_jpeg(image_content, channels=3)


def cache_with_tf_record(
    filename: Union[str, pathlib.Path], clear: bool = False
) -> Callable[[tf.data.Dataset], tf.data.TFRecordDataset]:
    """
    Similar to tf.data.Dataset.cache but writes a tf record file instead. Compared to base .cache method, it also insures that the whole
    dataset is cached

    Args:
        filename: path to the tf record file. The function calls mkdir on parent directory.
        clear: whether to enforce writing a new tf record file. Default parameter behaves like base tf.data.Dataset.cache method:
            if a file is found, it reads from it.
    """

    def _cache(dataset):
        if not isinstance(dataset.element_spec, dict):
            raise ValueError(f"dataset.element_spec should be a dict but is {type(dataset.element_spec)} instead")
        if clear or not Path(filename).is_file():
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with tf.io.TFRecordWriter(str(filename)) as writer:
                for sample in dataset.map(transform(**{name: tf.io.serialize_tensor for name in dataset.element_spec.keys()})):
                    writer.write(
                        tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    key: tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))
                                    for key, value in sample.items()
                                }
                            )
                        ).SerializeToString()
                    )
        return (
            tf.data.TFRecordDataset(str(filename), num_parallel_reads=tf.data.experimental.AUTOTUNE)
            .map(
                partial(
                    tf.io.parse_single_example,
                    features={name: tf.io.FixedLenFeature((), tf.string) for name in dataset.element_spec.keys()},
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .map(
                transform(
                    **{name: partial(tf.io.parse_tensor, out_type=spec.dtype) for name, spec in dataset.element_spec.items()}
                )
            )
            .map(
                transform(**{name: partial(tf.ensure_shape, shape=spec.shape) for name, spec in dataset.element_spec.items()})
            )
        )

    return _cache


def clear_cache(filename):
    """
    Clear cache created with tf.data.Dataset.cache given the name used for cache creation:
        e.g. dataset.cache(filename) will produce filename.index and filename.data-xxx files
    Args:
        filename (Union[str, pathlib.Path]): filename used during cache creation

    Returns:
        List[pathlib.Path]: list of deleted items
    """
    filename = Path(filename)
    files = list(filename.parent.glob(f"{filename.name}.*"))
    for file in files:
        file.unlink()
    return files


def cache(filename: Union[str, pathlib.Path], clear: bool = False) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    """
    Similar to tf.data.Dataset.cache but iterates over the whole dataset to insure everything is cached (see Note of
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache)

    Args:
        filename: path to cache file (see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache)
        clear: set to True to remove existing files
    """

    def _cache(dataset):
        if clear:
            clear_cache(filename)
        dataset = dataset.cache(str(filename))
        for _ in dataset:
            continue
        return dataset

    return _cache
