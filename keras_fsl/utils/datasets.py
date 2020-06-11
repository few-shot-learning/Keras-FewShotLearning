from typing import Callable

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


def read_decode_and_crop_jpeg(annotation: TENSOR_MAP) -> TF_TENSOR:
    """
    Args:
        annotation (TENSOR_MAP): contains 'image_name' (path to the image) and 'crop_window' (box coordinates)
    Returns:
        TF_TENSOR: the crop described by annotations as a uint8 array
    """
    return tf.io.decode_and_crop_jpeg(
        tf.io.read_file(filename=annotation["filename"]), crop_window=annotation["crop_window"], channels=3
    )
