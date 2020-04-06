from typing import Callable
import tensorflow as tf

from keras_fsl.utils.types import TENSOR_MAP, TF_TENSOR


def add_field(produce: Callable[[TENSOR_MAP], TF_TENSOR], key: str) -> Callable[[TENSOR_MAP], TENSOR_MAP]:
    """Wrap a tensor produce function to create a new field in the TENSOR_MAP"""
    def annotations_mapper(annotations: TENSOR_MAP) -> TENSOR_MAP:
        return {
            **annotations,
            key: produce(annotations),
        }

    return annotations_mapper


def transform_field(transform: Callable[[TF_TENSOR], TF_TENSOR], key: str) -> Callable[[TENSOR_MAP], TENSOR_MAP]:
    """Wrap a tensor transform function to apply it on a specific field of a TENSOR_MAP"""
    def annotations_mapper(annotations: TENSOR_MAP) -> TF_TENSOR:
        return transform(annotations[key])

    return add_field(annotations_mapper, key)


####
#
# produce functions
#
####

def load_crop_as_ndarray(annotation: TENSOR_MAP) -> TF_TENSOR:
    """
    Args:
        annotation (TENSOR_MAP): with keys 'image_name': path to the image and 'crop_window' to be passed to tf.io.decode_and_crop_jpeg
    Returns:
        TF_TENSOR: the crop described by annotations
    """
    return tf.io.decode_and_crop_jpeg(
        tf.io.read_file(annotation["image_name"]), crop_window=annotation["crop_window"], channels=3
    )


def load_raw_crop(annotation: TENSOR_MAP) -> TF_TENSOR:
    """Decode the base64 jpeg image in the `image` field"""
    return tf.image.encode_jpeg(load_crop_as_ndarray(annotation))


def raw_image_to_numpy_array(annotation: TENSOR_MAP) -> TF_TENSOR:
    """Decode the base64 jpeg image in the `image` field"""
    return tf.image.decode_jpeg(annotation["image"])
