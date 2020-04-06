from typing import Callable
import tensorflow as tf

from keras_fsl.utils.types import TENSOR_MAP, TF_TENSOR


def add_field(produce: Callable[[TENSOR_MAP], TF_TENSOR], key: str) -> Callable[[TENSOR_MAP], TENSOR_MAP]:
    def annotations_mapper(annotations: TENSOR_MAP) -> TENSOR_MAP:
        return {
            **annotations,
            key: produce(annotations),
        }

    return annotations_mapper


def load_crop_as_ndarray(annotation: TENSOR_MAP) -> TF_TENSOR:
    """
    Args:
        annotation (dict): with keys 'image_name': path to the image and 'crop_window' to be passed to tf.io.decode_and_crop_jpeg
    Returns:
        np.ndarray: the crop described by annotations as np array
    """
    return tf.io.decode_and_crop_jpeg(
        tf.io.read_file(annotation["image_name"]), crop_window=annotation["crop_window"], channels=3
    )


def load_raw_crop(annotation: TENSOR_MAP) -> TF_TENSOR:
    """
    Args:
        annotation (dict): with keys 'image_name': path to the image and 'crop_window' to be passed to tf.io.decode_and_crop_jpeg
    Returns:
        str: the crop described by annotations as unicode string
    """
    return tf.image.encode_jpeg(load_crop_as_ndarray(annotation))


def raw_image_to_numpy_array(annotation: TENSOR_MAP) -> TF_TENSOR:
    """
    Args:
        annotation (dict): with keys 'image': containing the image as a raw string
    Returns:
        np.ndarray: the crop described by annotations as np array
    """
    return tf.image.decode_jpeg(annotation["image"])
