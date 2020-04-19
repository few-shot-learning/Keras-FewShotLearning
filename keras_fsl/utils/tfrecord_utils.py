from pathlib import Path
from typing import Any, Callable, Mapping, Tuple, Union

import tensorflow as tf

from keras_fsl.utils.types import TENSOR_MAP, TF_TENSOR

ENCODER_TYPE = Callable[[TENSOR_MAP], bytes]
DECODER_TYPE = Callable[[TF_TENSOR], TENSOR_MAP]
FEATURE_DESCRIPTOR = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
ENCODE_FEATURE = Callable[[TF_TENSOR], FEATURE_DESCRIPTOR]


def _int64_feature(value: TF_TENSOR):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value: TF_TENSOR):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _byte_feature(value: TF_TENSOR):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


PROTO_DTYPE_TO_FEATURE = {
    tf.string: _byte_feature,
    tf.int64: _int64_feature,
    tf.float32: _float_feature,
}

DTYPE_TO_PROTO_DTYPE = {
    tf.string: tf.string,
    tf.int32: tf.int64,
    tf.uint32: tf.int64,
    tf.int64: tf.int64,
    tf.uint64: tf.int64,
    tf.float32: tf.float32,
}


def apply_on_scalar_tensor(f: ENCODE_FEATURE) -> ENCODE_FEATURE:
    def _f(value):
        return f([value.numpy()])

    return _f


def encoder_factory_from_dict(map: Mapping[str, Callable[[TF_TENSOR], tf.train.Feature]]) -> ENCODER_TYPE:
    def _encoder(sample: TENSOR_MAP):
        return tf.train.Example(
            features=tf.train.Features(feature={key: make_feature(sample[key]) for key, make_feature in map.items()})
        ).SerializeToString()

    return _encoder


def decoder_factory_from_dict(feature_map: Mapping[str, FEATURE_DESCRIPTOR], dtype_map: Mapping[str, Any]) -> DECODER_TYPE:
    def _decoder(sample: TF_TENSOR):
        return {
            key: tf.cast(tensor, dtype=dtype_map[key])
            for key, tensor in tf.io.parse_single_example(sample, feature_map).items()
        }

    return _decoder


def infer_tfrecord_encoder_decoder_from_sample(sample: TENSOR_MAP) -> Tuple[ENCODER_TYPE, DECODER_TYPE]:
    # TODO : support variable length arrays
    encode_function_map = {}
    decode_feature_map = {}
    dtype_map = {}

    for key, tensor in sample.items():
        dtype = tensor.dtype
        shape = tensor.shape.as_list()

        if len(shape) > 1:
            # TODO : support dimensions non 1d or scalar tensors
            raise TypeError(f"infer_tfrecord_encoder_decoder_from_sample does not support {len(shape)}d tensors")
        if len(shape) == 1 and dtype is tf.string:
            # TODO : support 1d arrays of string (currently converted to bytes)
            raise TypeError(f"infer_tfrecord_encoder_decoder_from_sample only support scalar strings")

        proto_dtype = DTYPE_TO_PROTO_DTYPE[dtype]
        encode = PROTO_DTYPE_TO_FEATURE[proto_dtype]
        if not shape:
            encode = apply_on_scalar_tensor(encode)
        encode_function_map[key] = encode
        decode_feature_map[key] = tf.io.FixedLenFeature(shape, proto_dtype)
        dtype_map[key] = dtype

    return encoder_factory_from_dict(encode_function_map), decoder_factory_from_dict(decode_feature_map, dtype_map)


def clear_cache(filename):
    """
    Clear cache created with tfrecord given the name used for cache creation:
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
