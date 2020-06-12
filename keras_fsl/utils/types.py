from typing import Mapping, List
import tensorflow as tf

TF_TENSOR = tf.Tensor
TF_TENSOR_SPEC = tf.TensorSpec
TENSOR_DTYPE_STR = str
TENSOR_SHAPE = List[int]
TENSOR_NDIM = int
TENSOR_MAP = Mapping[str, TF_TENSOR]
TENSOR_SPEC_MAP = Mapping[str, TF_TENSOR_SPEC]
