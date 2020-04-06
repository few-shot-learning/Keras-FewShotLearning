from typing import Mapping, Type, List
import tensorflow as tf

TF_TENSOR = Type[tf.Tensor]
TENSOR_DTYPE_STR = str
TENSOR_SHAPE = List[int]
TENSOR_NDIM = int
TENSOR_MAP = Mapping[str, TF_TENSOR]
