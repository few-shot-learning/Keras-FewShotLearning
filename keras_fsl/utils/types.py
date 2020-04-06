from typing import Mapping, Type
import tensorflow as tf

TF_TENSOR = Type[tf.Tensor]
TENSOR_MAP = Mapping[str, TF_TENSOR]
