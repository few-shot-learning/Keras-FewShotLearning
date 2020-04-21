import pytest
import tensorflow as tf
import numpy as np

from keras_fsl.utils.types import TENSOR_DTYPE_STR, TENSOR_SHAPE, TF_TENSOR


@pytest.fixture
def make_tensor_for_number_dtype_shape():
    def _make_tensor(shape: TENSOR_SHAPE, dtype: TENSOR_DTYPE_STR) -> TF_TENSOR:
        return tf.constant(np.random.randint(0, 500, shape), dtype=dtype)

    return _make_tensor


@pytest.fixture
def make_tensor_for_string_shape():
    strings = ["string1", "string2", "hello", "/this/is/a/path"]

    def _make_tensor(shape: TENSOR_SHAPE) -> TF_TENSOR:
        return tf.constant(np.random.choice(strings, shape))

    return _make_tensor


@pytest.fixture
def make_tensor_for_dtype_shape(make_tensor_for_number_dtype_shape, make_tensor_for_string_shape):
    def _make_tensor(shape: TENSOR_SHAPE, dtype: TENSOR_DTYPE_STR) -> TF_TENSOR:
        if dtype == "string":
            return make_tensor_for_string_shape(shape)
        return make_tensor_for_number_dtype_shape(shape, dtype)

    return _make_tensor
