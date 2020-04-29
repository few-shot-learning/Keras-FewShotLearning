import pandas as pd
import numpy as np
import tensorflow as tf

from keras_fsl.utils.tensors import get_dummies


class TestGetDummies:
    @staticmethod
    def test_should_handle_multi_dimensional_tensor():
        samples = tf.constant([[1, 2, 0, 2, 3, 0, 10, 10]])
        one_hot, columns = get_dummies(samples)
        dummies_df = pd.get_dummies(samples.numpy().flatten())[columns.numpy()]
        np.testing.assert_array_equal(dummies_df.values, one_hot.numpy())

    @staticmethod
    def test_should_handle_string_dtype():
        samples = tf.constant(["a", "b", "b", "a", "c"])
        one_hot, columns = get_dummies(samples)
        np.testing.assert_array_equal(pd.get_dummies(samples.numpy().flatten()).values, one_hot.numpy())
