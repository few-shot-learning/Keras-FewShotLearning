from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import pytest
import tensorflow as tf

from keras_fsl.layers import CentroidsMatrix
from keras_fsl.utils.tensors import get_dummies


class TestCentroidsMatrix:
    class TestBuildSupportSet:
        @staticmethod
        def test_should_raise_value_error_when_inputs_is_not_list():
            with pytest.raises(ValueError) as error:
                CentroidsMatrix(kernel=sentinel.kernel).build_support_set(sentinel.inputs)
            assert str(error.value) == f"CentroidsMatrix should be called on a list of inputs [embeddings, labels_one_hot]"

        @staticmethod
        def test_should_raise_value_error_when_inputs_is_not_list_of_len_2():
            with pytest.raises(ValueError) as error:
                CentroidsMatrix(kernel=sentinel.kernel).build_support_set([sentinel.inputs])
            assert str(error.value) == f"CentroidsMatrix should be called on a list of inputs [embeddings, labels_one_hot]"

        @staticmethod
        def test_should_return_centroids_of_input_tensors_according_to_their_class():
            input_tensors = tf.cast(tf.tile(tf.expand_dims(tf.range(5), 1), [1, 4]), tf.float32)
            labels_one_hot = get_dummies(tf.constant([0, 0, 0, 1, 1]))[0]
            support_tensors = CentroidsMatrix(kernel=sentinel.kernel).build_support_set([input_tensors, labels_one_hot])
            np.testing.assert_array_equal([[1, 1, 1, 1], [3.5, 3.5, 3.5, 3.5]], support_tensors.numpy())

    class TestCall:
        @staticmethod
        @patch("keras_fsl.layers.centroids_matrix.SupportLayer.call")
        def test_should_call_activation_on_super_call(mock_super_call):
            layer = CentroidsMatrix(kernel=sentinel.kernel, activation=MagicMock(name="activation"))
            mock_super_call.return_value = sentinel.super_return
            layer.call(sentinel.inputs)
            mock_super_call.assert_called_once_with(sentinel.inputs)
            layer.activation.assert_called_once_with(sentinel.super_return)
