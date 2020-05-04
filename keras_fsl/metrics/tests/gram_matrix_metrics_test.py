import numpy as np
import pytest
import tensorflow as tf

from ..gram_matrix_metrics import classification_accuracy


def perfect_prediction_from_classes(classes, n):
    y_true = tf.one_hot(classes, n)
    return y_true @ tf.transpose(y_true)


class TestMetrics:
    class TestTopScoreClassificationAccuracy:
        @staticmethod
        @pytest.mark.parametrize(
            "predicted_classes,accuracy", [([0, 0, 2, 2, 1, 1], 1), ([0, 0, 2, 1, 2, 1], 1 / 3), ([0, 1, 0, 1, 0, 1], 0)]
        )
        def test_return_proper_result(predicted_classes, accuracy):
            y_true = tf.one_hot([0, 0, 2, 2, 1, 1], 5)
            y_pred = perfect_prediction_from_classes(predicted_classes, 5)
            np.testing.assert_almost_equal(classification_accuracy()(y_true, y_pred), accuracy)
