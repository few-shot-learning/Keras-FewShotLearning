import numpy as np
import pytest
import tensorflow as tf

from ..gram_matrix_metrics import top_score_classification_accuracy


def perfect_prediction_from_classes(classes, n):
    y_true = tf.one_hot(classes, n)
    return y_true @ tf.transpose(y_true)


class TestMetrics:
    class TestTopScoreClassificationAccuracy:
        @staticmethod
        def test_on_perfect_prediction():
            classes = [0, 0, 2, 2, 1, 1]
            y_true = tf.one_hot(classes, 5)
            y_pred = perfect_prediction_from_classes(classes, 5)
            np.testing.assert_almost_equal(top_score_classification_accuracy(y_true, y_pred), 1)

        @staticmethod
        def test_on_normal_prediction():
            y_true = tf.one_hot([0, 0, 2, 2, 1, 1], 5)
            y_pred = perfect_prediction_from_classes([0, 0, 2, 1, 2, 1], 5)
            np.testing.assert_almost_equal(top_score_classification_accuracy(y_true, y_pred), 1 / 3)

        @staticmethod
        def test_on_bad_predictions():
            y_true = tf.one_hot([0, 0, 2, 2, 1, 1], 5)
            y_pred = perfect_prediction_from_classes([0, 1, 0, 1, 0, 1], 5)
            np.testing.assert_almost_equal(top_score_classification_accuracy(y_true, y_pred), 0)
