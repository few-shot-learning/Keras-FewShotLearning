import numpy as np
import pytest
import tensorflow as tf

from ..gram_matrix_metrics import top_score_classification_accuracy


class TestMetrics:
    class TestTopScoreClassificationAccuracy:
        @staticmethod
        @pytest.fixture
        def y_true():
            n_classes = 5
            classes = [0, 0, 2, 2, 1, 1]
            return tf.one_hot(classes, n_classes)

        @staticmethod
        def test_on_perfect_prediction(y_true):
            y_pred = y_true @ tf.transpose(y_true)
            np.testing.assert_almost_equal(top_score_classification_accuracy(y_true, y_pred), 1)

        @staticmethod
        def test_on_normal_prediction(y_true):
            n_classes = 5
            predicted_classes = [0, 0, 2, 1, 2, 1]
            one_hot_prediction = tf.one_hot(predicted_classes, n_classes)
            y_pred = one_hot_prediction @ tf.transpose(one_hot_prediction)
            np.testing.assert_almost_equal(top_score_classification_accuracy(y_true, y_pred), 1 / 3)
