import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from keras_fsl.losses.gram_matrix_losses import (
    ClassConsistencyLoss,
    MeanScoreClassificationLoss,
    TripletLoss,
    BinaryCrossentropy,
    ClippedBinaryCrossentropy,
    MaxBinaryCrossentropy,
    StdBinaryCrossentropy,
    PercentileBinaryCrossentropy,
)
from keras_fsl.utils.tensors import get_dummies


class TestGramMatrixLoss:
    class TestTripletLoss:
        @staticmethod
        def test_loss_should_equal_literal_calculation_for_semi_hard_mining():
            np.random.seed(0)
            margin = 1
            batch_size = 64
            distance_matrix = np.random.rand(batch_size, batch_size)
            distance_matrix = distance_matrix + distance_matrix.T  # distance matrix is symmetric
            labels = np.random.choice(list(range(4)), batch_size)

            # compute loss with for loop
            loss = []
            for anchor_index in range(batch_size):
                for positive_index in range(batch_size):
                    if positive_index == anchor_index:
                        continue
                    if labels[positive_index] == labels[anchor_index]:
                        positive_distance = distance_matrix[anchor_index, positive_index]
                        negative_distances = []
                        for negative_index in range(batch_size):
                            if labels[negative_index] == labels[anchor_index]:
                                continue
                            negative_distances += [distance_matrix[anchor_index, negative_index]]
                        negative_distances = np.array(negative_distances)
                        if np.any(negative_distances - positive_distance > 0):
                            negative_distance = np.min(negative_distances[negative_distances - positive_distance > 0])
                        else:
                            print(anchor_index)
                            print(positive_index)
                            negative_distance = np.max(negative_distances)
                        loss += [positive_distance - negative_distance + margin]
            np_loss = np.mean(loss)

            # assert value is equal
            y_true = get_dummies(labels)[0]
            y_pred = tf.convert_to_tensor(distance_matrix, dtype=tf.float32)
            tf_loss = TripletLoss(margin)(y_true, y_pred)
            np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)

    class TestMeanScoreClassificationLoss:
        def test_loss_should_equal_literal_calculation(self):
            np.random.seed(0)
            batch_size = 16
            labels = np.random.choice(["a", "b", "c"], batch_size)
            y_true = pd.get_dummies(labels).values
            y_pred = np.random.rand(batch_size, batch_size)
            np_loss = (
                -np.sum(
                    y_true
                    * np.log(
                        pd.melt(
                            pd.DataFrame(y_pred, columns=pd.Index(labels, name="support_label")).reset_index(),
                            id_vars=["index"],
                        )
                        .groupby(["index", "support_label"])
                        .agg("mean")
                        .groupby(level="index")
                        .transform(lambda x: x / x.sum())
                        .unstack("support_label")
                        .values
                    )
                )
                / batch_size
            )
            tf_loss = MeanScoreClassificationLoss()(
                tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32)
            )
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=5)

    class TestClassConsistencyLoss:
        def test_loss_should_equal_literal_calculation(self):
            np.random.seed(0)
            batch_size = 16
            labels = np.random.choice(["a", "b", "c"], batch_size)
            y_true = pd.get_dummies(labels).values
            y_pred = np.random.rand(batch_size, batch_size)
            classes_scores = (
                pd.melt(
                    pd.DataFrame(
                        y_pred, columns=pd.Index(labels, name="support_label"), index=pd.Index(labels, name="query_label"),
                    ).reset_index(),
                    id_vars=["query_label"],
                )
                .groupby(["query_label", "support_label"])
                .agg("mean")
                .unstack("support_label")
                .values
            )
            np_loss = -np.mean(np.eye(3) * np.log(classes_scores) + (1 - np.eye(3)) * np.log(1 - classes_scores))
            tf_loss = ClassConsistencyLoss()(
                tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32)
            )
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=5)

    class TestBinaryCrossentropy:
        @pytest.fixture
        def y_true(self):
            np.random.seed(0)
            labels = np.random.choice(["a", "b", "c"], 16)
            return pd.get_dummies(labels).values

        @pytest.fixture
        def adjacency_matrix(self, y_true):
            return y_true @ y_true.T

        @pytest.fixture
        def y_pred(self):
            np.random.seed(0)
            return np.random.rand(16, 16)

        def test_mean_loss_should_equal_literal_calculation(self, y_true, adjacency_matrix, y_pred):
            np_loss = -np.mean(adjacency_matrix * np.log(y_pred) + (1 - adjacency_matrix) * np.log(1 - y_pred))
            tf_loss = BinaryCrossentropy()(tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32))
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=5)

        def test_clipped_loss_should_equal_literal_calculation(self, y_true, adjacency_matrix, y_pred):
            clip_mask = ((0.05 < np.abs(adjacency_matrix - y_pred)) * (np.abs(adjacency_matrix - y_pred) < 0.95)).astype(int)
            np_loss = np.mean(-clip_mask * (adjacency_matrix * np.log(y_pred) + (1 - adjacency_matrix) * np.log(1 - y_pred)))
            tf_loss = ClippedBinaryCrossentropy(lower=0.05, upper=0.95)(
                tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32)
            )
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=5)

        @staticmethod
        @pytest.mark.parametrize("dtype_policy", (tf.float16, tf.bfloat16, tf.float32, tf.float64))
        def test_clipped_loss_computes_in_all_float_dtypes(dtype_policy, y_true, y_pred):
            y_true_tensor = tf.convert_to_tensor(y_true)
            y_pred_tensor = tf.convert_to_tensor(y_pred)
            ClippedBinaryCrossentropy(lower=0.05, upper=0.95)(
                tf.cast(y_true_tensor, dtype=dtype_policy), tf.cast(y_pred_tensor, dtype=dtype_policy)
            )

        def test_max_loss_should_equal_literal_calculation(self, y_true, adjacency_matrix, y_pred):
            np_loss = np.max(-(adjacency_matrix * np.log(y_pred) + (1 - adjacency_matrix) * np.log(1 - y_pred)))
            tf_loss = MaxBinaryCrossentropy()(
                tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32)
            )
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=4)

        def test_std_loss_should_equal_literal_calculation(self, y_true, adjacency_matrix, y_pred):
            np_loss = np.std(-(adjacency_matrix * np.log(y_pred) + (1 - adjacency_matrix) * np.log(1 - y_pred)))
            tf_loss = StdBinaryCrossentropy()(
                tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32)
            )
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=4)

        def test_median_loss_should_equal_literal_calculation(self, y_true, adjacency_matrix, y_pred):
            np_loss = np.percentile(
                -(adjacency_matrix * np.log(y_pred) + (1 - adjacency_matrix) * np.log(1 - y_pred)),
                50,
                interpolation="midpoint",
            )
            tf_loss = PercentileBinaryCrossentropy()(
                tf.convert_to_tensor(y_true, tf.float32), tf.convert_to_tensor(y_pred, tf.float32)
            )
            np.testing.assert_almost_equal(tf_loss, np_loss, decimal=4)
