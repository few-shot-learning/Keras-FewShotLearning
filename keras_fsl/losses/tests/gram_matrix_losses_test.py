import tensorflow as tf
import numpy as np

from keras_fsl.losses.gram_matrix_losses import triplet_loss
from keras_fsl.utils.tensors import get_dummies


class TestGramMatrixLoss:
    class TestTripletLoss:
        @staticmethod
        def test_loss_should_equals_literal_calculation_for_semi_hard_mining():
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
            tf_loss = triplet_loss(margin)(y_true, y_pred)
            np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)
