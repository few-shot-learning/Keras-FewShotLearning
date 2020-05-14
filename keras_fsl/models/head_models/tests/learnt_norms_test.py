import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.keras_parameterized import TestCase, run_all_keras_modes, run_with_all_model_types

from keras_fsl.models.head_models import LearntNorms


class TestLearntNorms(TestCase):
    @run_all_keras_modes
    @run_with_all_model_types(exclude_models=["sequential"])
    @parameterized.named_parameters(("flat_input", (10,)), ("3D_input", (3, 3, 10)))
    def test_should_fit(self, input_shape):
        learnt_norms = LearntNorms(input_shape=input_shape)
        optimizer = RMSprop(learning_rate=0.001)
        learnt_norms.compile(optimizer, loss="binary_crossentropy")

        inputs = [np.zeros((1, *input_shape)), np.zeros((1, *input_shape))]
        targets = np.zeros((1,))
        dataset = (
            tf.data.Dataset.from_tensor_slices((*inputs, targets))
            .map(lambda x_0, x_1, y: ({learnt_norms.input_names[0]: x_0, learnt_norms.input_names[1]: x_1}, y))
            .repeat()
            .batch(10)
        )

        learnt_norms.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)


if __name__ == "__main__":
    tf.test.main()
