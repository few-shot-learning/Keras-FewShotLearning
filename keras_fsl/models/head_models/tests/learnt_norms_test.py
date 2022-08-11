import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras.optimizers import RMSprop
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

    @parameterized.named_parameters(
        ("mixed_float16", "mixed_float16", "float32"),
        ("mixed_bfloat16", "mixed_bfloat16", "float32"),
        ("float32", "float32", "float32"),
        ("float64", "float64", "float64"),
    )
    def test_last_activation_fp32_in_mixed_precision(self, mixed_precision_policy, expected_last_layer_dtype_policy):
        policy = tf.keras.mixed_precision.Policy(mixed_precision_policy)
        tf.keras.mixed_precision.set_policy(policy)
        learnt_norms = LearntNorms(input_shape=(10,))

        # Check dtype policy of internal non-input layers
        for layer in learnt_norms.layers[2:-1]:
            assert layer._dtype_policy.name == mixed_precision_policy

        # Check dtype policy of last layer always at least FP32
        assert learnt_norms.layers[-1]._dtype_policy.name == expected_last_layer_dtype_policy


if __name__ == "__main__":
    tf.test.main()
