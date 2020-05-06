from unittest.mock import sentinel

from keras_fsl.layers.gram_matrix import GramMatrix


class TestGramMatrix:
    class TestBuildSupportSet:
        @staticmethod
        def test_should_return_inputs():
            support_set = GramMatrix(kernel=sentinel.kernel).build_support_set(sentinel.inputs)
            assert support_set == sentinel.inputs
