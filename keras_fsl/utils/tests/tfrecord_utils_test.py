import pytest
import numpy as np
import tensorflow as tf

from keras_fsl.utils.tfrecord_utils import infer_tfrecord_encoder_decoder_from_sample, DTYPE_TO_PROTO_DTYPE


class TestTFRecordUtils:
    @staticmethod
    @pytest.mark.parametrize("dtype", [dtype.name for dtype in DTYPE_TO_PROTO_DTYPE.keys()])
    def test_encode_decode_scalar_tensor_with_eager_execution(dtype, make_tensor_for_dtype_shape):
        shape = []
        tensor = make_tensor_for_dtype_shape(dtype, shape)
        sample = {"feature": tensor}
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

        np.testing.assert_array_equal(decoder(encoder(sample))["feature"].numpy(), tensor.numpy())

    @staticmethod
    @pytest.mark.parametrize("dtype", [dtype.name for dtype in DTYPE_TO_PROTO_DTYPE.keys() if dtype is not tf.string])
    def test_encode_decode_1d_tensor_with_eager_execution(dtype, make_tensor_for_dtype_shape):
        shape = [5]
        tensor = make_tensor_for_dtype_shape(dtype, shape)
        sample = {"feature": tensor}
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

        np.testing.assert_array_equal(decoder(encoder(sample))["feature"].numpy(), tensor.numpy())

    @staticmethod
    def test_encode_decode_multiple_tensors_with_eager_execution(make_tensor_for_dtype_shape):
        sample = {
            "string_feature": make_tensor_for_dtype_shape("string", []),
            "int_feature": make_tensor_for_dtype_shape("int64", []),
            "int_list_feature": make_tensor_for_dtype_shape("int32", [5]),
        }
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

        result = decoder(encoder(sample))

        assert result.keys() == sample.keys()
        for key, tensor in sample.items():
            np.testing.assert_array_equal(result[key].numpy(), tensor.numpy())

    @staticmethod
    def test_infer_tfrecord_encoder_decoder_generalize_from_sample_to_sample(make_tensor_for_dtype_shape):
        reference_sample = {
            "string_feature": make_tensor_for_dtype_shape("string", []),
            "int_feature": make_tensor_for_dtype_shape("int64", []),
            "int_list_feature": make_tensor_for_dtype_shape("int32", [5]),
        }
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(reference_sample)

        new_sample = {
            "string_feature": make_tensor_for_dtype_shape("string", []),
            "int_feature": make_tensor_for_dtype_shape("int64", []),
            "int_list_feature": make_tensor_for_dtype_shape("int32", [5]),
        }

        result = decoder(encoder(new_sample))

        assert result.keys() == new_sample.keys()
        for key, tensor in new_sample.items():
            np.testing.assert_array_equal(result[key].numpy(), tensor.numpy())
