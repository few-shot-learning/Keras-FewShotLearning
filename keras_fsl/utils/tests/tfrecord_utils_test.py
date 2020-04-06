import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from keras_fsl.utils.tfrecord_utils import infer_tfrecord_encoder_decoder_from_sample, DTYPE_TO_PROTO_DTYPE


class TestTFRecordUtils:
    @staticmethod
    @pytest.fixture
    def dataframe():
        return pd.DataFrame({
            "image_name": [f'data/im_{i}.jpg' for i in range(5)],
            "label": ['DOG', 'CAT', 'FISH', 'FISH', 'DOG'],
            "split": "val",
            **{
                column: np.random.randint(100, 600, 5)
                for column in ["crop_x", "crop_y", "crop_height", "crop_width"]
            },
        })

    @staticmethod
    @pytest.mark.parametrize("dtype", [dtype.name for dtype in DTYPE_TO_PROTO_DTYPE.keys()])
    def test_properly_encode_decode_scalar_tensor_with_eager_execution(dtype, make_tensor_for_dtype_shape):
        shape = []
        tensor = make_tensor_for_dtype_shape(dtype, shape)
        sample = {"feature": tensor}
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

        assert decoder(encoder(sample))["feature"].numpy() == tensor.numpy()

    @staticmethod
    @pytest.mark.parametrize("dtype", [dtype.name for dtype in DTYPE_TO_PROTO_DTYPE.keys() if dtype is not tf.string])
    def test_properly_encode_decode_1d_tensor_with_eager_execution(dtype, make_tensor_for_dtype_shape):
        shape = [5]
        tensor = make_tensor_for_dtype_shape(dtype, shape)
        sample = {"feature": tensor}
        encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

        assert (decoder(encoder(sample))["feature"].numpy() == tensor.numpy()).all()
