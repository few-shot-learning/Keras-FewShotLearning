from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from keras_fsl.utils.tfrecord_utils import DTYPE_TO_PROTO_DTYPE, infer_tfrecord_encoder_decoder_from_sample, clear_cache


class TestTFRecordUtils:
    class TestInferTFRecordEncoderDecoder:
        @staticmethod
        @pytest.fixture
        def make_multi_features_sample(make_tensor_for_dtype_shape):
            return lambda: {
                "string_feature": make_tensor_for_dtype_shape([], "string"),
                "int_feature": make_tensor_for_dtype_shape([], "int64"),
                "int_list_feature": make_tensor_for_dtype_shape([5], "int32"),
            }

        @staticmethod
        @pytest.mark.parametrize("dtype", [dtype.name for dtype in DTYPE_TO_PROTO_DTYPE.keys()])
        def test_encode_decode_scalar_tensor_with_eager_execution(dtype, make_tensor_for_dtype_shape):
            shape = []
            tensor = make_tensor_for_dtype_shape(shape, dtype)
            sample = {"feature": tensor}
            encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

            np.testing.assert_array_equal(decoder(encoder(sample))["feature"].numpy(), tensor.numpy())

        @staticmethod
        @pytest.mark.parametrize("dtype", [dtype.name for dtype in DTYPE_TO_PROTO_DTYPE.keys() if dtype is not tf.string])
        def test_encode_decode_1d_tensor_with_eager_execution(dtype, make_tensor_for_dtype_shape):
            shape = [5]
            tensor = make_tensor_for_dtype_shape(shape, dtype)
            sample = {"feature": tensor}
            encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

            np.testing.assert_array_equal(decoder(encoder(sample))["feature"].numpy(), tensor.numpy())

        @staticmethod
        def test_encode_decode_multiple_tensors_with_eager_execution(make_multi_features_sample):
            sample = make_multi_features_sample()
            encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(sample)

            result = decoder(encoder(sample))

            assert result.keys() == sample.keys()
            for key, tensor in sample.items():
                np.testing.assert_array_equal(result[key].numpy(), tensor.numpy())

        @staticmethod
        @pytest.fixture
        def dataframe():
            return pd.DataFrame(
                {
                    "image_name": [f"data/im_{i}.jpg" for i in range(5)],
                    "label": ["DOG", "CAT", "FISH", "FISH", "DOG"],
                    "split": "val",
                    **{column: np.random.randint(100, 600, 5) for column in ["crop_x", "crop_y", "crop_height", "crop_width"]},
                }
            ).assign(crop_window=lambda df: df[["crop_y", "crop_x", "crop_height", "crop_width"]].values.tolist())

        @staticmethod
        def test_infer_tfrecord_encoder_decoder_generalize_from_sample_to_sample(dataframe, tmp_path):
            filename = tmp_path / "example.tfrecord"
            original_dataset = tf.data.Dataset.from_tensor_slices(dataframe.to_dict("list"))

            first_sample = next(iter(original_dataset))
            encoder, decoder = infer_tfrecord_encoder_decoder_from_sample(first_sample)
            with tf.io.TFRecordWriter(str(filename)) as writer:
                for sample in original_dataset:
                    writer.write(encoder(sample))

            parsed_dataset = tf.data.TFRecordDataset(str(filename), num_parallel_reads=tf.data.experimental.AUTOTUNE).map(
                decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

            for i, (original_sample, parsed_sample) in enumerate(zip(original_dataset, parsed_dataset)):
                assert parsed_sample.keys() == original_sample.keys()
                for key in original_sample:
                    np.testing.assert_array_equal(parsed_sample[key].numpy(), original_sample[key].numpy())

    class TestClearCache:
        @staticmethod
        def test_should_delete_files_after_cache(tmpdir):
            filename = tmpdir.join("filename")
            dataset = tf.data.Dataset.range(4).cache(str(filename))
            list(dataset.as_numpy_iterator())
            cache_files = list(Path(tmpdir).glob("*"))
            deleted_files = clear_cache(filename)
            assert cache_files == deleted_files

        @staticmethod
        def test_should_not_delete_other_files(tmpdir):
            filename = tmpdir.join("filename")
            other_filename = tmpdir.join("other_filename")
            other_filename.write("content")
            clear_cache(filename)
            assert Path(other_filename).is_file()

        @staticmethod
        def test_should_not_delete_other_files_with_same_prefix(tmpdir):
            filename = tmpdir.join("filename")
            filename.write("content")
            filename_suffix = tmpdir.join("filename_suffix")
            filename_suffix.write("content")
            deleted_files = clear_cache(filename)
            assert Path(filename_suffix).is_file()
            assert deleted_files == []
