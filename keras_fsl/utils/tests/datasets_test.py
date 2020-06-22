from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from keras_fsl.utils.datasets import (
    assign,
    cache,
    cache_with_tf_record,
    clear_cache,
    filter_items,
    read_decode_and_crop_jpeg,
    transform,
)


class TestDatasetsUtils:
    class TestAssign:
        @staticmethod
        def test_should_add_given_key():
            assert (
                list(
                    tf.data.Dataset.from_tensor_slices({"key": tf.constant([2])})
                    .map(assign(new_key=lambda x: tf.math.square(x["key"])))
                    .as_numpy_iterator()
                )[0]["new_key"]
                == 4
            )

    class TestTransform:
        @staticmethod
        def test_should_update_given_key():
            assert (
                list(
                    tf.data.Dataset.from_tensor_slices({"key": tf.constant([2])})
                    .map(transform(key=tf.math.square))
                    .as_numpy_iterator()
                )[0]["key"]
                == 4
            )

    class TestFilterItems:
        @staticmethod
        def test_should_return_filtered_dataframe():
            assert list(
                tf.data.Dataset.from_tensor_slices({"key_0": [0], "key_1": [1]})
                .map(filter_items(["key_1"]))
                .element_spec.keys()
            ) == ["key_1"]

    class TestReadDecodeAndCropJpeg:
        @staticmethod
        @pytest.fixture
        def filename():
            return tf.keras.utils.get_file(
                "cat.jpg",
                "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg",
            )

        @staticmethod
        def test_should_return_full_image(filename):
            tf.debugging.assert_equal(
                tf.io.decode_jpeg(tf.io.read_file(filename)), read_decode_and_crop_jpeg({"filename": tf.constant(filename)})
            )

        @staticmethod
        def test_should_return_cropped_image(filename):
            tf.debugging.assert_equal(
                tf.io.decode_jpeg(tf.io.read_file(filename))[:50, :100, ...],
                read_decode_and_crop_jpeg({"filename": tf.constant(filename), "crop_window": [0, 0, 50, 100]}),
            )

    class TestCacheWithTfRecord:
        @staticmethod
        @pytest.fixture
        def input_tensor():
            tf.random.set_seed(0)
            return tf.random.normal((10, 10))

        @staticmethod
        def test_should_raise_with_dataset_without_named_features(tmp_path, input_tensor):
            with pytest.raises(ValueError) as value_error:
                tf.data.Dataset.from_tensor_slices(input_tensor).apply(cache_with_tf_record(tmp_path))
            assert (
                str(value_error.value)
                == "dataset.element_spec should be a dict but is <class 'tensorflow.python.framework.tensor_spec.TensorSpec'> instead"
            )

        @staticmethod
        def test_should_create_tf_record_file(tmp_path, input_tensor):
            tf.data.Dataset.from_tensor_slices({"input_tensor": input_tensor}).apply(
                cache_with_tf_record(tmp_path / "dataset")
            )
            assert (tmp_path / "dataset").is_file()

        @staticmethod
        def test_should_create_directory(tmp_path, input_tensor):
            tf.data.Dataset.from_tensor_slices({"input_tensor": input_tensor}).apply(
                cache_with_tf_record(tmp_path / "sub_dir" / "dataset")
            )
            assert (tmp_path / "sub_dir" / "dataset").is_file()

        @staticmethod
        def test_should_return_dataset_with_same_values(tmp_path, input_tensor):
            dataset = tf.data.Dataset.from_tensor_slices({"input_tensor": input_tensor})
            cached_dataset = dataset.apply(cache_with_tf_record(tmp_path / "dataset"))
            np.testing.assert_array_equal(
                np.array([example["input_tensor"] for example in dataset]),
                np.array([example["input_tensor"] for example in cached_dataset]),
            )

        @staticmethod
        @patch("keras_fsl.utils.datasets.tf.data.TFRecordDataset", wraps=tf.data.TFRecordDataset)
        def test_should_return_a_tf_record_dataset(tf_record_dataset, tmp_path, input_tensor):
            tf.data.Dataset.from_tensor_slices({"input_tensor": input_tensor}).apply(
                cache_with_tf_record(tmp_path / "dataset")
            )
            tf_record_dataset.assert_called_once_with(str(tmp_path / "dataset"), num_parallel_reads=-1)

    class TestCache:
        @staticmethod
        def test_should_cache_all_dataset_with_partial_iteration(tmp_path):
            dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal((10, 5)))
            list(dataset.cache(str(tmp_path / "tf_cache")).take(1).as_numpy_iterator())
            list(dataset.apply(cache(tmp_path / "keras_fsl_cache")).take(1).as_numpy_iterator())

            cached_dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal((1, 5)))
            tf_cache_dataset = cached_dataset.cache(str(tmp_path / "tf_cache"))
            assert len(list(tf_cache_dataset.as_numpy_iterator())) == 1

            keras_fsl_dataset = cached_dataset.apply(cache(tmp_path / "keras_fsl_cache"))
            assert len(list(keras_fsl_dataset.as_numpy_iterator())) == 10

        @staticmethod
        @patch("keras_fsl.utils.datasets.clear_cache")
        def test_should_clear_cache(mock_clear_cache, tmp_path):
            tf.data.Dataset.from_tensor_slices(tf.random.normal((10, 5))).apply(cache(tmp_path / "cache", clear=True))
            mock_clear_cache.assert_called_once_with(tmp_path / "cache")

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
