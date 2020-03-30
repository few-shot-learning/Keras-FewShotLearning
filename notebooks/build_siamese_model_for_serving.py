import tensorflow as tf
from tensorflow.keras.models import load_model

#%% Load siamese nets
classifier = load_model("siamese_nets_classifier/1")
preprocessing = classifier.signatures["preprocessing"]


#%% Build saved_model signatures
@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes"),
        tf.TensorSpec(shape=[None, 4], dtype=tf.int32, name="crop_window"),
    )
)
def decode_and_crop(image_bytes, crop_window):
    # currently not working on GPU, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_and_crop_jpeg(contents=tf.io.decode_base64(x[0]), crop_window=x[1], channels=3))[
                "output_0"
            ],
            (image_bytes, crop_window),
            dtype=tf.float32,
        )
    return input_tensor


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes"),))
def decode(image_bytes):
    # currently not working on GPU, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_jpeg(contents=tf.io.decode_base64(x), channels=3))["output_0"],
            image_bytes,
            dtype=tf.float32,
        )
    return input_tensor


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string), tf.TensorSpec(shape=[None, 4], dtype=tf.int32)))
def decode_and_crop_and_serve(image_bytes, crop_window):
    return {
        tf.saved_model.CLASSIFY_OUTPUT_SCORES: classifier(decode_and_crop(image_bytes=image_bytes, crop_window=crop_window)),
        tf.saved_model.CLASSIFY_OUTPUT_CLASSES: classifier.layers[1].columns,
    }


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string),))
def decode_and_serve(image_bytes):
    return {
        tf.saved_model.CLASSIFY_OUTPUT_SCORES: classifier(decode(image_bytes=image_bytes)),
        tf.saved_model.CLASSIFY_OUTPUT_CLASSES: classifier.layers[1].columns,
    }


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes"),
        tf.TensorSpec(shape=[None, 4], dtype=tf.int32, name="crop_window"),
        tf.TensorSpec(shape=[None], dtype=tf.string, name="label"),
        tf.TensorSpec(shape=None, dtype=tf.bool, name="overwrite"),
    )
)
def set_support_set(image_bytes, crop_window, label, overwrite):
    support_tensors = classifier.layers[0](decode_and_crop(image_bytes=image_bytes, crop_window=crop_window))
    return classifier.layers[1].set_support_set(support_tensors=support_tensors, support_labels_name=label, overwrite=overwrite)


tf.saved_model.save(
    classifier,
    export_dir="siamese_nets_classifier/2",
    signatures={
        "serving_default": decode_and_crop_and_serve,
        "from_crop": decode_and_serve,
        "preprocessing": preprocessing,
        "set_support_set": set_support_set,
        "get_support_set": classifier.layers[1].get_support_set,
    },
)
