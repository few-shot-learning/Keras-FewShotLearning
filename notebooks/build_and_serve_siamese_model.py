import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img, array_to_img

#%% Load siamese nets
classifier = load_model("siamese_nets_classifier")
preprocessing = classifier.signatures["preprocessing"]


#%% Build saved_model signatures
@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="image_name"),
        tf.TensorSpec(shape=[None, 4], dtype=tf.int32, name="crop_window"),
    )
)
def decode_and_crop(image_name, crop_window):
    # currently not working on GPU, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_and_crop_jpeg(contents=tf.io.read_file(x[0]), crop_window=x[1], channels=3))[
                "output_0"
            ],
            (image_name, crop_window),
            dtype=tf.float32,
        )
    return input_tensor


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string, name="image_name"),))
def decode(image_name):
    # currently not working on GPU, see https://github.com/tensorflow/tensorflow/issues/28007
    with tf.device("/cpu:0"):
        input_tensor = tf.map_fn(
            lambda x: preprocessing(tf.io.decode_jpeg(contents=tf.io.read_file(x), channels=3))["output_0"],
            image_name,
            dtype=tf.float32,
        )
    return input_tensor


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string), tf.TensorSpec(shape=[None, 4], dtype=tf.int32)))
def decode_and_crop_and_serve(image_name, crop_window):
    return classifier(decode_and_crop(image_name=image_name, crop_window=crop_window))


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string),))
def decode_and_serve(image_name):
    return classifier(decode(image_name=image_name))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="image_name"),
        tf.TensorSpec(shape=[None, 4], dtype=tf.int32, name="crop_window"),
        tf.TensorSpec(shape=[None, None], dtype=tf.uint8, name="label"),
    )
)
def set_support_set(image_name, crop_window, label):
    support_tensors = classifier.layers[0](decode_and_crop(image_name=image_name, crop_window=crop_window))
    return classifier.layers[1].set_support_set(support_tensors=support_tensors, support_labels=tf.cast(label, tf.float32))


tf.saved_model.save(
    classifier,
    export_dir="siamese_nets_classifier",
    signatures={
        "serving_default": decode_and_crop_and_serve,
        "from_crop": decode_and_serve,
        "preprocessing": preprocessing,
        "set_support_set": set_support_set,
    },
)

#%% Load model
classifier = tf.saved_model.load("siamese_nets_classifier")

#%% Build fake catalog
[save_img(f"catalog_{i}.jpg", array_to_img(np.random.rand(224, 224, 3))) for i in range(10)]

catalog = tf.convert_to_tensor([f"catalog_{i}.jpg" for i in range(10)])
label = tf.one_hot(np.random.choice(range(3), 10), depth=3, dtype=tf.uint8)
crop_window = tf.tile([[0, 0, 224, 224]], [10, 1])

#%% Set support set
_ = classifier.signatures["set_support_set"](image_name=catalog, crop_window=crop_window, label=label)

#%% Use Siamese as usual classifier
classifier.signatures["serving_default"](image_name=catalog, crop_window=crop_window)
