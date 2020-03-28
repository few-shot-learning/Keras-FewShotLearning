import json
import requests

import numpy as np
import pandas as pd
import tensorflow as tf

#%% Build and run docker images
# docker run -p 8501:8501 --mount type=bind,source=/aboslute/path/to/siamese_nets_classifier,target=/models/siamese_nets_classifier -e MODEL_NAME=siamese_nets_classifier -t tensorflow/serving

#%% Check API status
requests.get("http://localhost:8501/v1/models/siamese_nets_classifier")

#%% Create fake catalog
support_set_size = 10
image_bytes = tf.convert_to_tensor(
    [
        tf.io.encode_jpeg(tf.cast(tf.random.uniform((350, 250, 3), minval=0, maxval=255), dtype=tf.uint8))
        for _ in range(support_set_size)
    ]
)

label = tf.one_hot(np.random.choice(range(3), support_set_size), depth=3, dtype=tf.uint8)
crop_window = tf.tile([[0, 0, 200, 224]], [support_set_size, 1])

#%% Set support set
response = requests.post(
    "http://localhost:8501/v1/models/siamese_nets_classifier:predict",
    json={
        "signature_name": "set_support_set",
        "inputs": {
            "image_bytes": [image.decode("utf-8") for image in image_bytes.numpy().tolist()],
            "crop_window": crop_window.numpy().tolist(),
            "label": label.numpy().tolist(),
            "overwrite": True,
        },
    },
)
json.loads(response.content)

#%% Use Siamese as usual classifier over random images
response = requests.post(
    "http://localhost:8501/v1/models/siamese_nets_classifier:predict",
    json={
        "inputs": {
            "image_bytes": [image.decode("utf-8") for image in image_bytes.numpy().tolist()][:2],
            "crop_window": crop_window.numpy().tolist()[:2],
        },
    },
)
pd.DataFrame(np.array(json.loads(response.content)["outputs"]))
