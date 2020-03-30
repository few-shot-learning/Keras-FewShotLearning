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
image_bytes = [
    tf.io.encode_base64(tf.io.encode_jpeg(tf.cast(tf.random.uniform((350, 250, 3), minval=0, maxval=255), dtype=tf.uint8)))
    for _ in range(support_set_size)
]

label = np.random.choice(["label_A", "label_B", "label_C"], support_set_size).tolist()
crop_window = np.tile([0, 0, 200, 224], [support_set_size, 1]).tolist()

#%% Set support set
response = requests.post(
    "http://localhost:8501/v1/models/siamese_nets_classifier:predict",
    json={
        "signature_name": "set_support_set",
        "inputs": {
            "image_bytes": [image.numpy().decode("utf-8") for image in image_bytes],
            "crop_window": crop_window,
            "label": label,
            "overwrite": True,
        },
    },
)
json.loads(response.content)

#%% Use Siamese as usual classifier over random images
response = requests.post(
    "http://localhost:8501/v1/models/siamese_nets_classifier:predict",
    json={
        "inputs": {"image_bytes": [image.numpy().decode("utf-8") for image in image_bytes][:2], "crop_window": crop_window[:2]},
    },
)
pd.DataFrame(
    np.array(json.loads(response.content)["outputs"]["scores"]), columns=json.loads(response.content)["outputs"]["classes"]
)

#%% Update support set with new label
response = requests.post(
    "http://localhost:8501/v1/models/siamese_nets_classifier:predict",
    json={
        "signature_name": "set_support_set",
        "inputs": {
            "image_bytes": [image.numpy().decode("utf-8") for image in image_bytes][:1],
            "crop_window": crop_window[:1],
            "label": ["label_other"],
            "overwrite": False,
        },
    },
)

#%% Make new prediction, label is available for prediction
response = requests.post(
    "http://localhost:8501/v1/models/siamese_nets_classifier:predict",
    json={
        "inputs": {"image_bytes": [image.numpy().decode("utf-8") for image in image_bytes][:2], "crop_window": crop_window[:2]},
    },
)
pd.DataFrame(
    np.array(json.loads(response.content)["outputs"]["scores"]), columns=json.loads(response.content)["outputs"]["classes"]
)
