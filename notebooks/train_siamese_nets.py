from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.image import array_to_img

from keras_fsl.models import SiameseNets
from keras_fsl.sequences import BalancedPairsSequence, RandomBalancedPairsSequence, RandomPairsSequence, ProtoNetsSequence

# %% Generate fake data
image_dir = Path('data') / 'images'
image_dir.mkdir(parents=True, exist_ok=True)
train_set = pd.DataFrame(columns=['image_name', 'label'])
for i in range(10):
    array_to_img((np.random.rand(224, 224, 3) * 255).astype(int)).save(image_dir / f'image_{i}.jpg')
    train_set = train_set.append(
        {'image_name': image_dir / f'image_{i}.jpg', 'label': np.random.randint(9)},
        ignore_index=True,
    )

# %% ## Train Siamese net
model = SiameseNets({'name': 'SingleConv2D', 'init': {'input_shape': (224, 224, 3)}})
model.get_layer('branch_model').summary()
model.get_layer('head_model').summary()
model.summary()

# %% Train with random pairs
train_sequence = RandomPairsSequence(train_set, batch_size=16)
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit_generator(train_sequence)

# %% Train with balanced random pairs
train_sequence = RandomBalancedPairsSequence(train_set, batch_size=16)
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit_generator(train_sequence)

# %% Train iterating over the support set to generate batch of episode for the query set
train_sequence = BalancedPairsSequence(train_set, batch_size=16)
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit_generator(train_sequence)

# %% ## Train ProtoNets
proto_net_parameters = {'k_shot': 1, 'n_way': 3}
model = SiameseNets(
    branch_model={
        'name': 'SingleConv2D',
        'init': {'input_shape': (224, 224, 3)}
    },
    head_model={
        'name': 'ProtoNets',
        'init': proto_net_parameters
})
model.get_layer('branch_model').summary()
model.get_layer('head_model').summary()
model.summary()

train_sequence = ProtoNetsSequence(train_set, batch_size=16, **proto_net_parameters)
model.compile('sgd', 'categorical_crossentropy')
model.fit(train_sequence)
