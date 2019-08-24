from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import array_to_img

from src.models import SiameseNets
from src.sequences import BalancedPairsSequence, RandomBalancedPairsSequence, RandomPairsSequence, ProtoNetsSequence

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

# %% Create light branch model (encoder)
branch_model = Sequential()
branch_model.add(Conv2D(10, (3, 3), input_shape=(224, 224, 3)))
branch_model.add(GlobalAveragePooling2D())

# %% ## Train Siamese net
model = SiameseNets(branch_model)
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
model = SiameseNets(branch_model, head_model={
    'name': 'ProtoNets',
    'init': proto_net_parameters
})
model.get_layer('branch_model').summary()
model.get_layer('head_model').summary()
model.summary()

train_sequence = ProtoNetsSequence(train_set, batch_size=16, **proto_net_parameters)
model.compile('sgd', 'categorical_crossentropy')
model.fit(train_sequence)
