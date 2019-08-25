import pandas as pd

from datasets import omniglot
from src.models import SiameseNets
from src.sequences import DeterministicSequence, RandomPairsSequence

# %% Get data
train_set, test_set = omniglot.load_data()

# %% Update label columns to be able to mix alphabet during training
train_set = train_set.assign(label=lambda df: df.alphabet + '_' + df.label)
test_set = test_set.assign(label=lambda df: df.alphabet + '_' + df.label)

# %% Training
model = SiameseNets({'name': 'SingleConv2D', 'init': {'input_shape': (105, 105, 3)}})
train_sequence = RandomPairsSequence(train_set, batch_size=16)
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit_generator(train_sequence)

# %% Prediction
encoder = model.get_layer('branch_model')
head_model = model.get_layer('head_model')
test_sequence = DeterministicSequence(test_set, batch_size=16)
embeddings = encoder.predict_generator(test_sequence, verbose=1)

k_shot = 1
n_way = 5
query = (
    test_set
    .loc[lambda df: df.label.isin(support.label.unique())]
    .loc[lambda df: ~df.index.isin(support.index)]
    .loc[lambda df: df.index.repeat(k_shot * n_way)]
    .reset_index()
)
support = (
    pd.concat([(
        test_set
        .loc[lambda df: df.label.isin(test_set.label.drop_duplicates().sample(n_way))]
        .groupby('label')
        .apply(lambda group: group.sample(k_shot).drop('label', axis=1))
        .reset_index('label')
    )] * (len(query) // k_shot // n_way))
    .reset_index()
)
predictions = (
    pd.concat([
        query,
        pd.DataFrame(head_model.predict([embeddings[query['index']], embeddings[support['index']]]), columns=['score']),
        support.add_suffix('_support'),
    ], axis=1)
    .pivot_table(
        values='score',
        index=query.columns.to_list(),
        columns=support.add_suffix('_support').columns.to_list(),
    )
    .assign(label_predicted=lambda df: df.idxmin(axis=1))
    .reset_index()
)
