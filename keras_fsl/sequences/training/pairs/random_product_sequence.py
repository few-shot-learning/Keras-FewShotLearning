import math

import pandas as pd
from tensorflow.keras.utils import Sequence


class RandomProductSequence(Sequence):
    def __init__(self, images_array, labels, batch_size):
        """

        Args:
            images_array (np.array): with len matching len of labels
            labels (Union[list, np.array]): with len matching len of images_array
            batch_size (int): batch_size for the sequence
        """
        self.pairs_indexes = (
            pd.DataFrame(
                {
                    "query_index": pd.np.repeat(pd.np.arange(len(labels)), len(labels)),
                    "support_index": pd.np.tile(pd.np.arange(len(labels)), reps=len(labels)),
                }
            )
            .assign(
                query_label=lambda df: labels[df.query_index],
                support_label=lambda df: labels[df.support_index],
                target=lambda df: df.query_label == df.support_label,
            )
            .sample(frac=1)
        )
        self.images_array = images_array
        self.batch_size = batch_size

    def __getitem__(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        query_index = self.pairs_indexes.query_index.iloc[start:end]
        support_index = self.pairs_indexes.support_index.iloc[start:end]
        return (
            [self.images_array[query_index], self.images_array[support_index]],
            self.pairs_indexes.target.iloc[start:end],
        )

    def __len__(self):
        return math.ceil(len(self.pairs_indexes) / self.batch_size)

    def on_epoch_end(self):
        self.pairs_indexes = self.pairs_indexes.sample(frac=1)
