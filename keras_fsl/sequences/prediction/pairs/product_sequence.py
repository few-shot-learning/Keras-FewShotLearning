import math

import pandas as pd
from tensorflow.keras.utils import Sequence


class ProductSequence(Sequence):
    def __init__(self, support_images_array, query_images_array, support_labels, query_labels=None, batch_size=16):
        """

        Args:
            query_images_array (np.array): embeddings of the query images
            support_images_array (np.array): embeddings of the support images
            support_labels (Union[list, np.array]): labels going with the support images embedding (same len)
            query_labels (Union[list, np.array]): labels going with the query images embedding (same len)
            batch_size (int): batch_size for the sequence
        """
        self.pairs_indexes = pd.DataFrame(
            {
                "query_index": pd.np.repeat(pd.np.arange(len(query_images_array)), len(support_labels)),
                "support_index": pd.np.tile(pd.np.arange(len(support_images_array)), reps=len(query_images_array)),
            }
        ).assign(
            query_label=lambda df: query_labels[df.query_index] if query_labels is not None else pd.np.NaN,
            support_label=lambda df: support_labels[df.support_index],
            target=lambda df: df.query_label == df.support_label,
        )
        self.query_images_array = query_images_array
        self.support_images_array = support_images_array
        self.batch_size = batch_size

    def __getitem__(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        query_index = self.pairs_indexes.query_index.iloc[start:end]
        support_index = self.pairs_indexes.support_index.iloc[start:end]
        return [self.query_images_array[query_index], self.support_images_array[support_index]]

    def __len__(self):
        return math.ceil(len(self.pairs_indexes) / self.batch_size)
