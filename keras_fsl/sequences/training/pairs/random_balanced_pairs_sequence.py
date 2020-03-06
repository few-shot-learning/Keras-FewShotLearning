import pandas as pd

from .abstract_pairs_sequence import AbstractPairsSequence


class RandomBalancedPairsSequence(AbstractPairsSequence):
    """Generate random pairs with half matching and half non matching pairs in each batch"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.query_samples = self.query_annotations.sample(frac=1).reset_index(drop=True)
        self.support_samples = pd.concat(
            self.query_samples.apply(
                lambda row: self.support_annotations_by_label[self.select_support_label(row)].sample(1), axis=1
            ).to_list()
        ).reset_index(drop=True)

    def select_support_label(self, row):
        if row.name % 2:  # name is index in original dataframe
            return row.label
        labels = set(self.support_annotations_by_label.keys()) - {row.name}
        return pd.np.random.choice(list(labels))
