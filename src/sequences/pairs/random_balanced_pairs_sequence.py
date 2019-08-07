import pandas as pd

from .abstract_pairs_sequence import AbstractPairsSequence


class RandomBalancedPairsSequence(AbstractPairsSequence):
    """Generate random pairs with half matching and half non matching pairs in each batch"""

    def __init__(self, annotations, batch_size, **load_img_kwargs):
        super().__init__(annotations, batch_size, **load_img_kwargs)
        self.support_annotations_by_label = {
            group[0]: group[1]
            for group in self.support_annotations.groupby('label')
        }

    def on_epoch_end(self):
        self.query_samples = self.query_annotations.sample(frac=1)
        self.support_samples = self.query_samples.apply(lambda row: (
            self.support_annotations_by_label[self.select_support_label(row)].sample(1)
        ))

    def select_support_label(self, row):
        if row.index % 2:
            return row.label
        labels = set(self.support_annotations_by_label.keys()) - set(row.label)
        return pd.np.random.choice(list(labels))
