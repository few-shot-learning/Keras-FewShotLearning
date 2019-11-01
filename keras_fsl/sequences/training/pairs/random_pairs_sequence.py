from .abstract_pairs_sequence import AbstractPairsSequence


class RandomPairsSequence(AbstractPairsSequence):
    """Generate random pairs from query and support annotations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.query_samples = self.query_annotations.sample(frac=1).reset_index(drop=True)
        self.support_samples = self.support_annotations.sample(frac=1).reset_index(drop=True)
