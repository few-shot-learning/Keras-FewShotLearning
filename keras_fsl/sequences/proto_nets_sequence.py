import numpy as np

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class ProtoNetsSequence(AbstractSequence):

    def __init__(self, *args, k_shot=5, n_way=5, **kwargs):
        self.k_shot = k_shot
        self.n_way = n_way
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        selected_labels = np.random.choice(self.support_labels.index.to_list(), size=self.n_way, replace=True)
        query = (
            self.query_annotations
            .loc[lambda df: df.label.isin(selected_labels)]
            .sample(self.batch_size, replace=True)
            .reset_index(drop=True)
        )
        support = [
            self.support_annotations_by_label[label].sample(self.batch_size, replace=True).reset_index(drop=True)
            for label in selected_labels
            for _ in range(self.k_shot)
        ]
        targets = np.stack(query.label.map(lambda label: label == selected_labels)).astype(int)
        return [
            self.load_img(dataframe)
            for dataframe in [query, *support]
        ], targets
