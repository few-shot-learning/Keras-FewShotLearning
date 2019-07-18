import pandas as pd

from keras.preprocessing.image import load_img, img_to_array

from src.sequences.sequence_abc import AbstractSequence


class DeterministicLeftSequence(AbstractSequence):
    """
    Iterate over all the left annotations and produce for each left image a batch with half matching and half
    non-matching pairs.

    Pairs are produced either with another (so called) right dataframe or from within the same dataframe if only one
    dataframe is provided.
    """

    def __init__(self, annotations, batch_size, model=None, **load_img_kwargs):
        if batch_size % 2 == 1:
            raise ValueError(f'batch_size should be even')
        super().__init__(annotations, batch_size, model)
        self.left_annotations = self.annotations[0]
        self.left_samples = pd.DataFrame()
        self.right_annotations = self.annotations[-1]
        self.right_samples = pd.DataFrame()
        self.right_labels = self.right_annotations.label.value_counts()
        self.targets = pd.Series()
        self.load_img_kwargs = load_img_kwargs
        self.on_epoch_end()

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return [
            pd.np.stack(
                self.left_samples
                .iloc[start_index:end_index]
                .apply(lambda row: img_to_array(load_img(row.image_name, **self.load_img_kwargs)))
            ),
            pd.np.stack(
                self.right_samples
                .iloc[start_index:end_index]
                .apply(lambda row: img_to_array(load_img(row.image_name, **self.load_img_kwargs)))
            ),
            ], self.targets[start_index:end_index]

    def __len__(self):
        return len(self.left_annotations)

    def on_epoch_end(self):
        self.left_annotations.sample(frac=1)
        self.left_samples = (
            self.left_annotations
            .iloc[pd.np.repeat(len(self.left_annotations), self.batch_size)]
            .reset_index(drop=True)
        )
        self.right_samples = (
            pd.concat(
                self.left_annotations
                .apply(lambda row: (self.get_batch_for_sample(row.label)), axis=1)
                .values,
            )
            .reset_index(drop=True)
        )
        self.targets = self.left_samples.label == self.right_samples.label

    def get_batch_for_sample(self, anchor_label):
        replace = self.right_labels[anchor_label] < self.batch_size
        positive_samples = (
            self.right_annotations
            .loc[lambda df: df.label == anchor_label]
            .sample(self.batch_size // 2, replace=replace)
        )
        negative_samples = (
            self.right_annotations
            .loc[lambda df: df.label != anchor_label]
            .sample(self.batch_size // 2)
        )
        return pd.concat([positive_samples, negative_samples])
