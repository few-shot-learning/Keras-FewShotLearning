import pandas as pd

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class DeterministicSequence(AbstractSequence):
    """
    Iterate over the query dataframe deterministically
    """

    def __init__(self, annotations, batch_size, classes=None, **load_img_kwargs):
        super().__init__(annotations, batch_size, **load_img_kwargs)
        self.targets = pd.get_dummies(self.annotations[0].label)
        if classes is not None:
            self.targets = self.targets.reindex(classes, axis=1, fill_value=0)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return (
            pd.np.stack(
                self.preprocessings[0].augment_images(self.load_img(self.annotations[0].iloc[start_index:end_index])),
                axis=0,
            ),
            self.targets.iloc[start_index:end_index],
        )
