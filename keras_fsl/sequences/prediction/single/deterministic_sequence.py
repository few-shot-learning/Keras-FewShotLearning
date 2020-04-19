import pandas as pd

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class DeterministicSequence(AbstractSequence):
    """
    Iterate over the query dataframe deterministically
    """

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return [
            pd.np.stack(
                self.preprocessings[0].augment_images(self.load_img(self.annotations[0].iloc[start_index:end_index])), axis=0,
            )
        ]
