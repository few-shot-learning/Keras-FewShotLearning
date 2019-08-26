import pandas as pd
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class DeterministicSequence(AbstractSequence):
    """
    Iterate over the query dataframe deterministically
    """
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return pd.np.stack(
            self.query_annotations
            .iloc[start_index:end_index]
            .apply(lambda row: img_to_array(load_img(row.image_name, **self.load_img_kwargs)), axis=1)
        )
