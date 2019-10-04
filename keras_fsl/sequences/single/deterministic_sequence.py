from keras_fsl.sequences.abstract_sequence import AbstractSequence


class DeterministicSequence(AbstractSequence):
    """
    Iterate over the query dataframe deterministically
    """
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        return self.load_query_img(
            self.query_annotations
            .iloc[start_index:end_index]
        )
