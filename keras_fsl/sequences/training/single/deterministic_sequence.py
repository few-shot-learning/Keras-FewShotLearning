import pandas as pd

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class DeterministicSequence(AbstractSequence):
    """
    Iterate over the query dataframe deterministically
    """

    def __init__(
        self,
        annotations,
        batch_size,
        shuffle=False,
        classes=None,
        labels_in_input=False,
        labels_in_output=True,
        to_categorical=True,
        **kwargs,
    ):
        """
        Args:
            annotations (List[pd.DataFrame]): list of annotations
            batch_size (int): batch_size of the sequence.
            shuffle (bool): shuffle the annotations on epoch end
            classes (list): list of classes to insure codes are the sames with some references
            labels_in_input (bool): add the target labels in the input
            labels_in_output (bool): add the target labels in the output
            to_categorical:
            **kwargs:
        """
        super().__init__(annotations, batch_size, **kwargs)
        self.labels_in_input = labels_in_input
        self.labels_in_output = labels_in_output
        self.annotations[0].label = pd.Categorical(self.annotations[0].label, categories=classes)
        self.targets = self.annotations[0].label.cat.codes
        self.shuffle = shuffle
        if to_categorical:
            self.targets = pd.get_dummies(self.targets).reindex(list(range(len(self.classes))), axis=1).fillna(0)
        self.on_epoch_end()

    @property
    def classes(self):
        return self.annotations[0].label.cat.categories

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        inputs = [
            pd.np.stack(
                self.preprocessings[0].augment_images(self.load_img(self.annotations[0].iloc[start_index:end_index])), axis=0,
            )
        ]
        output = [self.targets.iloc[start_index:end_index].values.astype("float32")]
        if self.labels_in_input:
            inputs += output
        if not self.labels_in_output:
            output.pop()
        return inputs, output

    def on_epoch_end(self):
        if self.shuffle:
            indexes = pd.np.random.permutation(self.annotations[0].index)
            self.annotations[0] = self.annotations[0].loc[indexes]
            self.targets = self.targets.loc[indexes]
