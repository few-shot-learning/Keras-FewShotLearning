import pandas as pd

from keras_fsl.sequences.abstract_sequence import AbstractSequence


class AbstractPairsSequence(AbstractSequence):
    """
    Base class for pairs sequences. All initialization should
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            annotations (Union[pandas.DataFrame, list[pandas.DataFrame]]): query and support annotations. If a single
            dataframe is given, it will be used both for query and support set.
            batch_size (int): number of images per batch
        """
        self.query_samples = pd.DataFrame()
        self.support_samples = pd.DataFrame()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        query_images, query_keypoints = self.query_preprocessing(
            images=self.load_img(self.query_samples.iloc[start_index:end_index]),
            keypoints=self.query_samples.iloc[start_index:end_index].center.tolist()[:, pd.np.newaxis, :],
        )
        support_images = self.support_preprocessing(
            images=self.load_img(self.support_samples.iloc[start_index:end_index])
        )
        targets = pd.np.stack([
            kp.to_keypoint_image().clip(max=1).squeeze() * self.targets[start_index:end_index][i].astype(int)
            for i, kp in enumerate(query_keypoints)
        ])
        return [query_images, support_images], targets

    @property
    def targets(self):
        return self.query_samples.label == self.support_samples.label
