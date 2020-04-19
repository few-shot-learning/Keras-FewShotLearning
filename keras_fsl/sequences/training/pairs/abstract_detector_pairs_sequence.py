import pandas as pd

from keras_fsl.sequences.abstract_sequence import AbstractSequence
import imgaug.augmenters as iaa


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
        super().__init__(*args, **kwargs)
        self.query_preprocessing = self.preprocessings[0]
        self.support_preprocessing = self.preprocessings[-1]
        self.query_annotations = self.annotations[0]
        self.support_annotations = self.annotations[-1]
        self.support_annotations_by_label = {group[0]: group[1] for group in self.support_annotations.groupby("label")}

        self.query_samples = pd.DataFrame()
        self.support_samples = pd.DataFrame()
        self.target_augmenter = iaa.MaxPooling((32, 32), keep_size=False, deterministic=True)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        query_images, query_bounding_boxes = self.query_preprocessing(
            images=self.load_img(self.query_samples.iloc[start_index:end_index]),
            bounding_boxes=self.query_samples.iloc[start_index:end_index].bounding_boxes,
        )
        support_images = self.support_preprocessing(images=self.load_img(self.support_samples.iloc[start_index:end_index]))
        targets = pd.np.stack(
            [
                self.target_augmenter.augment_keypoints(kp).to_keypoint_image().clip(max=1).squeeze()
                * self.targets[start_index:end_index].iloc[i].astype(int)
                for i, kp in enumerate(query_bounding_boxes)
            ]
        )
        return [query_images, support_images], targets

    @property
    def targets(self):
        return self.query_samples.label == self.support_samples.label

    @property
    def support_labels(self):
        if self._support_labels is None:
            self._support_labels = self.support_annotations.label.value_counts()
        return self._support_labels
