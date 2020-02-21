import tensorflow as tf

from keras_fsl.dataframe.operators.abstract_operator import AbstractOperator


class ToKShotDataset(AbstractOperator):
    """
    Create tf.data.Dataset with random groups of k_shot consecutive images with the same label
    """

    def __init__(self, k_shot):
        self.k_shot = k_shot

    @staticmethod
    def load_img(annotation):
        """
        Args:
            annotation (dict): with keys 'image_name': path to the image and 'crop_window' to be passed to tf.io.decode_and_crop_jpeg
        Returns:
            dict: the input dict with an extra image key.
        """
        return (
            {
                'image': tf.io.decode_and_crop_jpeg(
                    tf.io.read_file(annotation['image_name']),
                    crop_window=annotation['crop_window'],
                    channels=3,
                ),
                **annotation,
            }
        )

    def repeat_k_shot(self, index):
        return tf.data.Dataset.from_tensors(index).repeat(self.k_shot)

    def to_dataset(self, group):
        """
        Transform a pd.DataFrame into a tf.data.Dataset and load images
        """
        return (
            tf.data.Dataset.from_tensor_slices(group.to_dict('list'))
            .map(self.load_img)
            .cache()
            .shuffle(buffer_size=len(group), reshuffle_each_iteration=True)
            .repeat()
        )

    def __call__(self, input_dataframe):
        return tf.data.experimental.choose_from_datasets(
            datasets=(
                input_dataframe
                .groupby('label')
                .apply(self.to_dataset)
            ),
            choice_dataset=(
                tf.data.Dataset.range(len(input_dataframe.label.unique()))
                .shuffle(buffer_size=len(input_dataframe.label.unique()), reshuffle_each_iteration=True)
                .flat_map(self.repeat_k_shot)
            ),
        )
