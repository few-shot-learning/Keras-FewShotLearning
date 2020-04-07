import io
from typing import Mapping, Callable

from matplotlib import pyplot as plt
import tensorflow as tf


class TensorBoardSaveFigures(tf.keras.callbacks.Callback):
    def __init__(self, logdir, figure_producers: Mapping[str, Callable[[tf.keras.Model, int], plt.Figure]]):
        super().__init__()
        self._file_writer = tf.summary.create_file_writer(logdir)
        self._figure_producers = figure_producers

    def on_epoch_end(self, epoch, logs={}):
        with self._file_writer.as_default():
            for figure_name, make_figure in self._figure_producers.items():
                tf.summary.image(figure_name, self.plot_to_image(make_figure(self.model, epoch)), step=epoch)
        return logs

    @staticmethod
    def plot_to_image(figure: plt.Figure) -> tf.Tensor:
        """Converts the matplotlib plot specified by 'figure' to a uint8 tensor image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(figure)
        buffer.seek(0)
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        return tf.expand_dims(image, 0)
