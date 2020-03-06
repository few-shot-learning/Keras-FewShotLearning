import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tensorflow.keras.callbacks import Callback


class BinaryStatistics(Callback):
    def __init__(self, validation_data, log_dir):
        super().__init__()
        X, y = validation_data
        self.X = tf.convert_to_tensor(X, tf.float32)
        self.y = tf.convert_to_tensor(y, tf.bool)
        self.writer = tf.summary.create_file_writer(str(log_dir))

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model(self.X)
        matching_scores = tf.boolean_mask(y_pred, self.y).numpy().flatten()
        non_matching_scores = tf.boolean_mask(y_pred, tf.math.logical_not(self.y)).numpy().flatten()
        plt.subplot(1, 2, 1)
        plt.boxplot(matching_scores)
        plt.boxplot(non_matching_scores)
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        image = np.fromstring(s, np.uint8).reshape((1, height, width, 4))
        with self.writer.as_default():
            tf.summary.image(self.__class__.__name__, image, step=epoch)
