import imgaug.augmenters as iaa
import numpy as np
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input


def imagenet():
    return iaa.Lambda(lambda images_list, *_: preprocess_input(np.stack(images_list), data_format='channels_last'))
