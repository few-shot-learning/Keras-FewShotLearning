from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Lambda, Flatten


def CenterSlicing2D():
    """
    Select the center of a 2D grid across all the channels
    """
    return Sequential([
        Lambda(lambda grid: grid[:, grid.shape[1] // 2, grid.shape[2] // 2, :]),
        Flatten(),
    ], name='slicing_2D')
