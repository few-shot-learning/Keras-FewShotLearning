from .basic_cnn import BasicCNN
from .darknet import Darknet7, Darknet53
from .koch_net import KochNet
from .single_conv_2d import SingleConv2D
from .vinyals_net import VinyalsNet


__all__ = ["BasicCNN", "Darknet7", "Darknet53", "KochNet", "SingleConv2D", "VinyalsNet"]
