from keras_fsl.dataframe.operators.center_coordinates_to_keypoint import CenterCoordinatesToKeypoint
from keras_fsl.dataframe.operators.coordinates_to_bounding_box import CoordinatesToBoundingBox
from keras_fsl.dataframe.operators.corner_to_center_coordinates import CornerToCenterCoordinates


def CoordinatesToImgAug(input_dataframe):
    return (
        input_dataframe.pipe(CoordinatesToBoundingBox())
        .pipe(CornerToCenterCoordinates())
        .pipe(CenterCoordinatesToKeypoint())
        .drop(["x1", "y1", "x2", "y2", "x", "y", "width", "height"], axis=1)
    )
