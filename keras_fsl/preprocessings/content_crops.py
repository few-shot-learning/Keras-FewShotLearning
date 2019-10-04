import imgaug.augmenters as iaa

from .augmenters import imagenet


def content_crops():
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-180, 180)),
        iaa.CropToFixedSize(224, 224, position='center'),
        iaa.PadToFixedSize(224, 224, position='center'),
        iaa.AssertShape((None, 224, 224, 3)),
        imagenet,
    ])
