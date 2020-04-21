import imgaug.augmenters as iaa
from imgaug import parameters as iap


def YOLO():
    """
    Data augmentation model for YOLOv3 training
    """
    return iaa.Sequential(
        [
            iaa.KeepSizeByResize(iaa.Affine(scale=iap.Normal(1, 0.125), translate_percent=0.1, cval=128)),
            iaa.Fliplr(0.5),
            iaa.Resize({"height": iap.Normal(1, 0.1), "width": iap.Normal(1, 0.1)}),
            iaa.Resize({"longer-side": 416, "shorter-side": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(416, 416, pad_cval=128),
            iaa.MultiplyHueAndSaturation(mul_hue=iap.Uniform(0, 2), mul_saturation=iap.Uniform(1 / 1.5, 1.5)),
            iaa.AssertShape((None, 416, 416, 3)),
        ]
    )
