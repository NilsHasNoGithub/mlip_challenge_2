from dataclasses import dataclass
import albumentations as A
import cv2

DEFAULT = [
    A.LongestMaxSize(512),
    A.PadIfNeeded(512, 512),
    A.RandomCrop(456, 456),
    A.HorizontalFlip(),
    A.RandomRotate90(),
    A.Transpose(),
    A.OneOf(
        [
            A.GaussNoise(),
        ],
        p=0.2,
    ),
    A.OneOf(
        [
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ],
        p=0.2,
    ),
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.2,
        rotate_limit=45,
        p=0.2,
    ),
    A.OneOf(
        [
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ],
        p=0.2,
    ),
    A.OneOf(
        [
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ],
        p=0.3,
    ),
    A.HueSaturationValue(p=0.3),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    # A.Blur(blur_limit=3, p=0.5),
    # A.GaussNoise(var_limit=(10, 50), p=0.5),
    # A.Normalize(
    #     mean=(0.485, 0.456, 0.406),
    #     std=(0.229, 0.224, 0.225),
    #     max_pixel_value=255.0,
    #     p=1.0,
    # ),
]


PRESETS = {
    "default": A.Compose(DEFAULT),
    "no_aug": A.Compose(DEFAULT[:2]),
    "flip_rot": A.Compose(DEFAULT[:5]),
    "happy_whale_4th": A.Compose(
        DEFAULT[:5]
        + [
            A.RandomContrast(p=0.75),
            A.ShiftScaleRotate(
                shift_limit=0.0, scale_limit=0.3, rotate_limit=10, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
        ]
    ),
}

VAL_PRESETS = {"default": A.Compose(DEFAULT[:2])}
