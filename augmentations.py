import random
import collections
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import transforms as A_torch
from albumentations.augmentations.functional import _maybe_process_in_chunks

def build_augmentations():
    """augmentations utilizadas pelo campeão do XView3. No caso do ScaleRotate, foi utilizada reflexão ao invés do padding com NaN""" 
    transforms = A.Compose(
		[
            # UnclippedRandomBrightnessContrast(brightness_limit=(-1,1), contrast_limit=0.1, image_in_log_space=False, p=0.25),
		    # UnclippedGaussNoise(image_in_log_space=False, var_limit=(0.0001, 0.005), mean=0, per_channel=True, p=0.5),
		    A.HorizontalFlip(p=0.2),
			A.VerticalFlip(p=0.2),
			# ElasticTransform(alpha=(10,100), p=0.1),
			# A.ShiftScaleRotate(scale_limit=0, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
			# RandomGridShuffle(p=0.2),
			# A.MedianBlur(blur_limit=5, p=0.05),
			# A.GaussianBlur(blur_limit=(3,5),p=0.05),
			A_torch.ToTensorV2()
		]
    )
    return transforms

@A.preserve_shape
def elastic_transform(
    img,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


class ElasticTransform(A.DualTransform):
    def __init__(
        self,
        alpha=1,
        sigma=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = A.to_tuple(alpha)
        self.sigma = A.to_tuple(sigma)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def get_transform_init_args_names(self):
        return ("alpha", "sigma", "interpolation", "border_mode", "value", "mask_value")

    def apply(self, img, sigma=0, alpha=0, map_x=None, map_y=None, interpolation=cv2.INTER_LINEAR, **params):
        return elastic_transform(
            img,
            map_x=map_x,
            map_y=map_y,
            interpolation=interpolation,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(self, img, sigma=0, alpha=0, map_x=None, map_y=None, **params):
        return elastic_transform(
            img,
            map_x=map_x,
            map_y=map_y,
            interpolation=cv2.INTER_NEAREST,
            border_mode=self.border_mode,
            value=self.value,
        )

    def update_params(self, params, **kwargs):
        height, width = kwargs["image"].shape[:2]

        dx = np.zeros((height, width))
        dy = np.zeros((height, width))

        for _ in range(128):
            dx[random.randrange(0, height), random.randrange(0, width)] = random.uniform(self.alpha[0], self.alpha[1])
            dy[random.randrange(0, height), random.randrange(0, width)] = random.uniform(self.alpha[0], self.alpha[1])

        for _ in range(32):
            dx = cv2.blur(dx, (7, 7))
            dy = cv2.blur(dy, (7, 7))

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        params["map_x"] = np.float32(x + dx)
        params["map_y"] = np.float32(y + dy)
        return params

    def apply_to_keypoint(self, keypoint, **params):
        x, y = keypoint[:2]
        map_x, map_y = params["map_x"], params["map_y"]
        mask = np.zeros(map_x.shape[:2], dtype=np.uint8)
        mask[y, x] = 255
        mask = cv2.remap(mask, map_x, map_y, borderMode=cv2.BORDER_CONSTANT, borderValue=0, interpolation=cv2.INTER_LINEAR)
        _, _, _, maxLoc = cv2.minMaxLoc(mask)
        xn, yn = maxLoc
        return (xn, yn) + keypoint[2:]

class RandomGridShuffle(A.RandomGridShuffle):
    """
    RandomGridShuffle with keypoints support
    """
    def apply(self, img, tiles=None, **params):
        if tiles is None:
            tiles = []

        return A.swap_tiles_on_image(img, tiles)

    def apply_to_keypoint(self, keypoint, tiles=None, rows=0, cols=0, **params):
        if tiles is None:
            return keypoint

        # for curr_x, curr_y, old_x, old_y, shift_x, shift_y in tiles:
        for (
            current_left_up_corner_row,
            current_left_up_corner_col,
            old_left_up_corner_row,
            old_left_up_corner_col,
            height_tile,
            width_tile,
        ) in tiles:
            x, y = keypoint[:2]

            if (old_left_up_corner_row <= y < (old_left_up_corner_row + height_tile)) and (
                old_left_up_corner_col <= x < (old_left_up_corner_col + width_tile)
            ):
                x = x - old_left_up_corner_col + current_left_up_corner_col
                y = y - old_left_up_corner_row + current_left_up_corner_row
                keypoint = (x, y) + tuple(keypoint[2:])
                break

        return keypoint

def unclipped_gauss_noise(image, gauss):
    return image.astype(np.float32, copy=False) + gauss.astype(np.float32, copy=False)


class UnclippedGaussNoise(A.ImageOnlyTransform):
    def __init__(self, var_limit=(0.01, 0.1), mean=0, per_channel=True, always_apply=False, p=0.5, image_in_log_space=True):
        super().__init__(always_apply, p)
        if isinstance(var_limit, collections.Iterable) and len(var_limit) == 2:
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = tuple(var_limit)
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError("Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit)))

        self.mean = A.to_tuple(mean)
        self.per_channel = per_channel
        self.image_in_log_space = image_in_log_space

    def apply(self, img, gauss=None, **params):
        if self.image_in_log_space:
            img = np.power(10, img)
        img = unclipped_gauss_noise(img, gauss=gauss)
        if self.image_in_log_space:
            img = np.log10(img)
        return img

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        mean = random.uniform(self.mean[0], self.mean[1])

        if self.per_channel:
            gauss = random_state.normal(mean, sigma, image.shape)
        else:
            gauss = random_state.normal(mean, sigma, image.shape[:2])
            if len(image.shape) == 3:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss.astype(np.float32)}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit", "per_channel", "mean")

def brightness_contrast_adjust_fixed(img, alpha=1.0, beta=0.0):
    if not np.isfinite(img).any():
        return img

    img = img * alpha + beta
    return img


class UnclippedRandomBrightnessContrast(A.ImageOnlyTransform):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5, image_in_log_space=True, per_channel=False):
        super().__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)
        self.contrast_limit = A.to_tuple(contrast_limit)
        self.image_in_log_space = image_in_log_space
        self.per_channel = per_channel

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        if self.image_in_log_space:
            img = np.power(10, img)
        img = brightness_contrast_adjust_fixed(img, alpha, beta)
        if self.image_in_log_space:
            img = np.log10(img)
        return img

    def get_params_dependent_on_targets(self, params):
        image = params["image"]

        if self.per_channel:
            num_channels = image.shape[2]
            alphas = [1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]) for _ in range(num_channels)]
            betas = [0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]) for _ in range(num_channels)]

            return {
                "alpha": np.array(alphas, dtype=np.float32),
                "beta": np.array(betas, dtype=np.float32),
            }
        else:
            return {
                "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
                "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
            }

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "brightness_limit", "contrast_limit", "image_in_log_space", "per_channel"