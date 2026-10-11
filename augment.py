#!/usr/bin/env python3
"""
augment.py
==========

Online data augmentation for the 2D slice pipeline.

The SegTHOR paper (Lambert et al., 2019, sec. 4.1) tripled its training set
with two transforms: a random affine transform, and a random deformation
driven by a 2x2x2 control point grid with B-spline interpolation. This
module does the same two things, with two deliberate differences:

  - Online rather than offline. The paper generated three fixed copies of
    the dataset in advance. Here a fresh random transform is drawn every
    time a slice is loaded, so the model never sees the same variation
    twice and no extra data is written to disk.

  - 2D rather than 3D. Our pipeline loads one axial slice at a time, so
    only in-plane transforms are possible. A true 3D deformation would
    have to be applied to the volumes before slicing.

Both transforms are applied to the image and its label with the same random
parameters, and the label is always resampled with nearest-neighbour
interpolation. Anything else averages neighbouring class indices and
invents organs at the boundaries: halfway between esophagus (1) and
trachea (3) is heart (2).

Randomness comes from numpy's global generator, which main.py seeds per
worker through worker_init_fn, so runs stay reproducible.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, zoom
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


@dataclass
class AugmentConfig:
    """Ranges are deliberately conservative: thoracic anatomy sits in a
    fairly consistent position and orientation, so large rotations or
    scalings would produce images unlike anything at test time."""

    p_affine: float = 0.5        # probability of applying the affine transform
    degrees: float = 10.0        # rotation, +/- this many degrees
    translate: float = 0.05      # fraction of image width/height
    scale: tuple = (0.9, 1.1)
    shear: float = 5.0           # +/- degrees

    p_elastic: float = 0.3       # probability of applying the deformation
    grid: int = 3                # control points per axis (the paper used 2)
    alpha: float = 12.0          # displacement magnitude in pixels

    p_intensity: float = 0.0     # off by default; see note below
    brightness: float = 0.1      # +/- fraction
    contrast: float = 0.1


def _random_affine(img: Image.Image, gt: Image.Image, cfg: AugmentConfig):
    """One affine transform, identical for image and label."""
    angle = float(np.random.uniform(-cfg.degrees, cfg.degrees))
    max_dx = cfg.translate * img.size[0]
    max_dy = cfg.translate * img.size[1]
    translate = (int(np.random.uniform(-max_dx, max_dx)),
                 int(np.random.uniform(-max_dy, max_dy)))
    scale = float(np.random.uniform(*cfg.scale))
    shear = [float(np.random.uniform(-cfg.shear, cfg.shear)), 0.0]

    img = TF.affine(img, angle=angle, translate=translate, scale=scale,
                    shear=shear, interpolation=InterpolationMode.BILINEAR, fill=0)
    gt = TF.affine(gt, angle=angle, translate=translate, scale=scale,
                   shear=shear, interpolation=InterpolationMode.NEAREST, fill=0)
    return img, gt


def _elastic_field(shape, grid: int, alpha: float):
    """A smooth displacement field from a coarse control grid.

    Random offsets are drawn at grid x grid control points and then
    upsampled with cubic interpolation, which is what makes the result
    smooth and globally coherent rather than per-pixel noise. Same idea as
    the paper's B-spline grid, in 2D.
    """
    h, w = shape
    coarse_y = np.random.uniform(-1, 1, (grid, grid))
    coarse_x = np.random.uniform(-1, 1, (grid, grid))

    dy = zoom(coarse_y, (h / grid, w / grid), order=3)[:h, :w] * alpha
    dx = zoom(coarse_x, (h / grid, w / grid), order=3)[:h, :w] * alpha
    return dy, dx


def _apply_elastic(img: Image.Image, gt: Image.Image, cfg: AugmentConfig):
    """One deformation, identical for image and label."""
    img_a = np.asarray(img)
    gt_a = np.asarray(gt)
    h, w = img_a.shape[:2]

    dy, dx = _elastic_field((h, w), cfg.grid, cfg.alpha)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([yy + dy, xx + dx])

    # order=1 for the image (smooth), order=0 for the label (never average
    # class indices).
    if img_a.ndim == 3:  # multi-window slice: the same deformation for every channel
        img_out = np.stack([map_coordinates(img_a[..., c], coords, order=1, mode="constant", cval=0)
                            for c in range(img_a.shape[2])], axis=-1)
    else:
        img_out = map_coordinates(img_a, coords, order=1, mode="constant", cval=0)
    gt_out = map_coordinates(gt_a, coords, order=0, mode="constant", cval=0)

    return (Image.fromarray(img_out.astype(img_a.dtype)),
            Image.fromarray(gt_out.astype(gt_a.dtype)))


def _intensity_jitter(img: Image.Image, cfg: AugmentConfig):
    """Brightness and contrast jitter. The label is untouched: intensity
    changes do not move anatomy.

    Motivated by our own measurements rather than borrowed: the scans come
    from at least two acquisition protocols (slice thickness is bimodal at
    2.0 and 2.5 mm) and intensity maxima range from 1892 to 31743 HU
    because of metal artifacts, so the model should not rely on absolute
    brightness.
    """
    b = 1.0 + float(np.random.uniform(-cfg.brightness, cfg.brightness))
    c = 1.0 + float(np.random.uniform(-cfg.contrast, cfg.contrast))
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    return img


class SliceAugmentor:
    """Applies the augmentations to one (image, label) PIL pair.

    Used in SliceDataset.__getitem__ before the tensor transforms, so the
    label is still a plain PNG of class indices and has not yet been
    one-hot encoded.
    """

    def __init__(self, cfg: AugmentConfig | None = None):
        self.cfg = cfg or AugmentConfig()

    def __call__(self, img: Image.Image, gt: Image.Image):
        cfg = self.cfg

        if np.random.rand() < cfg.p_affine:
            img, gt = _random_affine(img, gt, cfg)

        if np.random.rand() < cfg.p_elastic:
            img, gt = _apply_elastic(img, gt, cfg)

        if cfg.p_intensity and np.random.rand() < cfg.p_intensity:
            img = _intensity_jitter(img, cfg)

        return img, gt