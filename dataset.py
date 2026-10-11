#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from augment import SliceAugmentor


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 context_slices=1, position=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentor = None
        if augment:
            cfg = augment if not isinstance(augment, bool) else None
            self.augmentor = SliceAugmentor(cfg)
        self.equalize: bool = equalize
        self.context_slices: int = context_slices

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)

        # Position channels: a 2D network cannot tell at which height of the
        # body a slice lies, and our error analysis showed the esophagus fails
        # mostly in its lower half. x and y give the pixel position in the
        # image, z the slice's relative height within its own scan.
        self.position: bool = position
        # highest slice index per patient, not the file count, so the height
        # stays right when a variant leaves out slices (e.g. --skip_empty)
        self.last_slice: dict[str, int] = {}
        for img_path, _ in self.files:
            patient, idx = img_path.stem.rsplit('_', 1)
            self.last_slice[patient] = max(self.last_slice.get(patient, 0), int(idx))

        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def _get_adjacent_slices(self, img_path: Path, offset: int) -> Path:
        patient_id, slice_str = img_path.stem.rsplit('_', 1)
        idx = int(slice_str)
        step = 1 if offset > 0 else -1
        # at the top/bottom of the volume, use the closest slice that exists
        for o in range(offset, 0, -step):
            p = img_path.parent / f"{patient_id}_{idx + o:04d}.png"
            if idx + o >= 0 and p.exists():
                return p
        return img_path

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        r = self.context_slices // 2
        slice_paths = [self._get_adjacent_slices(img_path, o) if o != 0 else img_path
                       for o in range(-r, r + 1)]
        slices_pil = [Image.open(p) for p in slice_paths]
        gt_pil = Image.open(gt_path) if not self.test_mode else None

        # Augment before the tensor transforms, while the label is still a
        # plain image of class indices and has not been one-hot encoded.
        if self.augmentor is not None and not self.test_mode:
            # each slice of the stack needs to get the same random transform
            # otherwise the neighbours no longer line up with the centre slice.
            state = np.random.get_state()
            augmented = []
            for k, s in enumerate(slices_pil):
                np.random.set_state(state)
                s_aug, g_aug = self.augmentor(s, gt_pil)
                augmented.append(s_aug)
                if k == r:  
                    gt_out = g_aug
            slices_pil, gt_pil = augmented, gt_out

        img: Tensor = torch.cat([self.img_transform(s) for s in slices_pil], dim=0)

        if self.position:
            # added after augmentation: they describe where a pixel is in the
            # image and in the scan, which the augmentation does not move
            patient, idx = img_path.stem.rsplit('_', 1)
            _, W, H = img.shape
            x = torch.linspace(0, 1, W).view(W, 1).expand(W, H)
            y = torch.linspace(0, 1, H).view(1, H).expand(W, H)
            z = torch.full((W, H), int(idx) / max(self.last_slice[patient], 1))
            img = torch.cat([img, torch.stack([x, y, z])], dim=0)

        data_dict = {"images": img,
                     "stems": img_path.stem}
        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt_pil)

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict