import os
import numpy as np
import torch
import rasterio
import albumentations as A

from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size=128,
        augment=False
    ):

        self.root_dir = root_dir

        self.pre_dir = os.path.join(root_dir, 'pre')
        self.post_dir = os.path.join(root_dir, 'post')
        self.mask_dir = os.path.join(root_dir, 'mask')

        # self.files = sorted(os.listdir(self.pre_dir))
        self.files = []

        all_files = sorted(os.listdir(self.pre_dir))

        for file_name in all_files:
        
            mask_path = os.path.join(
                self.mask_dir,
                file_name
            )

            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            mask = self.remap_mask(mask)

            # keep only masks with change
            if np.sum(mask) > 0:
                self.files.append(file_name)


        self.transform = self.get_transforms(
            image_size,
            augment
        )

    def get_transforms(self, image_size, augment):

        transforms = [
            A.Resize(image_size, image_size)
        ]

        if augment:

            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])

        return A.Compose(transforms)

    def remap_mask(self, mask):

        mask = mask.copy()

        # background
        mask[mask == 1] = 0

        # changed
        mask[mask == 2] = 1
        mask[mask == 3] = 1

        return mask

    def read_tiff(self, path):

        with rasterio.open(path) as src:
            image = src.read()

        # Convert from (C,H,W) -> (H,W,C)
        image = np.transpose(image, (1, 2, 0))

        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        pre_path = os.path.join(
            self.pre_dir,
            file_name
        )

        post_path = os.path.join(
            self.post_dir,
            file_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            file_name
        )

        # =========================
        # LOAD IMAGES
        # =========================

        pre = self.read_tiff(pre_path)

        post = self.read_tiff(post_path)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # print("Mask unique:", np.unique(mask))
        # =========================
        # REMAP MASK
        # =========================

        mask = self.remap_mask(mask)

        # =========================
        # AUGMENTATIONS
        # =========================

        augmented = self.transform(
            image=pre,
            mask=mask
        )

        pre = augmented['image']
        mask = augmented['mask']

        augmented_post = self.transform(
            image=post
        )

        post = augmented_post['image']

        # =========================
        # EO NORMALIZATION
        # =========================

        pre = pre.astype(np.float32) / 255.0

        pre = np.clip(pre, 0, 1)

        # =========================
        # SAR NORMALIZATION
        # =========================

        post = np.nan_to_num(post)

        p1 = np.percentile(post, 1)
        p99 = np.percentile(post, 99)

        post = np.clip(post, p1, p99)

        post = (
            (post - p1) /
            (p99 - p1 + 1e-6)
        )

        post = np.clip(post, 0, 1)

        # =========================
        # SAR 1-CHANNEL -> 3-CHANNEL
        # =========================

        if len(post.shape) == 2:
            post = np.expand_dims(
                post,
                axis=-1
            )

        if post.shape[-1] == 1:

            post = np.repeat(
                post,
                3,
                axis=-1
            )

        # =========================
        # CLEAN NaNs
        # =========================

        pre = np.nan_to_num(
            pre,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        post = np.nan_to_num(
            post,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        mask = np.nan_to_num(
            mask,
            nan=0.0
        )

        # =========================
        # TO TENSORS
        # =========================

        pre = torch.tensor(pre) \
            .permute(2, 0, 1) \
            .float()

        post = torch.tensor(post) \
            .permute(2, 0, 1) \
            .float()

        mask = torch.tensor(mask) \
            .unsqueeze(0) \
            .float()

        return {
            'pre_image': pre,
            'post_image': post,
            'mask': mask
        }