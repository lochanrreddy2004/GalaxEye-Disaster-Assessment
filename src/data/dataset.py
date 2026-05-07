import os
import cv2
import torch
import numpy as np
import tifffile as tiff

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, image_size=512, train=True):

        self.pre_dir = os.path.join(root_dir, "pre_event")
        self.post_dir = os.path.join(root_dir, "post_event")
        self.mask_dir = os.path.join(root_dir, "target")

        # self.files = sorted(os.listdir(self.pre_dir))
        self.files = sorted(os.listdir(self.pre_dir))[:100]

        if train:
            self.transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406, 0.5),
                            std=(0.229, 0.224, 0.225, 0.25)),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406, 0.5),
                            std=(0.229, 0.224, 0.225, 0.25)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        pre_path = os.path.join(self.pre_dir, file_name)
        post_path = os.path.join(self.post_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        pre_img = tiff.imread(pre_path)
        post_img = tiff.imread(post_path)
        mask = tiff.imread(mask_path)

        pre_img = pre_img.astype(np.float32) / 255.0

        post_img = post_img.astype(np.float32)
        post_img = (post_img - post_img.min()) / (
            post_img.max() - post_img.min() + 1e-8
        )

        post_img = np.expand_dims(post_img, axis=-1)

        combined = np.concatenate([pre_img, post_img], axis=-1)

        transformed = self.transforms(
            image=combined,
            mask=mask
        )

        image = transformed["image"]
        mask = transformed["mask"]

        mask = mask.unsqueeze(0).float()

        return image, mask