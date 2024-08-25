import glob
import json
import random

import numpy as np
import torch
from PIL import Image


def norm(image):
    return (image / 127.5) - 1.0

def denorm(image):
    return (image + 1.0) * 127.5

def augment(dt_im, eh_im):
    # Random interpolation
    a = random.random()
    dt_im = dt_im * a + eh_im * (1 - a)

    # Random flip left right
    if random.random() < 0.25:
        dt_im = np.fliplr(dt_im)
        eh_im = np.fliplr(eh_im)

    # Random flip up down
    if random.random() < 0.25:
        dt_im = np.flipud(dt_im)
        eh_im = np.flipud(eh_im)

    return dt_im, eh_im


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size, split):
        super(PairDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.split = split

        # Load JSON of splits
        names = json.load(open(f"{self.data_root}/splits.json", "r"))[self.split]

        # Build image paths
        self.dt_ims = [f"{self.data_root}/trainA/{n}" for n in names]
        self.eh_ims = [f"{self.data_root}/trainB/{n}" for n in names]
        print(f"Total {len(self.dt_ims)} data")

    def __getitem__(self, index):
        # Read and resize image pair
        dt_im = Image.open(self.dt_ims[index]).convert("RGB")
        eh_im = Image.open(self.eh_ims[index]).convert("RGB")
        dt_im = dt_im.resize(self.im_size)
        eh_im = eh_im.resize(self.im_size)

        # Transfrom image pair to float32 np.ndarray
        dt_im = np.array(dt_im, dtype=np.float32)
        eh_im = np.array(eh_im, dtype=np.float32)

        # Augment image pair
        if self.split == "train":
            dt_im, eh_im = augment(dt_im, eh_im)

        # Transfrom image pair to (C, H, W) torch.Tensor
        dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
        eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
        return dt_im, eh_im

    def __len__(self):
        return len(self.dt_ims)

from random import shuffle
from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
import torchvision
import albumentations


def prepare_image(image_dir):
    image = cv2.imread(filename=image_dir)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    return image


class CustomDataset(Dataset):
    def __init__(self, images_dir: str,
                 device: torch.device = None,
                 transform=None):
        super().__init__()
        self.root_image_dir:str = images_dir

        self.lr_images:str = os.path.join(self.root_image_dir, "trainA")
        self.hr_images:str = os.path.join(self.root_image_dir, "trainB")
        self.external_dataset:str = "/home/muahmmad/projects/Image_enhancement/dataset/Enhancement_Dataset"

        self.lr_images_list = []
        self.hr_images_list = []
        for file in os.listdir(self.lr_images):
            self.lr_images_list.append(
                os.path.join(self.root_image_dir, self.lr_images, file)
            )
        for file in os.listdir(self.external_dataset):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.lr_images_list.append(
                    os.path.join(self.external_dataset,file))
        for file in os.listdir(self.hr_images):
            self.hr_images_list.append(
                os.path.join(self.root_image_dir, self.hr_images, file)
            )
        self.lr_images_list = sorted(self.lr_images_list)
        self.hr_images_list = sorted(self.hr_images_list)
        self.hr_length = len(self.hr_images_list)
        self.lr_length = len(self.lr_images_list)
        self.data_set_length = max(self.lr_length, self.hr_length)
        self.transform = transform
        self.device = device

    def __len__(self) -> int:
        return self.data_set_length

    def __getitem__(self, idx: int):
        lr_image_file = self.lr_images_list[idx % self.lr_length]
        hr_image_file = self.hr_images_list[idx % self.hr_length]
        lr_image = cv2.cvtColor(src=cv2.imread(lr_image_file), code=cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(src=cv2.imread(hr_image_file), code=cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            aug = self.transform(image=lr_image, hr_image=hr_image)
            lr_image = aug["image"]
            hr_image = aug["hr_image"]
        return lr_image, hr_image

class UnpairDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size, split):
        super(UnpairDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.split = split

        # Load JSON of splits
        names = json.load(open(f"{self.data_root}/splits.json", "r"))[self.split]

        # Build image paths
        self.dt_ims = [f"{self.data_root}/{n}" for n in names if "trainA" in n]
        self.eh_ims = [f"{self.data_root}/{n}" for n in names if "trainB" in n]
        print(f"Total {len(self.dt_ims)} poor quality data")
        print(f"Total {len(self.eh_ims)} good quality data")

        # Force # of images to the least amount
        num = min(len(self.dt_ims), len(self.eh_ims))
        self.dt_ims = self.dt_ims[:num]
        self.eh_ims = self.eh_ims[:num]
        print(f"Total {len(self.eh_ims)} data used")

    def __getitem__(self, index):
        # Read and resize image pair
        dt_im = Image.open(self.dt_ims[index]).convert("RGB")
        eh_im = Image.open(self.eh_ims[index]).convert("RGB")
        dt_im = dt_im.resize(self.im_size)
        eh_im = eh_im.resize(self.im_size)

        # Transfrom image pair to float32 np.ndarray
        dt_im = np.array(dt_im, dtype=np.float32)
        eh_im = np.array(eh_im, dtype=np.float32)

        # Augment image pair
        if self.split == "train":
            dt_im, eh_im = augment(dt_im, eh_im)

        # Transfrom image pair to (C, H, W) torch.Tensor
        dt_im = torch.Tensor(norm(dt_im)).permute(2, 0, 1)
        eh_im = torch.Tensor(norm(eh_im)).permute(2, 0, 1)
        return dt_im, eh_im

    def __len__(self):
        return len(self.dt_ims)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size):
        super(TestDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.ims = glob.glob(f"{self.data_root}/*")

    def __getitem__(self, index):
        # Read and resize image
        path = self.ims[index]
        im = Image.open(path).convert("RGB")
        im = im.resize(self.im_size)

        # Transfrom image to float32 np.ndarray
        im = np.array(im, dtype=np.float32)

        # Transfrom image to (C, H, W) torch.Tensor
        im = torch.Tensor(norm(im)).permute(2, 0, 1)
        return path, im

    def __len__(self):
        return len(self.ims)


if __name__ == "__main__":
    dataset = PairDataset(
        data_root="../data/EUVP Dataset/Paired/underwater_dark", im_size=(256, 256))
    image, target = dataset[0]
