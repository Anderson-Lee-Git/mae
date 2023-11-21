# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# https://github.com/emma-mens/elk-recognition/blob/main/src/multimodal_species/datasets/birds_dataset.py#L498
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv("../.env"))

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = ImagenetDataset(transform, split='train' if is_train else 'val', subset=args.data_subset)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        transform.transforms.insert(0, transforms.ToPILImage())
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.ToPILImage())
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImagenetDataset(Dataset):
    def __init__(self, transform=None, split='train', size=256, crop=224, subset=1.0):
        # get length of total data for train and for val (test)
        # shuffle arange(train length) and pick last 20% for validation
        # index into imagenet h5py
        self.split = split
        self.transform = transform
        self.path = config("IMAGENET-PATH")
        self.h5_split = 'valid' if split == 'test' else 'train'
        self.val_p = 0.2
        with h5py.File(self.path, 'r') as hf:
            labels = np.array(hf[f"/{self.h5_split}/labels"][:]).reshape(-1)

        idx = np.arange(len(labels))
        # make train/val split
        train, val = idx, idx
        if split != 'test':
            train, val = self._train_val_split(idx, labels, self.val_p)
        if split == 'train':
            self.idx = train
        elif split == 'val':
            self.idx = val
        else:
            self.idx = idx
        # take subset of dataset if needed
        if subset < 1.0 and split == 'train':
            self.idx = np.random.choice(self.idx, int(subset*len(self.idx)))

    def __len__(self):
        return len(self.idx)

    def __del__(self):
        if hasattr(self, 'hdf5_path'):
            self.hdf5_path.close()

    def open_hdf5(self):
        self.hdf5_path = h5py.File(self.path, 'r')
        self.labels = self.hdf5_path[f'/{self.h5_split}/labels']
        self.imgs = self.hdf5_path[f'/{self.h5_split}/images']

    def _train_val_split(self, idx, labels, val_p=0.2):
        train = []
        val = []
        for c in np.unique(labels):
            c = idx[labels == c]
            # shuffle
            np.random.shuffle(c)
            n_val = int(val_p*len(c))
            n_train = len(c) - n_val
            train.extend(idx[c[:n_train]])
            val.extend(idx[c[-n_val:]])
        return train, val

    def __getitem__(self, idx):
        if not hasattr(self, f'hdf5_path'):
            # https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
            self.open_hdf5()
        image, label = self.imgs[idx], self.labels[idx][0]
        image = (image - image.min())/(image.max() - image.min()) * 255
        if self.transform:
            image = image.astype(np.uint8)
            image = self.transform(image)
        return image, image
    
def visualize_image(img: torch.Tensor, path: str):
    if not isinstance(img, torch.Tensor):
        raise NotImplementedError(f"Please pass in tensor objects, dtype ({type(img)}) is not supported")
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    img = img * torch.tensor(std).cuda().view(3, 1, 1) + torch.tensor(mean).cuda().view(3, 1, 1)
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    fig.savefig(path)

    