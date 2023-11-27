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
import matplotlib.pyplot as plt
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))

def build_dataset(is_train, args, include_path=False):
    if args.dataset == "tiny_imagenet":
        dataset = TinyImagenet(transform=simple_transform(args), split='train' if is_train else 'val',
                                subset=args.data_subset, group=args.data_group, include_path=include_path)
    else:
        raise NotImplementedError("Not implemented yet")
    print(dataset)
    return dataset

def simple_transform(args):
    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train

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
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class TinyImagenet(Dataset):
    def __init__(self, transform=None, split='train', subset=1.0, group=1, include_path=False) -> None:
        super().__init__()
        self.split = split
        self.transform = transform
        self.group = group
        self.path = self._get_path()
        self.md = self._get_md()
        self.subset = subset
        self.include_path = include_path
    
    def __len__(self):
        return int(len(self.md) * self.subset)
    
    def __getitem__(self, index):
        row = self.md.iloc[index]
        label = None
        if self.split == 'train' or self.split == 'val':
            img_path = os.path.join(os.path.join(self.path, row["class"]), row["id"])
            label = row["label"]
        else:
            img_path = os.path.join(self.path, row["id"])
        image = PIL.Image.open(img_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.include_path:
            return image, label, img_path
        else:
            return image, label
    
    def _get_path(self):
        if self.split == 'train':
            return config("TINYIMAGENET_TRAIN_PATH")
        elif self.split == 'val':
            return config("TINYIMAGENET_VAL_PATH")
        else:
            return config("TINYIMAGENET_TEST_PATH")
    
    def _get_md(self):
        if self.split == 'train':
            df = pd.read_csv(config("TINYIMAGENET_TRAIN_META_PATH"))
            md = df.loc[df["group"] == self.group]
        elif self.split == 'val':
            md = pd.read_csv(config("TINYIMAGENET_VAL_META_PATH"))
        else:
            md = pd.read_csv(config("TINYIMAGENET_TEST_META_PATH"))
        assert len(md) > 0
        return md
    
def visualize_image(img: torch.Tensor, path: str, wandb_log=False):
    if not isinstance(img, torch.Tensor):
        raise NotImplementedError(f"Please pass in tensor objects, dtype ({type(img)}) is not supported")
    # if passed in batched images
    if len(img.shape) > 3:
        img = img[0]
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    img = img * torch.tensor(std).cuda().view(3, 1, 1) + torch.tensor(mean).cuda().view(3, 1, 1)
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    fig.savefig(path)
    plt.clf()
