# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import random
from PIL import Image, ImageFilter, ImageOps

from folder import ImageFolder, default_loader
import os.path as osp



IMAGE_SIZE = {"stl10":96, "cifar10":36, "imagenet": 224, "mini-imagenet-1k": 84, "sub-imagenet100": 84, "imagenet100": 224}

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_imagenet(dataset, dataset_root, aug_batch_size, split="train"):
    # Data loading code
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if dataset == "mini-imagenet-1k" or dataset == "sub-imagenet100" or dataset == "imagenet100":
        image_size = IMAGE_SIZE[dataset]
    else:
        image_size = IMAGE_SIZE["imagenet"]

    normalize = transforms.Normalize(mean=mean, std=std)

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    # simclr
    augmentation = [
        transforms.RandomResizedCrop((image_size,image_size), scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]
    basic_preprocess = [
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        # normalize
    ]
    augmentation_transform = transforms.Compose(augmentation)
    basic_transform = transforms.Compose(basic_preprocess)

    if dataset == "mini-imagenet-1k":
        if split == "train":
            pathfile = "split/mini-imagenet-1k/MiniImagenet-1k-train.txt"
        elif split == "test":
            pathfile = "split/mini-imagenet-1k/MiniImagenet-1k-test.txt"
    elif dataset == "sub-imagenet100":
        if split == "train":
            pathfile = "split/imagenet-100/SubImageNet_100_train.txt"
        elif split == "test":
            pathfile = "split/imagenet-100/ImageNet_100_test.txt"
    elif dataset == "imagenet100":
        if split == "train":
            pathfile = "split/imagenet-100/ImageNet_100_train.txt"
        elif split == "val":
            pathfile = "split/imagenet-100/ImageNet_100_val.txt"
        else:
            pathfile = "split/imagenet-100/ImageNet_100_test.txt"
    elif dataset == "imagenet1000":    
        # automatically traverse the whole ~:/data/imagenet directory
        pathfile = None
        if split == "val":
            pathfile = ""
            raise NotImplementedError("validation split of imagenet1000 not defined.")

    dataset = ImageFolder(dataset_root, split, transform=basic_transform, aug_transform=augmentation_transform, aug_batch_size=aug_batch_size, pathfile=pathfile, distributed=True)
    rawdataset = ImageFolder(dataset_root, split, transform=basic_transform, aug_transform=None, aug_batch_size=None, pathfile=pathfile, distributed=True)
    global_rawdataset = ImageFolder(dataset_root, split, transform=basic_transform, aug_transform=None, aug_batch_size=None, pathfile=pathfile, distributed=False)
    return dataset, rawdataset, global_rawdataset


def get_dataset(dataset, rank, world_size, root, transform_batch_size, download=True, split=None, finite_aug=False, n_aug=None, get_subset=False):
    """
        When split is None, return the default unsupervised training set.
    """
    if not finite_aug:
        if dataset == "cifar10":
            torchdata = datasets.CIFAR10(
                root=root,
                train= split!="test",
                download=download,
            )


        elif dataset == "stl10":
            torchdata = datasets.STL10(
                root=root,
                split=split if not split is None else "unlabeled",
                download=True,
            )
    
        data, labels = [], []
        dataset_size = len(torchdata)
        for i in range(dataset_size):
            img, lab = torchdata[i]
            data.append(img)
            labels.append(lab)


        # splitting the dataset so that the sequence of batch indices is consistent for any world_size.
        local_data, local_labels = [], []
        for i in range(dataset_size // world_size):
            local_data.append( data[ rank + i * world_size ] )
            local_labels.append( labels[ rank + i * world_size ] )
        data, labels = local_data, local_labels
            

        image_size = IMAGE_SIZE[dataset]


        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        aug_transform = transforms.Compose([transforms.RandomResizedCrop(size=image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(),
                                            transforms.ToTensor()])
        basic_transform = transforms.Compose([transforms.Resize(size=image_size),
                                                transforms.ToTensor()])
        # aug_transform = basic_transform # debug: disabled augmentation

        return IndexedAugmentedDataset(data, labels, basic_transform, aug_transform, transform_batch_size), IndexedDataset(data, labels, basic_transform)
    else:
        assert n_aug is not None, "Please specify the number of augmentations."
        image_size = IMAGE_SIZE[dataset]
        basic_transform = transforms.Compose([transforms.Resize(size=image_size),
                                        transforms.ToTensor()])
        if dataset == "stl10" and split is None:
            if get_subset:
                total_num_of_samples = 500
            else:
                total_num_of_samples = 100000
            return IndexedFiniteAugmentedDataset(osp.join(root, "stl10_{}_aug".format(n_aug)), basic_transform, 2, total_num_of_samples, n_aug, rank, world_size, num_classes=10), \
                    IndexedFiniteDataset(osp.join(root, "stl10_{}_aug".format(n_aug)), basic_transform, total_num_of_samples, rank, world_size, num_classes=10)
        else:
            raise NotImplementedError("finite augmentation not implemented for dataset {} split {}".format(dataset, split))
             





class IndexedDataset(Dataset):
    def __init__(self, x, y, transform, num_classes=10):
        self._data = x
        self._label = y
        self.transform = transform
        self.num_classes = num_classes
    
    def __getitem__(self, idx):
        img, label = self._data[idx], self._label[idx]
        return self.transform(img), [], label, idx, []

    def __len__(self):
        return len(self._data)


class IndexedAugmentedDataset(Dataset):
    def __init__(self, x, y, transform, aug_transform, aug_batch_size, num_classes=10):
        self._data = x
        self._label = y
        self.transform = transform
        self.aug_transform = aug_transform
        self.aug_batch_size = aug_batch_size
        self.num_classes = num_classes
    
    def __getitem__(self, idx):
        img, label = self._data[idx], self._label[idx]
        aug_imgs = [self.aug_transform(img) for _ in range(self.aug_batch_size)]
        return self.transform(img), aug_imgs, label, idx, []

    def __len__(self):
        return len(self._data)


""" For precomputed augmentation dataset. """
class IndexedFiniteAugmentedDataset(Dataset):
    def __init__(self, root, transform, aug_batch_size, num_samples, n_augmentations, rank, world_size, num_classes=10):
        self.root = root
        self.transform = transform
        self.aug_batch_size = aug_batch_size
        self.n_augmentations = n_augmentations
        self.loader = default_loader
        self.range = np.arange( int(num_samples * rank / world_size), int(num_samples * (rank+1) / world_size) )
        self.num_samples = len(self.range)
        self.num_classes = num_classes
    

    def __getitem__(self, idx):
        g_idx = self.range[idx]
        file_dir = osp.join(self.root, "IMG-{}".format(g_idx))
        raw_img_path = osp.join(file_dir, "IMG-{}.png".format(g_idx))
        img = self.loader(raw_img_path)

        aug_idxs = np.random.choice(self.n_augmentations, self.aug_batch_size, replace=False)
        aug_img_paths = [osp.join(file_dir, "IMG-{}-aug-{}.png".format(g_idx, k)) for k in aug_idxs]
        aug_imgs = [self.transform(self.loader(p)) for p in aug_img_paths]

        label = -1
        return self.transform(img), aug_imgs, label, idx, aug_idxs.tolist()


    def get_by_aug_idx(self, idx, aug_idx):
        g_idx = self.range[idx]
        file_dir = osp.join(self.root, "IMG-{}".format(g_idx))
        aug_img_path = osp.join(file_dir, "IMG-{}-aug-{}.png".format(g_idx, aug_idx))
        aug_img = self.transform( self.loader(aug_img_path) )
        return aug_img


    def __len__(self):
        return self.num_samples


class IndexedFiniteDataset(Dataset):
    def __init__(self, root, transform, num_samples, rank, world_size, num_classes=10):
        self.root = root
        self.transform = transform
        self.num_samples = num_samples
        self.loader = default_loader
        self.range = np.arange( int(num_samples * rank / world_size), int(num_samples * (rank+1) / world_size) )
        self.num_samples = len(self.range)
        self.num_classes = num_classes
    

    def __getitem__(self, idx):
        g_idx = self.range[idx]
        file_dir = osp.join(self.root, "IMG-{}".format(g_idx))
        raw_img_path = osp.join(file_dir, "IMG-{}.png".format(g_idx))
        img = self.loader(raw_img_path)

        label = -1
        return self.transform(img), [], label, idx, []

    def __len__(self):
        return self.num_samples