import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data import IndexedAugmentedDataset, GaussianBlur

from tqdm import tqdm

import os.path as osp
from pathlib import Path

IMAGE_SIZE = {"stl10":96, "cifar10":36, "imagenet": 224, "mini-imagenet-1k": 84, "sub-imagenet100": 84, "imagenet100": 224}

def augment_and_save_to_disk(dataset, root, n_augmentations=10, download=True):
    split = "train"
    if dataset == "cifar10":
        torchdata = datasets.CIFAR10(
            root=root,
            train= split!="test",
            download=download,
        )


    elif dataset == "stl10":
        torchdata = datasets.STL10(
            root=root,
            split="unlabeled",
            download=True,
        )
    
    
    data, labels = [], []
    dataset_size = len(torchdata)
    for i in range(dataset_size):
        img, lab = torchdata[i]
        data.append(img)
        labels.append(lab)



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

    aug_dataset = IndexedAugmentedDataset(data, labels, basic_transform, aug_transform, n_augmentations)
    aug_dataloader = DataLoader(aug_dataset, batch_size=1024, shuffle=False, num_workers=8)

    # path_label_ln = []
    for images, aug_images, labels, idx, _ in tqdm(aug_dataloader):
        for i, j in enumerate(idx.tolist()):
            for k in range(n_augmentations):
                augimg = aug_images[k][i]
                save_dir = osp.join(root, "{}_{}_aug".format(dataset, n_augmentations), "IMG-{}".format(j))
                Path(save_dir).mkdir(parents=True, exist_ok=True)

                fn = "IMG-{}-aug-{}.png".format(j, k)
                file_path = osp.join(save_dir, fn)
                save_image(augimg, file_path)

                rawimg = images[i]
                fn = "IMG-{}.png".format(j)
                file_path = osp.join(save_dir, fn)
                save_image(rawimg, file_path)
                # print("saved {}".format(file_path))
                # path_label_ln.append("{} {}".format(file_path, labels[i]))
        # torch.save((aug_images, labels), )

if __name__ == "__main__":
    dataset = "stl10"
    root = "/home/ubuntu/oscar_const/data"
    augment_and_save_to_disk(dataset, root, 2)