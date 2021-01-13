# relabel_imagenet
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import math
import os

import torch
import torchvision
from torchvision.datasets.folder import ImageFolder

from utils.utils import IMAGENET_MEAN_01
from utils.utils import IMAGENET_STD_01
from utils.utils import get_lighting
from utils.data_augment import to_one_hot

from utils.relabel_functions import get_labelmaps
from utils.relabel_functions import get_relabel
from utils.relabel_functions import RandomResizedCropWithCoords
from utils.relabel_functions import ImageFolderWithCoordsAndLabelMap
from utils.relabel_functions import ComposeWithCoords


def get_torch_dataloader(data_path, batch_size, val_batch_size,
                         num_workers, args,
                         folder_train='train',
                         folder_val='val'):
    traindir = os.path.join(data_path, folder_train)
    valdir = os.path.join(data_path, folder_val)
    normalize = torchvision.transforms.Normalize(mean=IMAGENET_MEAN_01,
                                                 std=IMAGENET_STD_01)
    lighting = get_lighting()

    if args.data.relabel.use:
        relabel_root = args.data.relabel.path

        train_transform_list = [
            RandomResizedCropWithCoords(size=args.data.image_size,
                                        scale=(args.data.rrc_minimum, 1)),
            torchvision.transforms.RandomHorizontalFlip(),
        ]

        train_transform_list += [
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4),
            torchvision.transforms.ToTensor(),
            lighting,
            normalize,
        ]

        train_dataset = ImageFolderWithCoordsAndLabelMap(
            relabel_root=relabel_root,
            root=traindir,
            transform=ComposeWithCoords(transforms=train_transform_list)
        )

    else:
        train_transform_list = [
            torchvision.transforms.RandomResizedCrop(
                args.data.image_size,
                scale=(args.data.rrc_minimum, 1)),
            torchvision.transforms.RandomHorizontalFlip()]

        train_transform_list += [
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4),
            torchvision.transforms.ToTensor(),
            lighting,
            normalize,
        ]

        train_dataset = ImageFolder(
            root=traindir,
            transform=torchvision.transforms.Compose(train_transform_list)
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(valdir, torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                int(math.ceil(args.data.image_size / args.data.crop_ratio))),
            torchvision.transforms.CenterCrop(args.data.image_size),
            torchvision.transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    def original_batch_fn(batch, num_classes=1000):
        images = batch[0]
        original_label = to_one_hot(batch[1], num_classes)

        return images, original_label

    def relabel_batch_fn(batch, num_classes=1000, mode=None):
        images = batch[0]
        original_label = to_one_hot(batch[1], num_classes)

        if mode == 'val':
            return images, original_label

        num_batches = images.size(0) # number of mini-batch
        random_crop_coords = batch[2] # random crop augmentation coordinates
        label_maps = batch[3]  # sparse top-k label maps

        # make random crop coords to [x1,y1,x2,y2]
        random_crop_coords = torch.cat(random_crop_coords, 0).view(4, -1).transpose(1, 0)
        random_crop_coords[:, 2:] += random_crop_coords[:, :2]
        random_crop_coords = random_crop_coords.cuda().float()

        # make full tensor from sparse top-k label maps
        label_maps = get_labelmaps(label_maps_topk=label_maps,
                                   num_batches=num_batches)

        # LabelPooling operation
        relabel = get_relabel(label_maps=label_maps,
                              batch_coords=random_crop_coords,
                              num_batches=num_batches)

        return images, original_label, relabel

    print(f"Training epoch size: {len(train_loader.dataset)}")
    print(f"Validation epoch size: {len(val_loader.dataset)}")

    if args.data.relabel.use:
        batch_fn = relabel_batch_fn
    else:
        batch_fn = original_batch_fn

    return train_loader, val_loader, batch_fn


def load_data_loaders(dataset, args, context):
    data_path = args.data.data_path

    if dataset.lower() == 'imagenet':
        return get_torch_dataloader(
            data_path=data_path,
            batch_size=context.batch_size,
            val_batch_size=context.val_batch_size,
            num_workers=context.num_workers,
            args=args,
        )
    else:
        raise ValueError("Unknown dataset {}.".format(dataset))
