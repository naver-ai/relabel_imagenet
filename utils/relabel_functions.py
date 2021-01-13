# relabel_imagenet
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import os

import torch
import torch.distributed
import torch.nn as nn
import torchvision
from torchvision.ops import roi_align
from torchvision.transforms import functional as torchvision_F


def get_labelmaps(label_maps_topk, num_batches):
    label_maps_topk_sizes = label_maps_topk[0].size()

    label_maps = torch.zeros([num_batches, 1000, label_maps_topk_sizes[2],
                              label_maps_topk_sizes[3]])
    for _label_map, _label_topk in zip(label_maps, label_maps_topk):
        _label_map = _label_map.scatter_(
            0,
            _label_topk[1][:, :, :].long(),
            _label_topk[0][:, :, :]
        )
    label_maps = label_maps.cuda()
    return label_maps


def get_relabel(label_maps, batch_coords, num_batches):
    target_relabel = roi_align(
        input=label_maps,
        boxes=torch.cat(
            [torch.arange(num_batches).view(num_batches,
                                            1).float().cuda(),
             batch_coords.float() * label_maps.size(3) - 0.5], 1),
        output_size=(1, 1))
    target_relabel = torch.nn.functional.softmax(target_relabel.squeeze(), 1)
    return target_relabel


class ComposeWithCoords(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img):
        coords = None
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomHorizontalFlipWithCoords':
                img, coords = t(img, coords)
            else:
                img = t(img)
        return img, coords


class RandomResizedCropWithCoords(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        coords = (i / img.size[1],
                  j / img.size[0],
                  h / img.size[1],
                  w / img.size[0])
        return torchvision_F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords


class ImageFolderWithCoordsAndLabelMap(torchvision.datasets.ImageFolder):
    def __init__(self, **kwargs):
        self.relabel_root = kwargs['relabel_root']
        kwargs.pop('relabel_root')
        super(ImageFolderWithCoordsAndLabelMap, self).__init__(**kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        score_path = os.path.join(
            self.relabel_root,
            '/'.join(path.split('/')[-2:]).split('.')[0] + '.pt')

        sample = self.loader(path)
        if self.transform is not None:
            sample, coords = self.transform(sample)
        else:
            coords = None
        if self.target_transform is not None:
            target = self.target_transform(target)

        score_maps = torch.load(score_path)

        return sample, target, coords, score_maps
