"""Code is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
"""
import numpy as np
import torch


def to_one_hot(x, num_classes, on_value=1., off_value=0.):
    if len(x.size()) > 1 and x.size(-1) == num_classes:
        # already one-hot form
        return x
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device
                      ).scatter_(dim=1, index=x, value=on_value)


def squeeze_one_hot(x):
    return x.topk(1)[1].long()


def smooth_target(target, smoothing=0.1, num_classes=1000):
    target *= (1. - smoothing)
    target += (smoothing / num_classes)
    return target


def mix_target(target_a, target_b, num_classes, lam=1., smoothing=0.0):
    y1 = smooth_target(target_a, smoothing, num_classes)
    y2 = smooth_target(target_b, smoothing, num_classes)
    return lam * y1 + (1. - lam) * y2


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[2] * size[3]))

    return bbx1, bby1, bbx2, bby2, lam


def cutmix_batch(image, target, mix_args):
    if not mix_args.disable and np.random.rand(1) < mix_args.prob:
        lam = np.random.beta(mix_args.beta, mix_args.beta)
    else:
        target = smooth_target(
            target=target,
            smoothing=mix_args.smoothing,
            num_classes=mix_args.num_classes)
        return image, target

    rand_index = torch.randperm(image.size()[0], device=image.device)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2, lam = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2,
                                              bby1:bby2]

    target = mix_target(target_a=target_a,
                        target_b=target_b,
                        num_classes=mix_args.num_classes,
                        lam=lam,
                        smoothing=mix_args.smoothing)
    return image, target
