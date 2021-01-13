# relabel_imagenet
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.optim as optim
from adamp import AdamP
import numpy as np

def set_init_lr(param_groups):
    for group in param_groups:
        group['init_lr'] = group['lr']

def load_optimizer(args, param_group):

    if args.optim.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            param_group,
            args.optim.lr.init,
            momentum=args.optim.momentum,
            weight_decay=args.optim.wd.base,
            nesterov=args.optim.nesterov,
        )
    elif args.optim.optimizer.lower() == 'adamp':
        optimizer = AdamP(
            param_group,
            args.optim.lr.init,
            betas=(args.optim.momentum, 0.999),
            weight_decay=args.optim.wd.base,
            nesterov=args.optim.nesterov,
        )
    else:
        raise ValueError("Unknown optimizer : {}".format(args.optim.optimizer))

    set_init_lr(optimizer.param_groups)

    return optimizer


def warmup_learnig_rate(init_lr, warmup_lr, warmup_epochs,
                        iteration, epoch, dataset_len):
    lr = warmup_lr + (init_lr - warmup_lr) * \
         float(iteration + epoch * dataset_len) / (warmup_epochs * dataset_len)
    return lr


def adjust_learning_rate_cosine(epoch, iteration, dataset_len,
                                epochs, warmup_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    total_iter = (epochs - warmup_epochs) * dataset_len
    current_iter = iteration + (epoch - warmup_epochs) * dataset_len

    lr = 1 / 2 * (np.cos(np.pi * current_iter / total_iter) + 1)

    return lr


def adjust_learning_rate_linear(epoch, iteration, dataset_len,
                                epochs, warmup_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    total_iter = (epochs - warmup_epochs) * dataset_len
    current_iter = iteration + (epoch - warmup_epochs) * dataset_len

    lr = 1 - current_iter / total_iter

    return lr


def adjust_learning_rate_default(epoch, iteration, dataset_len,
                                 epochs, warmup_epochs):
    """
    LR schedule that should yield 76% converged accuracy
    with batch size 256"""
    factor = epoch // 30

    lr = 0.1 ** factor

    return lr


def adjust_learning_rate(optimizer, epoch, iteration, lr_decay_type,
                         epochs, train_len, warmup_lr, warmup_epochs):
    if epoch < warmup_epochs:
        for param_group in optimizer.param_groups:
            lr = warmup_learnig_rate(init_lr=param_group['init_lr'],
                                     warmup_lr=warmup_lr,
                                     warmup_epochs=warmup_epochs,
                                     epoch=epoch,
                                     iteration=iteration,
                                     dataset_len=train_len)
            param_group['lr'] = lr
    else:
        if lr_decay_type == 'cos':
            lr_function = adjust_learning_rate_cosine
        elif lr_decay_type == 'linear':
            lr_function = adjust_learning_rate_linear
        elif lr_decay_type == 'default':
            lr_function = adjust_learning_rate_default
        else:
            raise ValueError("Unknown lr decay type {}."
                             .format(lr_decay_type))

        lr_factor = lr_function(epoch=epoch,
                                iteration=iteration,
                                dataset_len=train_len,
                                epochs=epochs,
                                warmup_epochs=warmup_epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['init_lr'] * lr_factor

    return optimizer.param_groups[0]['lr']
