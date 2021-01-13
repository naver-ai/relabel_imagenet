# relabel_imagenet
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import fire
import warnings

import torch

from utils.configs import parse_config
from utils.utils import mch
from utils.utils import set_distributed_worker
from utils.utils import set_dataset_len
from train import Trainer


def main_worker(gpu, args, context):
    set_distributed_worker(gpu, args, context)

    trainer = Trainer(args=args, context=context)

    if args.util.test_only:
        trainer.validate()
        return

    for epoch in range(args.optim.start_epoch, args.optim.epochs):
        trainer.train(epoch=epoch)
        trainer.validate(epoch=epoch)

def get_context(args):
    ngpus_per_node = torch.cuda.device_count()
    print(f'device_count : {ngpus_per_node}')

    context = mch()

    # Number of the GPUs per node
    context.ngpus_per_node = ngpus_per_node

    # Number of the workers per GPU
    context.num_workers = int((args.compute.num_workers + ngpus_per_node - 1) /
                              ngpus_per_node)

    # The total number of processes, so that the master knows how many workers
    # to wait for.
    if args.compute.distributed.use:
        context.world_size = args.compute.distributed.num_processes_per_node * \
                             context.ngpus_per_node
    else:
        context.world_size = args.compute.distributed.num_processes_per_node

    # Rank of each process, so they will know whether it is the master of a
    # worker. Will be set on the set_distributed_worker()
    context.rank = 0

    # Training batch size per GPU
    context.batch_size = int(args.optim.batch_size / ngpus_per_node)

    # Validation batch size per GPU
    context.val_batch_size = int(args.util.val_batch_size / ngpus_per_node)

    # Data path
    context.data_path = args.data.data_path

    # Number of classes
    context.num_classes = 1000

    # Dataset length
    context.train_len, context.val_len = set_dataset_len(
        batch_size=context.batch_size,
        val_batch_size=context.val_batch_size,
        ngpus_per_node=ngpus_per_node)

    return context

def main(config_file_path, **kwargs):
    args = parse_config(config_fname=config_file_path, **kwargs)

    if args.compute.gpu >= 0:
        warnings.warn('You have chosen a specific GPU. This will '
                      'disable data parallelism.')

    # Common set of data passed to workers
    context = get_context(args)

    torch.set_num_threads(torch.get_num_threads() * 2)
    print('CPU threads :' + str(torch.get_num_threads()))

    if args.compute.distributed.use:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(main_worker,
                                    nprocs=context.ngpus_per_node,
                                    args=(args, context))
    else:
        # Simply call main_worker function
        main_worker(args.compute.gpu, args, context)


if __name__ == '__main__':
    fire.Fire(main)
