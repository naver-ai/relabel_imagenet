# relabel_imagenet
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import os
import sys
import time
import shutil

import torch
import torch.nn as nn
import torchvision

from utils.data_loader import load_data_loaders
from utils.data_augment import cutmix_batch
from utils.data_augment import squeeze_one_hot
from utils.loss import SoftTargetCrossEntropy
from utils.optimizers import adjust_learning_rate
from utils.optimizers import load_optimizer
from utils.utils import AverageMeterCollection
from utils.utils import PrintCollection
from utils.utils import mch
from utils.utils import compute_accuracy_dist
from utils.utils import get_state_dict
from utils.utils import reduce_tensor
from utils.utils import remove_prefix_checkpoint

try:
    import apex
    from apex import amp
    # pylint: disable=unused-import
    from apex.multi_tensor_apply import multi_tensor_applier
    # pylint: enable=unused-import
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex "
        "to run this example.")

class Trainer(object):
    def __init__(self, args, context):
        self.args = args
        self.context = context
        self.best_acc1 = 0

        # Load initial network architecture
        if hasattr(torchvision.models, args.model.arch):
            self.model = getattr(torchvision.models, args.model.arch)()
        else:
            raise ValueError(
                f"Not supported model architecture {args.model.arch}")

        if self._get_condition_for_save_and_log():
            print(self.model)

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 1 - args.optim.bn.momentum
                m.eps = args.optim.bn.eps

        # Load pretrained checkpoint if "test_only" mode
        if self.args.util.test_only:
            self.load_checkpoint(self.args.util.test_weight_file)

        # Set criterion and opimizer
        self.criterion = SoftTargetCrossEntropy()

        param_group = self.model.parameters()
        self.optimizer = load_optimizer(args, param_group)
        if self._get_condition_for_save_and_log():
            print(self.optimizer)

        # Distributed data parallel
        torch.cuda.set_device(device=self.context.gpu_no)
        self.model.cuda(device=self.context.gpu_no)

        self.model, self.optimizer = amp.initialize(
            models=self.model,
            optimizers=self.optimizer,
            opt_level=self.args.compute.opt_level)

        self.model = torch.nn.parallel.DistributedDataParallel(
            module=self.model,
            device_ids=[self.context.gpu_no])

        torch.backends.cudnn.benchmark = True

        # Set training and validation data loaders
        self.train_loader, self.val_loader, self.batch_fn = \
            load_data_loaders(dataset=args.data.dataset,
                              args=args,
                              context=context)

    def _train_loop(self, iteration, batch, epoch, batch_start_time,
                    train_objs, train_args):
        if train_args.use_relabel:
            # load ReLabel ground truth
            image, target_original, target_relabel = train_objs.batch_fn(
                batch=batch,
                num_classes=train_args.num_classes,
                mode='train')
            target_original = target_original.cuda()
            target = target_relabel
        else:
            # load original imagenet ground truth
            image, target_original = train_objs.batch_fn(batch=batch,
                                                         num_classes=train_args.num_classes,
                                                         mode='train')
            target_original = target_original.cuda()
            target = target_original

        batch_size = image.size(0)

        current_lr = adjust_learning_rate(
            optimizer=train_objs.optimizer,
            epoch=epoch,
            iteration=iteration,
            lr_decay_type=self.args.optim.lr.decay_type,
            epochs=train_args.epochs,
            train_len=train_args.len,
            warmup_lr=train_args.warmup_lr,
            warmup_epochs=train_args.warmup_epochs)

        image = image.cuda()

        # apply cutmix augmentation
        if self.args.data.cutmix.prob > 0. and self.args.data.cutmix.beta > 0.:
            cutmix_args = mch(
                beta=self.args.data.cutmix.beta,
                prob=self.args.data.cutmix.prob,
                num_classes=self.context.num_classes,
                smoothing=self.args.optim.label_smoothing,
                disable=epoch >= (self.args.optim.epochs
                                  - self.args.data.cutmix.off_epoch))
            image, target = cutmix_batch(image, target, cutmix_args)

        # forward and compute loss
        output = train_objs.model(image)
        loss = self.criterion(output, target)

        train_objs.optimizer.zero_grad()
        with amp.scale_loss(loss, train_objs.optimizer) as scaled_loss:
            scaled_loss.backward()

        # optimizer steps
        train_objs.optimizer.step()
        train_objs.optimizer.zero_grad()

        if iteration % self.args.util.print_freq != 0 or iteration == 0:
            return

        # print intermediate results
        target_squeezed = squeeze_one_hot(target_original)
        train_objs.meters = compute_accuracy_dist(output=output,
                                                  target_squeezed=target_squeezed,
                                                  meters=train_objs.meters,
                                                  world_size=self.context.world_size)

        reduced_loss = reduce_tensor(
            loss.data, self.context.world_size)
        train_objs.meters.get('losses').update(reduced_loss.item(), batch_size)

        torch.cuda.synchronize()
        train_objs.meters.get('batch_time').update(
            (time.time() - batch_start_time))

        if self.context.gpu_no != 0:
            return

        PrintCollection.print_train_batch_info(args=self.args,
                                               epoch=epoch,
                                               iteration=iteration,
                                               train_len=train_args.len,
                                               meters=train_objs.meters,
                                               current_lr=current_lr)

        sys.stdout.flush()

    def _validate_loop(self, batch, iteration, val_objs, val_args):
        start_time = time.time()

        # load validation image and ground truth labels
        image, target = val_objs.batch_fn(batch=batch,
                                          num_classes=val_args.num_classes,
                                          mode='val')

        image = image.cuda()
        target = target.cuda()
        target_squeezed = squeeze_one_hot(target)

        # forward pass and compute loss
        with torch.no_grad():
            output = val_objs.model(image)
            loss = self.criterion(output, target)

        val_objs.meters = compute_accuracy_dist(output=output,
                                                target_squeezed=target_squeezed,
                                                meters=val_objs.meters,
                                                world_size=self.context.world_size)

        reduced_loss = reduce_tensor(
            tensor=loss.data,
            world_size=self.context.world_size)

        val_objs.meters.get('losses').update(reduced_loss.item(), image.size(0))
        val_objs.meters.get('batch_time').update(time.time() - start_time)

        if (self.context.gpu_no != 0 or
                iteration % self.args.util.print_freq != 0 or
                iteration == 0):
            return

        # print intermediate results
        PrintCollection.print_val_batch_info(args=self.args,
                                             iteration=iteration,
                                             meters=val_objs.meters,
                                             val_len=val_args.len)

    def train(self, epoch):
        meters = AverageMeterCollection('batch_time', 'losses', 'acc1', 'acc5')
        train_objs = mch(
            loader=self.train_loader,
            model=self.model,
            optimizer=self.optimizer,
            batch_fn=self.batch_fn,
            meters=meters
        )
        train_args = mch(
            epochs=self.args.optim.epochs,
            warmup_lr=self.args.optim.warmup.lr,
            warmup_epochs=self.args.optim.warmup.epochs,
            num_classes=self.context.num_classes,
            len=self.context.train_len,
            use_relabel=self.args.data.relabel.use,
        )
        self._train(epoch, train_objs, train_args)

    def _train(self, epoch, train_objs, train_args):
        train_objs.model.train()
        tic = time.time()
        batch_start_time = time.time()
        train_objs.optimizer.zero_grad()
        for iteration, batch in enumerate(train_objs.loader):
            self._train_loop(batch=batch, iteration=iteration, epoch=epoch,
                             batch_start_time=batch_start_time,
                             train_objs=train_objs, train_args=train_args)
            batch_start_time = time.time()

        if self._get_condition_for_save_and_log():
            PrintCollection.print_train_time_cost(
                total_epochs=self.args.optim.epochs, epoch=epoch,
                time_spent=time.time() - tic)

    def validate(self, epoch=0):
        meters = AverageMeterCollection('batch_time', 'losses', 'acc1', 'acc5')
        val_objs = mch(
            loader=self.val_loader,
            model=self.model,
            batch_fn=self.batch_fn,
            meters=meters
        )
        val_args = mch(
            num_classes=self.context.num_classes,
            len=self.context.val_len,
        )

        accuracy = self._validate(val_objs, val_args)

        if not self.args.util.test_only:
            self.save_checkpoint(epoch=epoch, accuracy=accuracy)


    def _validate(self, val_objs, val_args):
        if not val_objs.loader:
            return 0, {}

        val_objs.model.eval()

        for iteration, batch in enumerate(val_objs.loader):
            self._validate_loop(batch=batch, iteration=iteration,
                                val_objs=val_objs, val_args=val_args)

        if self._get_condition_for_save_and_log():
            PrintCollection.print_top1_top5_accuracies(val_objs.meters)

        return val_objs.meters.get('acc1').avg

    def _get_condition_for_save_and_log(self):
        return not self.args.compute.distributed.use or \
               (self.args.compute.distributed.use and
                (self.context.rank % self.context.ngpus_per_node) == 0)

    def load_checkpoint(self, weight_file):
        if os.path.isfile(weight_file):
            print(f"=> loading checkpoint '{weight_file}'")
            checkpoint = torch.load(weight_file)

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
            self.model.load_state_dict(checkpoint)
            print(f"=> checkpoint loaded '{weight_file}'")
        else:
            raise Exception(f"=> no checkpoint found at '{weight_file}'")

    def save_checkpoint(self, epoch,
                        accuracy=None):
        if self._get_condition_for_save_and_log():
            save_dict = {
                'epoch': epoch + 1,
                'arch': self.args.model.arch,
                'state_dict': get_state_dict(self.model),
                'accuracy': accuracy,
                'optimizer': self.optimizer.state_dict(),
            }

            checkpoint_dir = 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            filepath = f"{checkpoint_dir}/checkpoint-{self.args.model.arch}-last.pth"
            torch.save(save_dict, filepath)

            if accuracy > self.best_acc1:
                self.best_acc1 = accuracy
                best_filepath = f"{checkpoint_dir}/checkpoint-{self.args.model.arch}-best.pth"
                shutil.copyfile(filepath, best_filepath)