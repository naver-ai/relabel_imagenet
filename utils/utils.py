import munch
import torch

DEFAULT_CONFIG_STR = '__DEFAULT__'

IMAGENET_MEAN_255 = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_MEAN_01 = [0.485, 0.456, 0.406]
IMAGENET_STD_255 = [0.229 * 255, 0.224 * 255, 0.225 * 255]
IMAGENET_STD_01 = [0.229, 0.224, 0.225]


def get_lighting():
    return Lighting(alphastd=0.1,
                    eigval=[0.2175, 0.0188, 0.0045],
                    eigvec=[[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]])


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('t', 'true', '1', 'y', 'yes'):
        return True
    if string.lower() in ('f', 'false', '0', 'n', 'no'):
        return False
    raise ValueError(f'method _str2bool cannot interpret the input: {string}')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def set_distributed_worker(gpu, args, context):
    if gpu >= 0:
        print(f'Use GPU: {gpu} for training')

    context.gpu_no = gpu

    if args.compute.distributed.use:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        context.rank = context.rank * context.ngpus_per_node + gpu

    torch.distributed.init_process_group(
        backend=args.compute.distributed.backend,
        init_method=args.compute.distributed.url,
        world_size=context.world_size,
        rank=context.rank)


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    accuracies = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        accuracies.append(correct_k.mul_(100.0 / batch_size))
    return accuracies


def compute_accuracy_dist(output, target_squeezed, meters, world_size):
    output_all = concat_diffent_size_distributed_tensors(
        tensor=output.data,
        world_size=world_size,
        dim=0)
    target_all = concat_diffent_size_distributed_tensors(
        tensor=target_squeezed,
        world_size=world_size,
        dim=0)
    acc1, acc5 = compute_accuracy(output=output_all,
                          target=target_all,
                          topk=(1, 5))
    meters.acc1.update(acc1.item(), output_all.size(0))
    meters.acc5.update(acc5.item(), output_all.size(0))
    return meters


def concat_diffent_size_distributed_tensors(tensor, world_size, dim=0):
    if dim > 0:
        tensor.transponse(0, dim)
    rt = tensor.clone()

    size_ten = torch.IntTensor([rt.shape[0]]).to(rt.device)
    gather_size = [torch.ones_like(size_ten) for _ in range(world_size)]
    torch.distributed.all_gather(tensor=size_ten, tensor_list=gather_size)
    max_size = torch.cat(gather_size, dim=0).max()

    padded = torch.empty(max_size, *rt.shape[1:],
                         dtype=rt.dtype,
                         device=rt.device)
    padded[:rt.shape[0]] = rt
    gather_t = [torch.ones_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(tensor=padded, tensor_list=gather_t)

    slices = []
    for i, sz in enumerate(gather_size):
        slices.append(gather_t[i][:sz.item()])

    concat_t = torch.cat(slices, dim=0)
    if dim > 0:
        concat_t.transponse(0, dim)
    return concat_t


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterCollection(object):
    def __init__(self, *meter_names):
        for name in meter_names:
            setattr(self, name, AverageMeter())

    def get(self, meter_name):
        return getattr(self, meter_name)


class PrintCollection(object):
    @staticmethod
    def _print_batch_time(meters):
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            batch_time=meters.batch_time), end='\t')

    @staticmethod
    def _print_speed(args, meters, mode='train'):
        if mode == 'train':
            batch_size = args.optim.batch_size
        elif mode == 'val':
            batch_size = args.util.val_batch_size
        else:
            raise ValueError('Wrong mode: {}'.format(mode))

        print('Speed {0:.3f} ({1:.3f})'.format(
            batch_size / meters.batch_time.val,
            batch_size / meters.batch_time.avg
        ), end='\t')

    @staticmethod
    def _print_loss_acc1_acc5(meters):
        print('Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=meters.losses),
              end='\t')
        print('Prec@1 {acc1.val:.3f} ({acc1.avg:.3f})'.format(acc1=meters.acc1),
              end='\t')
        print('Prec@5 {acc5.val:.3f} ({acc5.avg:.3f})'.format(acc5=meters.acc5),
              end='\t')

    @staticmethod
    def print_val_batch_info(args, iteration, meters, val_len):
        print('Test: [{0}/{1}]'.format(iteration, val_len), end='\t')
        PrintCollection._print_batch_time(meters)
        PrintCollection._print_speed(args, meters, mode='val')
        PrintCollection._print_loss_acc1_acc5(meters)
        print()

    @staticmethod
    def print_train_batch_info(args, iteration, meters, train_len, current_lr,
                               epoch=None):
        if epoch is not None:
            print('Epoch: [{0}][{1}/{2}]'
                  .format(epoch, iteration, train_len),
                  end='\t')
        else:
            print('Iter: [{0}/{1}]'
                  .format(iteration, train_len),
                  end='\t')
        PrintCollection._print_batch_time(meters)
        PrintCollection._print_speed(args, meters, mode='train')
        print('LR {0:.2E}'.format(current_lr), end='\t')
        PrintCollection._print_loss_acc1_acc5(meters)
        print()

    @staticmethod
    def print_train_time_cost(total_epochs, epoch, time_spent):
        print('[Epoch {}] {:.3f} sec/epoch'
              .format(epoch, time_spent), end='\t')
        print('remaining time: {:.3f} hours'.format(
            (total_epochs - epoch - 1) * time_spent / 3600))

    @staticmethod
    def print_top1_top5_accuracies(meters):
        print(' * Prec@1 {acc1.avg:.3f} Prec@5 {acc5.avg:.3f}'
              .format(acc1=meters.acc1, acc5=meters.acc5))


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_state_dict(model):
    if hasattr(model, 'module'):
        return model.module.state_dict()
    else:
        return model.state_dict()


def set_dataset_len(batch_size, val_batch_size, ngpus_per_node):
    num_train_samples = 1281167
    num_val_samples = 50000
    train_len = ((num_train_samples - 1) // (batch_size * ngpus_per_node)) + 1
    val_len = ((num_val_samples - 1) // (val_batch_size * ngpus_per_node)) + 1
    return train_len, val_len

def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary