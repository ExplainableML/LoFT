# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

##################################################
# Utility routines (functions and classes) used for training models.
# Some of these routines are re-used from
# - DINO (https://github.com/facebookresearch/dino)
# - MoCo (https://github.com/facebookresearch/moco)
# - PyTorch examples (https://github.com/pytorch/examples)
##################################################

import argparse
import datetime
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from collections import OrderedDict
import tarfile
from os.path import join as ospj

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


class UnNormalize(object):
    def __init__(self, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
#                  mean=(0.48145466, 0.4578275, 0.40821073), 
#                  std=(0.26862954, 0.26130258, 0.27577711)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormed_tensor = torch.zeros_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            unnormed_tensor[i] = t.mul(s).add(m)
            # The normalize code -> t.sub_(m).div_(s)
        return unnormed_tensor

unnorm = UnNormalize()


def make_dirs(fpath_dir):
#     if "." not in os.path.basename(fpath):
#     fpath_dir = fpath
#     else:
#         fpath_dir = os.path.dirname(fpath)
    if not os.path.exists(fpath_dir):
        os.makedirs(fpath_dir)

def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_mode(args):
#     if 'SLURM_PROCID' in os.environ:
#         # DDP via SLURM
#         args.local_rank, args.rank, args.world_size = world_info_from_env()
#         # SLURM var -> torch.distributed vars in case needed
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#         os.environ['RANK'] = str(args.rank)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#         torch.distributed.init_process_group(
#             backend="nccl",
#             init_method=args.dist_url,
#             world_size=args.world_size,
#             rank=args.rank,
#         )
#     else:
#         # DDP via torchrun, torch.distributed.launch
#         args.local_rank, _, _ = world_info_from_env()
#         torch.distributed.init_process_group(
#             backend="nccl",
#             init_method=args.dist_url
#         )
#         args.world_size = torch.distributed.get_world_size()
#         args.rank = torch.distributed.get_rank()
# 
#     if torch.cuda.is_available():
#         if args.is_distributed:
#             device = 'cuda:%d' % args.local_rank
#         else:
#             device = 'cuda:0'
#         torch.cuda.set_device(device)
#     else:
#         device = 'cpu'
#     args.device = device
#     device = torch.device(device)
#     return device
    
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
#     # launched naively with "python main.py"
#     # we manually add MASTER_ADDR and MASTER_PORT to env variables
#     elif torch.cuda.is_available():
#         print("==> Will run the code on one GPU.")
# #         args.rank, args.gpu, args.world_size = 0, 2, 1
#         args.rank, args.world_size = 0, 1
#         os.environ["MASTER_ADDR"] = "127.0.0.1"
#         os.environ["MASTER_PORT"] = str(12345 + gpu)
    else:
        print("==> Run with torchrun, otherwise it does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
#         backend="gloo",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
#     torch.cuda.set_device(gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def fix_random_seeds(seed=22):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_program_info(args, save_path=None):
    """
    Prints argparse arguments, and saves them into a file if save_path is provided.
    """
    if dist.get_rank() != 0:
        return

    fid = None
    if save_path is not None:
        fid = open(save_path, "w")

    def _print(text):
        print(text)
        if fid is not None:
            print(text, file=fid)

    _print("")
    _print("-" * 100)
    _print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    _print("-" * 100)
    _print(sys.argv[0])
    for parg in sys.argv[1:]:
        _print("\t{}".format(parg))
    _print("-" * 100)
    _print("")

    if fid is not None:
        fid.flush()
        fid.close()


def get_params_groups(model):
    """
    Returns two parameters group, one for regularized parameters with weight decay,
    and another for unregularized parameters.
    Unregularized parameters include: Bias terms and batch-norm parameters.
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    Creates a cosine scheduler with linear warm-up.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def restart_from_checkpoint(
    ckp_path, run_variables=None, is_distributed=False, **kwargs
):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # if module named in the model, remove it is_update_ckpt = False
    is_update_ckpt = False
    if not is_distributed:
        for k, v in checkpoint["model"].items():
            if k.startswith("module"):
                is_update_ckpt = True
            break
    if is_update_ckpt:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        checkpoint["model"] = new_state_dict

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def get_record(fpath):
    if not os.path.isfile(fpath):
        return None
    else:
        with open(fpath, "r") as f:
            record = json.load(f)
        return record


def write_record(fpath, **kwargs):
    if not os.path.isfile(fpath):
        record = {}
    else:
        with open(fpath, "r") as f:
            record = json.load(f)

    for k, v in kwargs.items():
        record[k] = v

    with open(fpath, "w") as f:
        json.dump(record, f, indent=2)


def save_logs(log_stats: dict, tb_writer: SummaryWriter, epoch, save_path: str):
    log_stats.update({"epoch": epoch})

    if dist.get_rank() == 0:
        with open(save_path, mode="a") as f:
            f.write(json.dumps(log_stats) + "\n")

    for k, v in log_stats.items():
        tb_writer.add_scalar(k, v, epoch)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def accuracy_per_class(data_loader, model, num_classes=100, mode="trex", fp16_scaler=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    apc = [MetricLogger(delimiter="  ") for _ in range(num_classes)]
    model.eval()
    topk = 1
    for it, (image, label) in enumerate(data_loader):
#         image = image.cuda(non_blocking=True)
        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True) \
                if mode in ("latent", "latent_from_image") else image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(image, phase="eval")
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))
        for cl in label.unique():
            idx = (label == cl)
            cl_correct = correct[:, idx]
            cls_topk = cl_correct[:topk].reshape(-1).float().sum(0) * 100.0 / cl_correct.size(1)
            apc[cl].update(top1=cls_topk.item())
    for i, cl in enumerate(apc):
        apc[i].synchronize_between_processes()
        print(i)
        print([meter.global_avg for k, meter in apc[i].meters.items()])
    return


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
