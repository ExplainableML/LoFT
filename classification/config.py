
import random
import os
import yaml
import json
import argparse
from os.path import join as ospj
import torch.distributed as dist

from logger import Logger
from utils import (
    bool_flag,
    init_distributed_mode,
)
from util_data import SUBSET_NAMES

_MODEL_TYPE = ("resnet50", "clip")

def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return v

def int2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return int(v)

def float2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return float(v)

def float_or_list_of_float2none(v):
    if v is None or v == "":
        return None
    elif v.lower() in ('none', 'null'):
        return None
    else:
        try:
            return float(v)
        except:
            return [float(_v) for _v in v.split(",")]

def list_int2none(vs):
    return_vs = []
    for v in vs:
        if v is None:
            pass
        elif v.lower() in ('none', 'null'):
            v = None
        else:
            v = int(v)
        return_vs.append(v)
    return return_vs


def set_log(output_dir):
    log_file_name = ospj(output_dir, 'log.log')
    Logger(log_file_name)


def set_data_dir(args):
    # local to args
    yaml_file = "local.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    args.output_dir = args_local["output_dir"]
    args.synth_train_data_dir = args_local["synth_train_data_dir"]
    args.real_test_data_dir = args_local["real_test_data_dir"]
    args.clip_download_root = args_local["clip_download_root"]
    args.metadata_dir = args_local["metadata_dir"]

    # set output directory
    mid0 = args.method
    if args.method == "loft":
        liw = (
            args.loft_interpolation_weight 
            if isinstance(args.loft_interpolation_weight, (float, int)) 
            else "-".join([str(_l) for _l in args.loft_interpolation_weight])
        )
        mid0 += f"_{liw}"
    mixaug = "_mixuag" if args.is_mix_aug else ""
    mid1 = f"lr{args.lr}_wd{args.wd}{mixaug}"
    args.output_dir = ospj(
        args.output_dir, 
        args.dataset, 
        mid0,
        f"shot{args.n_shot}_{args.fewshot_seed}", 
        f"n_img_per_cls_{args.n_img_per_cls}", 
        mid1,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # set synthetic training data directory
    mid0 = args.method
    if args.method == "loft":
        liw = (
            args.loft_interpolation_weight 
            if isinstance(args.loft_interpolation_weight, (float, int)) 
            else "-".join([str(_l) for _l in args.loft_interpolation_weight])
        )
        mid0 += f"_{liw}"
    args.synth_train_data_dir = ospj(
        args.synth_train_data_dir,
        args.dataset,
        mid0,
        f"shot{args.n_shot}_{args.fewshot_seed}", 
    )

    # set real test data directory
    args.real_test_data_dir = ospj(args.real_test_data_dir, args.dataset)

    return args


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Method
    parser.add_argument("--method", type=str, default="loft")
    parser.add_argument("--loft_interpolation_weight", 
                        type=float_or_list_of_float2none, default=0.5)

    # Data
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument("--n_img_per_cls", type=int2none, default=500)
    parser.add_argument("--is_mix_aug", type=str2bool, default=False,
                        help="use mixup and cutmix")
    parser.add_argument("--n_shot", type=int2none, default=0)
    parser.add_argument("--fewshot_seed", type=str2none, default="seed0",
                        help="seed{number}.")

    # Training/Optimization parameters
    parser.add_argument('--model_type', type=str2none, default=None,
                        choices=_MODEL_TYPE)
    parser.add_argument("--use_fp16", type=bool_flag, default=True)
    parser.add_argument("--batch_size_per_gpu", default=64, type=int)
    parser.add_argument("--batch_size_eval", default=1024, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--wd", type=float2none, default=1e-4)
    parser.add_argument("--lr", default=1e-4, type=float2none)
    parser.add_argument("--warmup_epochs", default=6, type=int)
    parser.add_argument("--min_lr", type=float, default=1e-8)

    # CLIP setting
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')
    parser.add_argument("--is_lora_image", type=str2bool, default=True)
    parser.add_argument("--is_lora_text", type=str2bool, default=True)
    parser.add_argument("--is_image_full_finetuning", type=str2bool, default=False,
                        help="Applies on CLIP ViT models.")
    parser.add_argument("--is_grad_clip", type=str2bool, default=False)

    # Util
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--eval_iteration", type=int2none, default=312, 
                        help="1281167 / 4096 = 312")
    parser.add_argument("--is_set_log", type=str2bool, default=True)
    parser.add_argument("--seed", default=22, type=int)


    # Distributed training
    parser.add_argument("--is_distributed", type=str2bool, default=False)
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="Url used to set up distributed training.")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore this argument; No need to set it manually.")

    args = parser.parse_args()

    args.n_classes = len(SUBSET_NAMES[args.dataset])
    args = set_data_dir(args)
    if args.is_distributed:
        init_distributed_mode(args)
    if args.is_set_log:
        set_log(args.output_dir)

    return args


