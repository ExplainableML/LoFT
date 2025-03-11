# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

# import os
# import torch
# 
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import os
import sys
import json
import time
import math
import random
import datetime
import traceback
from pathlib import Path
from os.path import join as ospj

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LinearLR
from torchvision.transforms import v2
from timm.layers import convert_sync_batchnorm

import utils
from models import ResNet50, CLIP

from config import get_args
from data import get_data_loader, get_synth_train_data_loader


def load_data_loader(args):
    val_loader, test_loader = get_data_loader(
        real_test_data_dir=args.real_test_data_dir,
        metadata_dir=args.metadata_dir,
        dataset=args.dataset, 
        eval_bs=args.batch_size_eval,
        model_type=args.model_type,
        is_validation=True,
    )
    return val_loader, test_loader


def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        bs=args.batch_size_per_gpu,
        n_img_per_cls=args.n_img_per_cls,
        dataset=args.dataset,
        model_type=args.model_type,
        is_distributed=args.is_distributed,
    )
    return synth_train_loader


def train_one_epoch(
    args,
    model, clf_loss, optimizer, epoch, fp16_scaler, cutmix_or_mixup,
    train_loader, val_loader, test_loader,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, batch in enumerate(
        metric_logger.log_every(train_loader, 1000, header)
    ):
        image, label = batch
        label_origin = label
        label_origin = label_origin.cuda(non_blocking=True)

        # apply CutMix and MixUp augmentation
        # or change label format for Binary CE in ViT training
        if args.is_mix_aug:
            p = random.random()
            p_threshold = 0.8
            if p >= p_threshold:
                image, label = cutmix_or_mixup(image, label)

        it = len(train_loader) * epoch + it  # global training iteration

        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        
        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        # forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logit = model(image)
            loss = clf_loss(logit, label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # parameter update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.is_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.is_grad_clip:
                fp16_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        with torch.no_grad():
            acc1, _ = utils.accuracy(logit.detach(), label_origin, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])


        # Evaluation
        if (it + 1) % args.eval_iteration == 0 or (it + 1) == (len(train_loader) * args.epochs):
            # ============ evaluate model ... ============
            val_stats = eval(
                model, clf_loss, val_loader, epoch, it + 1, fp16_scaler, args, phase='val')

            # ============ saving logs and model checkpoint ... ============

            test_stats = None
            if val_stats["val/top1"] > args.best_top1:
                args.best_top1 = val_stats["val/top1"]
                args.best_stats = val_stats
                if args.save_model:
                    save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")
                # evaluation on test dataset
                test_stats = eval(
                    model, clf_loss, test_loader, epoch, it + 1, fp16_scaler, args, phase='test')


            model.train()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    train_stats = {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}
    train_stats.update({"epoch": epoch})
    train_stats.update({"iter": it})


@torch.no_grad()
def eval(model, clf_loss, data_loader, epoch, n_iteration, fp16_scaler, args, phase):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "[Iteration: {}]".format(n_iteration)

    model.eval()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 1000, header)
    ):

        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(image, phase="eval")
            loss = clf_loss(output, label)

        acc1, _ = utils.accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())

    metric_logger.synchronize_between_processes()
    print(f"Averaged {phase} stats:", metric_logger)

    stat_dict = {
        "{}/{}".format(phase, k): meter.global_avg 
        for k, meter in metric_logger.meters.items()
    }

    return stat_dict


def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    if args.is_distributed:
        if dist.get_rank() == 0:
            pass
        else:
            return

    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))


def main(args):
    utils.fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    val_loader, test_loader = load_data_loader(args)
    train_loader = load_synth_train_data_loader(args)
    print("TRAIN DATASET SIZE:", len(train_loader.dataset))
    print("Loader length:", len(train_loader))
        
    # ==================================================
    # Model and optimizer
    # ==================================================
    if args.model_type == "clip":
        model = CLIP(
            dataset=args.dataset,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            download_root=args.clip_download_root,
            clip_version=args.clip_version,
            is_image_full_finetuning=args.is_image_full_finetuning,
        )
        params_groups = model.learnable_params()
    elif args.model_type == "resnet50": 
        model = ResNet50(n_classes=args.n_classes)
        params_groups = model.parameters()
    
    model = model.cuda()
    if args.is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
        )
    clf_loss = nn.CrossEntropyLoss().cuda()

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    if args.model_type == "resnet50":
        optimizer = torch.optim.SGD(
            params_groups, lr=args.lr, weight_decay=args.wd, momentum=0.9,
        )
    elif args.model_type == "clip":
        optimizer = torch.optim.AdamW(
            params_groups, lr=args.lr, weight_decay=args.wd,
        )
    args.lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.min_lr,
    )

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "best_checkpoint.pth"),
        run_variables=to_restore,
        is_distributed=args.is_distributed,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    # ==================================================
    # Training
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    args.best_stats = {}
    args.best_top1 = 0.

    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch ... ============
        train_one_epoch(
            args,
            model, clf_loss, optimizer, epoch, fp16_scaler, cutmix_or_mixup,
            train_loader, val_loader, test_loader,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(traceback.format_exc())
