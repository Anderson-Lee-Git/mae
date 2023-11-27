# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import build_dataset
from util.latent import save_latent

import models_mae
from models_mae import build_model

from engine_pretrain import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('MAE latent generating', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    # parser.add_argument('--mask_ratio', default=0.75, type=float,
    #                     help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='dataset option')
    parser.add_argument('--data_group', default=1, type=int, help='which data group')
    
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--data_subset', default=1.0, type=float,
                        help='subset of data to use')

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # misc
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--project_name', default='', type=str,
                        help='wandb project name')
    
    # inference model
    parser.add_argument('--ckpt', default='', type=str,
                        help="checkpoint to use for inference")
    
    # latent storage
    parser.add_argument('--latent_output_dir', default='./latent_output_dir',
                        type=str, help="output directory for latent space")
    return parser

def adjust_args(args):
    if args.dataset == "tiny_imagenet":
        args.input_size = 64
    else:
        args.input_size = 224

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.use_wandb:
        wandb.init(project=args.project_name)
        # args.__dict__.update(wandb.config.__dict__)
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)
        args.log_dir = os.path.join(args.log_dir, wandb.run.name)
    if args.distributed:
        misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # adjust args based on options
    adjust_args(args)

    dataset_train = build_dataset(is_train=True, args=args, include_path=True)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    if args.model in models_mae.__dict__:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=args.input_size)
    else:
        model = build_model(patch_size=args.patch_size, embed_dim=args.embed_dim,
                            num_heads=args.num_heads, decoder_embed_dim=args.decoder_embed_dim,
                            norm_pix_loss=args.norm_pix_loss, img_size=args.input_size)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    # load model from ckpt
    print(f"Load model from {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    model_without_ddp.load_state_dict(state_dict["model"])
    
    # inference
    with torch.no_grad():
        for samples, _, paths in data_loader:
            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                latent, _, _ = model.forward_encoder(samples, 0.0)
                latent = latent[:, 1:, :]
                save_latent(latent, paths)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    if args.use_wandb:
        wandb.finish()
