# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import os

import torch
import wandb

import util.misc as misc
import util.lr_sched as lr_sched
from util.datasets import visualize_image


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, _, _, reconstructed_images = model(samples, mask_ratio=args.mask_ratio)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # visualize the last batch randomly
    p = torch.rand(1)
    if p < 0.02 or epoch % 10 == 0:
        if not os.path.exists(os.path.join(args.log_dir, "examples")):
            os.makedirs(os.path.join(args.log_dir, "examples"))
        visualize_image(samples[0], os.path.join(args.log_dir, "examples/sample_image.png"))
        visualize_image(reconstructed_images[0], os.path.join(args.log_dir, "examples/sample_reconstruction.png"))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            loss, _, _, reconstructed_images = model(samples, mask_ratio=0.0)
            print(loss)
        
        loss_value = loss.item()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        # sample a random image/reconstruction pair in each batch to visualize
        idx = torch.randint(low=0, high=len(samples), size=(1,)).item()
        if not os.path.exists(os.path.join(args.log_dir, "examples")):
            os.makedirs(os.path.join(args.log_dir, "examples"))
        visualize_image(samples[idx], os.path.join(args.log_dir, f"examples/sample_image_{data_iter_step}.png"))
        visualize_image(reconstructed_images[idx], os.path.join(args.log_dir, f"examples/sample_reconstruction_{data_iter_step}.png"))
        if args.use_wandb:
            wandb.log({
                f"sample_image_{data_iter_step}": wandb.Image(os.path.join(args.log_dir, f"examples/sample_image_{data_iter_step}.png")),
                f"sample_reconstruction_{data_iter_step}": wandb.Image(os.path.join(args.log_dir, f"examples/sample_reconstruction_{data_iter_step}.png"))
            })
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}