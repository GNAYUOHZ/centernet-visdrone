from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from src.opts import opt
from src.dataset import Dataset
from src.losses import CtdetLoss
from src.utils.logger import Logger
from src.utils.average_meter import AverageMeter, TimeMeter
from src.model import get_model, load_model, save_model


def train(model, train_loader, criterion, optimizer, logger, opt, epoch, scaler, time_stats):
    model.train()
    avg_loss_stats = {l: AverageMeter()
                      for l in ['loss', 'hm_loss', 'wh_loss', 'off_loss']}

    for iter_id, batch in enumerate(train_loader):
        # to cuda
        for k in batch:
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        # amp
        with autocast():
            output = model(batch['input'])
            loss_stats = criterion(output, batch)
            loss = loss_stats['loss'].mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # else:
        #     # no amp
        #     output = model(batch['input'])
        #     loss_stats = criterion(output, batch)
        #     loss = loss_stats['loss'].mean()

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        info = f'train : [{epoch}][{iter_id}/{len(train_loader)}] |'
        for l in avg_loss_stats:
            avg_loss_stats[l].update(
                loss_stats[l].mean().item(), batch['input'].size(0))
            info += f'|{l} {avg_loss_stats[l].avg:.4f} '

        time_stats.update(epoch, iter_id)
        info += f'|left_time: {time_stats.left_time:.1f} hour'

        # log
        if iter_id % 100 == 0:
            logger.write(info)


def val(model, val_loader, criterion, logger, opt, epoch):
    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        avg_loss_stats = {l: AverageMeter()
                          for l in ['loss', 'hm_loss', 'wh_loss', 'off_loss']}

        for iter_id, batch in enumerate(val_loader):
            for k in batch:
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output = model(batch['input'])
            loss_stats = criterion(output, batch)

            info = f'val   : [{epoch}][{iter_id}/{len(val_loader)}] |'
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                info += f'|{l} {avg_loss_stats[l].avg:.4f} '

            # log
            if iter_id % 100 == 0:
                logger.write(info)


def main():
    torch.manual_seed(317)
    torch.backends.cudnn.benckmark = True

    train_logger = Logger(opt, "train")
    val_logger = Logger(opt, "val")

    start_epoch = 0
    print('Creating model...')
    model = get_model(opt.arch, opt.heads).to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    criterion = CtdetLoss(opt)

    print('Loading model...')
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.lr, opt.lr_step)
    model = torch.nn.DataParallel(model)

    # amp
    scaler = GradScaler()

    print('Setting up data...')
    train_dataset = Dataset(opt, 'train')
    val_dataset = Dataset(opt, 'val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    # cal left time
    time_stats = TimeMeter(opt.num_epochs, len(train_loader))

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('train...')
        train(model, train_loader, criterion, optimizer,
              train_logger, opt, epoch, scaler, time_stats)

        if epoch % opt.val_intervals == 0:
            print('val...')
            val(model, val_loader, criterion, val_logger, opt, epoch)
            save_model(os.path.join(opt.save_dir, f'model_{epoch}.pth'),
                       epoch, model, optimizer)

        # update learning rate
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # without optimizer
    save_model(os.path.join(opt.save_dir, 'model_final.pth'), epoch, model)


if __name__ == '__main__':
    main()
