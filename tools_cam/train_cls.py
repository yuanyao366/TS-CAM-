# ----------------------------------------------------------------------------------------------------------
# TS-CAM
# Copyright (c) Learning and Machine Perception Lab (LAMP), SECE, University of Chinese Academy of Science.
# ----------------------------------------------------------------------------------------------------------
import os
import sys
import datetime
import pprint

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
import numpy as np

def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    print(model)
    optimizer = create_optimizer(args, model)

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, cls_criterion


def main():
    args = update_config()

    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_BS{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.TRAIN.BATCH_SIZE), cfg.BASIC.TIME)
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = creat_model(cfg, args)

    best_top1_acc = 0
    best_top1_acc_epoch = 0
    update_train_step = 0
    update_val_step = 0
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        adjust_learning_rate_normal(optimizer, epoch, cfg)
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion,
                            optimizer, epoch, writer, cfg, update_train_step)

        update_val_step, loss_val, cls_top1_val, cls_top5_val = \
            val_loc_one_epoch(val_loader, model, device, cls_criterion, epoch, writer, cfg, update_val_step)

        if cfg.SOLVER.NUM_EPOCHS > 20:
            save_epoch = 10
        else:
            save_epoch = 5
        if epoch % save_epoch == 0:
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch)))

        if cls_top1_val > best_top1_acc:
            best_top1_acc = cls_top1_val
            best_top1_acc_epoch = epoch
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(ckpt_dir, 'model_best_top1_acc.pth'))

        print("Best TOP1_ACC: {} (epoch:{})".format(best_top1_acc, best_top1_acc_epoch))
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def train_one_epoch(train_loader, model, device, criterion, optimizer, epoch,
                    writer, cfg, update_train_step):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, (input, target, one_hot_target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)

        cls_logits = model(input)
        loss = criterion(cls_logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(cls_logits.data.contiguous(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top1', prec1.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top5', prec5.item(), update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.8f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), loss=losses,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    return update_train_step, losses.avg, top1.avg, top5.avg


def val_loc_one_epoch(val_loader, model, device, criterion,epoch, writer, cfg, update_val_step):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        model.eval()
        for i, (input, target, bbox, image_names, one_hot_target) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)

            cls_logits = model(input)
            loss = criterion(cls_logits, target)

            prec1, prec5 = accuracy(cls_logits.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            writer.add_scalar('loss_iter/val', loss.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top1', prec1.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top5', prec5.item(), update_val_step)

            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(val_loader) - 1:
                print(('Val Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i + 1, len(val_loader), loss=losses, top1=top1, top5=top5)))

    return update_val_step, losses.avg, top1.avg, top5.avg
if __name__ == "__main__":
    main()
