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
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal, DistillationLoss
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
    if args.extra_token_blk_idx >= 0:
        model = create_deit_model(
                cfg.MODEL.ARCH,
                pretrained=True,
                num_classes=cfg.DATA.NUM_CLASSES,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                extra_token_block_idx = args.extra_token_blk_idx,
            )
    else:
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
    cls_criterion_pt = torch.nn.CrossEntropyLoss().to(device)
    if cfg.SOLVER.LOSS == 'CE':
        cls_criterion_ct = torch.nn.CrossEntropyLoss().to(device)
    elif cfg.SOLVER.LOSS == 'BCE':
        cls_criterion_ct = torch.nn.BCEWithLogitsLoss().to(device)
    elif cfg.SOLVER.LOSS == 'SoftDist':
        cls_criterion_ct = DistillationLoss('soft', args.distillation_tau).to(device)
    elif cfg.SOLVER.LOSS == 'HardDist':
        cls_criterion_ct = DistillationLoss('hard', args.distillation_tau).to(device)
    else:
        assert NotImplementedError
    print(cls_criterion_pt)
    print(cls_criterion_ct)
    print('Preparing networks done!')
    return device, model, optimizer, cls_criterion_pt, cls_criterion_ct


def main():
    args = update_config()

    # create checkpoint directory
    datetime_dir = 'Logit{}_Loss{}{}_{}'.format(cfg.TEST.CLS_LOGITS, cfg.SOLVER.LOSS, cfg.SOLVER.LOSS_SCALE, cfg.BASIC.TIME)
    if args.extra_token_blk_idx >= 0:
        datetime_dir = 'ExtraBlk{}_Logit{}_Loss{}_{}'.format(args.extra_token_blk_idx, cfg.TEST.CLS_LOGITS, cfg.SOLVER.LOSS, cfg.BASIC.TIME)
    cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_{}-CAM-THR{}_BS{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_MERGE, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE), datetime_dir)
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
    device, model, optimizer, cls_criterion_pt, cls_criterion_ct = creat_model(cfg, args)

    best_gtknown = 0
    best_gtknown_epoch = 0
    best_top1_loc = 0
    best_top1_loc_epoch = 0
    update_train_step = 0
    update_val_step = 0
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        adjust_learning_rate_normal(optimizer, epoch, cfg)
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion_pt, cls_criterion_ct,
                            optimizer, epoch, writer, cfg, update_train_step)

        update_val_step, loss_val, cls_top1_val, cls_top5_val, \
        loc_top1_val, loc_top5_val, loc_gt_known = \
            val_loc_one_epoch(val_loader, model, device, cls_criterion_pt, cls_criterion_ct, epoch, writer, cfg, update_val_step)

        if cfg.SOLVER.NUM_EPOCHS > 20:
            save_epoch = 10
        else:
            save_epoch = 5
        if epoch % save_epoch == 0:
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch)))

        if loc_top1_val > best_top1_loc:
            best_top1_loc = loc_top1_val
            best_top1_loc_epoch = epoch
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, 'model_best_top1_loc.pth'))
        if loc_gt_known > best_gtknown:
            best_gtknown = loc_gt_known
            best_gtknown_epoch = epoch
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, 'model_best.pth'))

        print("Best GT_LOC: {} (epoch:{})".format(best_gtknown, best_gtknown_epoch))
        print("Best TOP1_LOC: {} (epoch:{})".format(best_top1_loc, best_top1_loc_epoch))
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def train_one_epoch(train_loader, model, device, criterion_pt, criterion_ct, optimizer, epoch,
                    writer, cfg, update_train_step):
    losses = AverageMeter()
    losses_pt = AverageMeter()
    losses_ct = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_pt = AverageMeter()
    top5_pt = AverageMeter()
    top1_ct = AverageMeter()
    top5_ct = AverageMeter()

    model.train()
    for i, (input, target, one_hot_target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)
        one_hot_target = one_hot_target.to(device)

        cls_logits_pt, cls_logits_ct = model(input, return_cam=False)
        loss_pt = criterion_pt(cls_logits_pt, target)
        if cfg.SOLVER.LOSS == 'CE':
            loss_ct = criterion_ct(cls_logits_ct, target)
        elif cfg.SOLVER.LOSS == 'BCE':
            loss_ct = criterion_ct(cls_logits_ct, one_hot_target)
        elif cfg.SOLVER.LOSS == 'SoftDist' or cfg.SOLVER.LOSS == 'HardDist':
            loss_ct = criterion_ct(cls_logits_ct, cls_logits_pt.detach())
        else:
            assert NotImplementedError
        loss = loss_pt + loss_ct * cfg.SOLVER.LOSS_SCALE
        # loss = (loss_pt + loss_ct * cfg.SOLVER.LOSS_SCALE) / 2
        cls_logits = (cls_logits_pt + cls_logits_ct) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(cls_logits.data.contiguous(), target, topk=(1, 5))
        prec1_pt, prec5_pt = accuracy(cls_logits_pt.data.contiguous(), target, topk=(1, 5))
        prec1_ct, prec5_ct = accuracy(cls_logits_ct.data.contiguous(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_pt.update(loss_pt.item(), input.size(0))
        losses_ct.update(loss_ct.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        top1_pt.update(prec1_pt.item(), input.size(0))
        top5_pt.update(prec5_pt.item(), input.size(0))
        top1_ct.update(prec1_ct.item(), input.size(0))
        top5_ct.update(prec5_ct.item(), input.size(0))
        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('loss_pt_iter/train', loss_pt.item(), update_train_step)
        writer.add_scalar('loss_ct_iter/train', loss_ct.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top1', prec1.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top5', prec5.item(), update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader)-1:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Loss_pt {loss_pt.val:.4f} ({loss_pt.avg:.4f})\t'
                   'Loss_ct {loss_ct.val:.4f} ({loss_ct.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                   'Prec@1_pt {top1_pt.val:.3f} ({top1_pt.avg:.3f})\t'
                   'Prec@5_pt {top5_pt.val:.3f} ({top5_pt.avg:.3f})\t'
                   'Prec@1_ct {top1_ct.val:.3f} ({top1_ct.avg:.3f})\t'
                   'Prec@5_ct {top5_ct.val:.3f} ({top5_ct.avg:.3f})\t'.format(
                epoch, i + 1, len(train_loader), loss=losses, loss_pt=losses_pt, loss_ct=losses_ct,
                top1=top1, top5=top5, top1_pt=top1_pt, top5_pt=top5_pt, top1_ct=top1_ct, top5_ct=top5_ct,
                lr=optimizer.param_groups[-1]['lr'])))
    return update_train_step, losses.avg, top1.avg, top5.avg


def val_loc_one_epoch(val_loader, model, device, criterion_pt, criterion_ct, epoch, writer, cfg, update_val_step):

    losses = AverageMeter()
    losses_pt = AverageMeter()
    losses_ct = AverageMeter()

    cls_top1 = []
    cls_top5 = []
    loc_top1 = []
    loc_top5 = []
    loc_gt_known = []
    top1_loc_right = []
    top1_loc_cls = []
    top1_loc_mins = []
    top1_loc_part = []
    top1_loc_more = []
    top1_loc_wrong = []

    with torch.no_grad():
        model.eval()
        for i, (input, target, bbox, image_names, one_hot_target) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)
            one_hot_target = one_hot_target.to(device)

            cls_logits_pt, cls_logits_ct, cams = model(input, return_cam=True)
            loss_pt = criterion_pt(cls_logits_pt, target)
            if cfg.SOLVER.LOSS == 'CE':
                loss_ct = criterion_ct(cls_logits_ct, target)
            elif cfg.SOLVER.LOSS == 'BCE':
                loss_ct = criterion_ct(cls_logits_ct, one_hot_target)
            elif cfg.SOLVER.LOSS == 'SoftDist' or cfg.SOLVER.LOSS == 'HardDist':
                loss_ct = criterion_ct(cls_logits_ct, cls_logits_pt.detach())
            else:
                assert NotImplementedError
            loss = loss_pt + loss_ct * cfg.SOLVER.LOSS_SCALE
            # loss = (loss_pt + loss_ct * cfg.SOLVER.LOSS_SCALE) / 2
            if cfg.TEST.CLS_LOGITS == 'PTCT':
                cls_logits = (cls_logits_pt + cls_logits_ct) / 2
            elif cfg.TEST.CLS_LOGITS == 'PT':
                cls_logits = cls_logits_pt
            elif cfg.TEST.CLS_LOGITS == 'CT':
                cls_logits = cls_logits_ct
            else:
                assert NotImplementedError

            prec1, prec5 = accuracy(cls_logits.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            losses_pt.update(loss_pt.item(), input.size(0))
            losses_ct.update(loss_ct.item(), input.size(0))
            writer.add_scalar('loss_iter/val', loss.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top1', prec1.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top5', prec5.item(), update_val_step)

            cls_top1_b, cls_top5_b, loc_top1_b, loc_top5_b, loc_gt_known_b, top1_loc_right_b, \
                top1_loc_cls_b,top1_loc_mins_b, top1_loc_part_b, top1_loc_more_b, top1_loc_wrong_b = \
                    evaluate_cls_loc(input, target, bbox, cls_logits, cams, image_names, cfg, epoch)
            cls_top1.extend(cls_top1_b)
            cls_top5.extend(cls_top5_b)
            loc_top1.extend(loc_top1_b)
            loc_top5.extend(loc_top5_b)
            top1_loc_right.extend(top1_loc_right_b)
            top1_loc_cls.extend(top1_loc_cls_b)
            top1_loc_mins.extend(top1_loc_mins_b)
            top1_loc_more.extend(top1_loc_more_b)
            top1_loc_part.extend(top1_loc_part_b)
            top1_loc_wrong.extend(top1_loc_wrong_b)

            loc_gt_known.extend(loc_gt_known_b)

            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(val_loader)-1:
                print('Val Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_pt {loss_pt.val:.4f} ({loss_pt.avg:.4f})\t'
                      'Loss_ct {loss_ct.val:.4f} ({loss_ct.avg:.4f})\t'.format(
                    epoch, i+1, len(val_loader), loss=losses, loss_pt=losses_pt, loss_ct=losses_ct))
                print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
                      'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}\n'.format(
                    list2acc(cls_top1), list2acc(cls_top5),
                    list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known)))
        wrong_details = []
        wrong_details.append(np.array(top1_loc_right).sum())
        wrong_details.append(np.array(top1_loc_cls).sum())
        wrong_details.append(np.array(top1_loc_mins).sum())
        wrong_details.append(np.array(top1_loc_part).sum())
        wrong_details.append(np.array(top1_loc_more).sum())
        wrong_details.append(np.array(top1_loc_wrong).sum())
        print('wrong_details:{} {} {} {} {} {}'.format(wrong_details[0], wrong_details[1], wrong_details[2],
                                                       wrong_details[3], wrong_details[4], wrong_details[5]))
    return update_val_step, losses.avg, list2acc(cls_top1), list2acc(cls_top5), \
           list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known)
if __name__ == "__main__":
    main()
