# ----------------------------------------------------------------------------------------------------------
# TS-CAM
# Copyright (c) Learning and Machine Perception Lab (LAMP), SECE, University of Chinese Academy of Science.
# ----------------------------------------------------------------------------------------------------------
import os
import sys
import datetime
import pprint
import numpy as np

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader,\
    AverageMeter, accuracy, list2acc, adjust_learning_rate
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer

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
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # loss
    cls_criterion_pt = torch.nn.CrossEntropyLoss().to(device)
    if cfg.SOLVER.LOSS == 'CE':
        cls_criterion_ct = torch.nn.CrossEntropyLoss().to(device)
    elif cfg.SOLVER.LOSS == 'BCE':
        cls_criterion_ct = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        assert NotImplementedError
    print(cls_criterion_pt)
    print(cls_criterion_ct)
    print('Preparing networks done!')
    return device, model, cls_criterion_pt, cls_criterion_ct

def main():
    args = update_config()

    # create checkpoint directory
    # cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
    #     cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    checkpoint = torch.load(args.resume)
    cfg.BASIC.SAVE_DIR = os.path.join(args.resume[:args.resume.find('/ckpt/model_')], 'eval', 'epoch{}_{}{}-CAM-THR{}_BS{}'.
                                      format(checkpoint["epoch"], cfg.MODEL.CAM_MERGE, cfg.MODEL.ATTN_DISCARD_RATIO, cfg.MODEL.CAM_THR, cfg.TEST.BATCH_SIZE))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    # log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    # ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    mkdir(cfg.BASIC.SAVE_DIR)
    if args.extra_token_blk_idx >= 0:
        log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_{}_ExtraBlk{}_'.format(cfg.TEST.CLS_LOGITS, args.extra_token_blk_idx) + cfg.BASIC.TIME + '.txt')
    else:
        log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_{}_{}_'.format(cfg.TEST.CLS_LOGITS, cfg.TEST.CAM_TYPE) + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    # writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, cls_criterion_pt, cls_criterion_ct = creat_model(cfg, args)

    update_val_step = 0
    update_val_step, _, cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known, wrong_details = \
        val_loc_one_epoch(val_loader, model, device, cls_criterion_pt, cls_criterion_ct, cfg, update_val_step, args)
    print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
          'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}\n'.format( cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known))
    print('wrong_details:{} {} {} {} {} {}'.format(wrong_details[0], wrong_details[1], wrong_details[2],
                                                   wrong_details[3], wrong_details[4], wrong_details[5]))
    print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def val_loc_one_epoch(val_loader, model, device, criterion_pt, criterion_ct, cfg, update_val_step, args):

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

            # if args.extra_token_blk_idx >= 0:
            #     cls_logits_pt, cls_logits_ct, cams = model(input, return_cam=True, extra_token_block_idx=args.extra_token_blk_idx)
            # else:
            # cls_logits_pt, cls_logits_ct, cams = model(input, return_cam=True)
            cls_logits_pt, cls_logits_ct, tscams, feats, ct0_attns, cls_attns = model(input, return_cam=True)
            if cfg.TEST.CAM_TYPE == 'MeanAttenCam':
                cams = tscams
            elif cfg.TEST.CAM_TYPE == 'Feat':
                cams = feats
            elif cfg.TEST.CAM_TYPE == 'Ct0Atten':
                cams = ct0_attns
            elif cfg.TEST.CAM_TYPE == 'ClsAtten':
                cams = cls_attns
            else:
                assert NotImplementedError
            loss_pt = criterion_pt(cls_logits_pt, target)
            if cfg.SOLVER.LOSS == 'BCE':
                loss_ct = criterion_ct(cls_logits_ct, one_hot_target) * cfg.SOLVER.LOSS_SCALE
            else:
                loss_ct = criterion_ct(cls_logits_ct, target)
            loss = loss_pt + loss_ct
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
            # writer.add_scalar('loss_iter/val', loss.item(), update_val_step)
            # writer.add_scalar('acc_iter/val_top1', prec1.item(), update_val_step)
            # writer.add_scalar('acc_iter/val_top5', prec5.item(), update_val_step)

            cls_top1_b, cls_top5_b, loc_top1_b, loc_top5_b, loc_gt_known_b, top1_loc_right_b, \
                top1_loc_cls_b,top1_loc_mins_b, top1_loc_part_b, top1_loc_more_b, top1_loc_wrong_b = \
                    evaluate_cls_loc(input, target, bbox, cls_logits, cams, image_names, cfg, 0)
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
                    0, i+1, len(val_loader), loss=losses, loss_pt=losses_pt, loss_ct=losses_ct))
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
    return update_val_step, losses.avg, list2acc(cls_top1), list2acc(cls_top5), \
           list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known), wrong_details


if __name__ == "__main__":
    main()
