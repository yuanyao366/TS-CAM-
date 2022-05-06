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

class GradAttneion:
    def __init__(self, model, attention_layer_name='attn_drop'):
        self.model = model
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)
        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def get_attention_gradient(self, module, grad_input, grad_output):
        # print('aa')
        # print(grad_output[0].size())
        # print(torch.sum(grad_input[0]-grad_output[0]))
        # print('bb')
        self.attention_gradients.append(grad_output[0])

    def __call__(self, input, index=None):
        cls_logits = []
        feats = []
        attn_weights = []
        attns = []
        attns_grad = []
        for i in range(input.size(0)):
            self.attentions = []
            self.attention_gradients = []
            cls_logits_bi, cams_bi, feats_bi, attn_weights_bi = self.model(input[[i]], return_cam=True, vis=True)
            if index == None:
                index = np.argmax(cls_logits_bi.cpu().data.numpy(), axis=-1)
            one_hot = np.zeros((1, cls_logits_bi.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(input.get_device())
            one_hot = torch.sum(one_hot * cls_logits_bi)
            self.model.zero_grad()
            # one_hot.backward(retain_graph=True)
            one_hot.backward()
            cls_logits.append(cls_logits_bi[0])
            feats.append(feats_bi[0])
            attn_weights.append(attn_weights_bi[:, 0])
            attns.append(torch.stack(self.attentions)[:,0])
            attns_grad.append(torch.stack(self.attention_gradients)[:,0])

        cls_logits = torch.stack(cls_logits, dim=0)
        feats = torch.stack(feats, dim=0) #BxCx14x14
        n, c, h, w = feats.shape
        attn_weights = torch.stack(attn_weights, dim=1)
        mean_attns = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
        attns = torch.stack(attns, dim=1)[:,:,:,0,1:] #LxBx6x196
        attns_grad = torch.stack(attns_grad, dim=1)[:,:,:,0,1:]
        # attns_grad_weight = attns_grad.mean(-1, keepdim=True) #LxBx6x1
        # grad_attns = torch.mean(attns*attns_grad_weight, dim=2).clamp(min=0) #LxBx196
        grad_attns = torch.mean(attns * attns_grad, dim=2).clamp(min=0)
        grad_attns = grad_attns.sum(0).reshape([n, h, w]).unsqueeze(1)
        mean_attn_cams = mean_attns * feats
        grad_attn_cams = grad_attns * feats

        return cls_logits, mean_attn_cams, grad_attn_cams, feats, mean_attns.expand(-1,c,-1,-1), grad_attns.expand(-1,c,-1,-1)



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
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)

    print('Preparing networks done!')
    return device, model, cls_criterion

def main():
    args = update_config()

    # create checkpoint directory
    # cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
    #     cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    checkpoint = torch.load(args.resume)
    cfg.BASIC.SAVE_DIR = os.path.join(args.resume[:args.resume.find('/ckpt/model_')], 'eval', 'epoch{}_{}{}-CAM-THR{}_BS{}'.
                                      format(checkpoint["epoch"], cfg.MODEL.CAM_MERGE, cfg.MODEL.ATTN_DISCARD_RATIO, cfg.MODEL.CAM_THR, cfg.TEST.BATCH_SIZE))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_{}_'.format(cfg.TEST.CAM_TYPE) + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, cls_criterion = creat_model(cfg, args)

    update_val_step = 0
    update_val_step, _, cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known, wrong_details = \
        val_loc_one_epoch(val_loader, model, device, cls_criterion, writer, cfg, update_val_step)
    print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
          'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}\n'.format( cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known))
    print('wrong_details:{} {} {} {} {} {}'.format(wrong_details[0], wrong_details[1], wrong_details[2],
                                                   wrong_details[3], wrong_details[4], wrong_details[5]))
    print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def val_loc_one_epoch(val_loader, model, device, criterion, writer, cfg, update_val_step):

    losses = AverageMeter()

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

    # with torch.no_grad():
    model.eval()
    model_gradwrap = GradAttneion(model)
    for i, (input, target, bbox, image_names, one_hot_target) in enumerate(val_loader):
        # update iteration steps
        update_val_step += 1

        target = target.to(device)
        input = input.to(device)

        # cls_logits, cams = model(input, return_cam=True)
        cls_logits, mean_attn_cams, grad_attn_cams, feats, mean_attns, grad_attns = model_gradwrap(input)
        if cfg.TEST.CAM_TYPE == 'MeanAttenCam':
            cams = mean_attn_cams
        elif cfg.TEST.CAM_TYPE == 'GradoutAttenCam' or cfg.TEST.CAM_TYPE == 'GradoutMaskAttenCam':
            cams = grad_attn_cams
        elif cfg.TEST.CAM_TYPE == 'Feat':
            cams = feats
        elif cfg.TEST.CAM_TYPE == 'MeanAtten':
            cams = mean_attns
        elif cfg.TEST.CAM_TYPE == 'GradoutAtten' or cfg.TEST.CAM_TYPE == 'GradoutMaskAtten':
            cams = grad_attns
        loss = criterion(cls_logits, target)

        prec1, prec5 = accuracy(cls_logits.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        writer.add_scalar('loss_iter/val', loss.item(), update_val_step)
        writer.add_scalar('acc_iter/val_top1', prec1.item(), update_val_step)
        writer.add_scalar('acc_iter/val_top5', prec5.item(), update_val_step)

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                0, i+1, len(val_loader), loss=losses))
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
