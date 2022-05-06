import os
import numpy as np
import shutil
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from datasets.cub import CUBDataset
from datasets.imagenet import ImageNetDataset


def creat_data_loader(cfg, root_dir):
    """ Create data_loaders for training and validation
    :param cfg: hyperparameter configuration
    :param root_dir: dataset root path
    :return:
    """
    print('==> Preparing data...')
    if cfg.DATA.DATASET == 'CUB':
        train_loader = torch.utils.data.DataLoader(
            CUBDataset(root=root_dir, cfg=cfg, is_train=True),
            batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CUBDataset(root=root_dir, cfg=cfg, is_train=False),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
    elif cfg.DATA.DATASET == 'ImageNet':
        train_loader = torch.utils.data.DataLoader(
            ImageNetDataset(root=root_dir, cfg=cfg, is_train=True),
            batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            ImageNetDataset(root=root_dir, cfg=cfg, is_train=False),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)

    print('done!')
    return train_loader, val_loader


def str_gpus(ids):
    str_ids = ''
    for id in ids:
        str_ids =  str_ids + str(id)
        str_ids =  str_ids + ','

    return str_ids


def map_sklearn(labels, results):
    map = average_precision_score(labels, results, average="micro")
    return map


def adjust_learning_rate(optimizer, epoch, cfg):
    """"Sets the learning rate to the initial LR decayed by lr_factor"""
    lr_decay = cfg.SOLVER.LR_FACTOR**(sum(epoch > np.array(cfg.SOLVER.LR_STEPS)))
    lr = cfg.SOLVER.START_LR * lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']


def adjust_learning_rate_normal(optimizer, epoch, cfg):
    """ Adjust the learning rate of model parameters
    :param optimizer: optimizer (e.g. SGD, AdamW, Adam)
    :param epoch: training epoch
    :param cfg: hyperparameter configuration
    :return:
    """
    e = False
    for step in cfg.SOLVER.LR_STEPS[::-1]:
        if epoch % step == 0:
            e = True
            break
    lr_decay = cfg.SOLVER.LR_FACTOR
    if e:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_decay * param_group['lr']


def save_checkpoint(state, save_dir, epoch, is_best):
    filename = os.path.join(save_dir, 'ckpt_'+str(epoch)+'.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
    :param output: tensor of shape B x K, predicted logits of image from model
    :param target: tensor of shape B X 1, ground-truth logits of image
    :param topk: top predictions
    :return: list of precision@k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def list2acc(results_list):
    """
    :param results_list: list contains 0 and 1
    :return: accuarcy
    """
    accuarcy = results_list.count(1)/len(results_list)
    return accuarcy


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, distillation_type: str, tau: float):
        super().__init__()
        assert distillation_type in ['soft', 'hard']
        self.distillation_type = distillation_type
        self.tau = tau

    def forward(self, outputs, teacher_outputs):
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))

        return distillation_loss

