import os
import sys
import datetime
import pprint

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, str_gpus, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams import evaluate_cls_loc
from datasets.cub import CUBDataset
from datasets.imagenet import ImageNetDataset

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
from urllib.request import urlretrieve

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.chdir('../')


def plt_gt_bboxes(image_names, dataset):
    num_img = len(image_names)
    fig, ax = plt.subplots(ncols=num_img, figsize=(num_img * 4, num_img * 4), frameon=False)
    for i in range(num_img):
        img_path = os.path.join(dataset.image_dir, image_names[i][:-4] + '.JPEG')
        with open(img_path, 'rb') as f:
            img = Image.open(img_path).convert('RGB')
        bboxes = dataset.bboxes[dataset.names.index(image_names[i][:-4])]
        ax[i].axis('off')
        ax[i].imshow(img)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            ax[i].add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=3))
    plt.show()


def plt_heatmap(heatmap, img):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 10))
    v = heatmap.cpu().numpy()
    mask_v = cv2.resize(v / (v.max() + 1e-5), img.size)
    im_mask_v = (mask_v[..., np.newaxis] * img).astype("uint8")
    ax1.imshow(mask_v)
    ax2.imshow(im_mask_v)
    ax3.imshow(np.array(img), cmap=plt.cm.rainbow)
    ax3.imshow(mask_v, cmap=plt.cm.jet, alpha=0.6)


def plt_heatmap_list(heatmap_list, img, map_name, norm=True, multi_img=False, on_img=False):
    num_map = len(heatmap_list)
    fig, ax = plt.subplots(1, num_map, figsize=(num_map*4, num_map*4), frameon=False)
    for i in range(num_map):
        v = heatmap_list[i].cpu().numpy()
        if norm:
            v = v / (v.max() + 1e-5)
        mask_v = cv2.resize(v, img.size)
        if multi_img:
            mask_v = (mask_v[...,np.newaxis] * img).astype("uint8")
        if isinstance(map_name,list):
            ax[i].set_title(map_name[i]+'{}'.format(i+1))
        else:
            ax[i].set_title(map_name+'{}'.format(i+1))
        ax[i].axis('off')
        if on_img:
            ax[i].imshow(np.array(img),  cmap=plt.cm.rainbow)
            ax[i].imshow(mask_v, cmap=plt.cm.jet, alpha=0.6)
        else:
            ax[i].imshow(mask_v)


def plt_heatmap_tokenlist(heatmap_list, img, map_name, token_inds, norm=True, multi_img=False, on_img=False):
    num_map = len(heatmap_list)
    num_token = len(token_inds)
    fig, ax = plt.subplots(num_token, num_map, figsize=(32,num_token*4), frameon=False)
    for i in range(num_token):
        if i == 0:
            token_name = 'ClsToken_'
        else:
            token_name = 'PatchToken{}_'.format(token_inds[i])
        for j in range(num_map):
            v = heatmap_list[j,token_inds[i]-1].cpu().numpy()
            if norm:
                v = v / (v.max() + 1e-5)
            mask_v = cv2.resize(v, img.size)
            if multi_img:
                mask_v = (mask_v[...,np.newaxis] * img).astype("uint8")
            if isinstance(map_name,list):
                ax[i,j].set_title(token_name+map_name[i]+'{}'.format(j+1), fontweight='light')
            else:
                ax[i,j].set_title(token_name+map_name+'{}'.format(j+1), fontweight='light')
            ax[i,j].axis('off')
            if on_img:
                ax[i,j].imshow(np.array(img),  cmap=plt.cm.rainbow)
                ax[i,j].imshow(mask_v, cmap=plt.cm.jet, alpha=0.6)
            else:
                ax[i,j].imshow(mask_v)
    plt.tight_layout()
    plt.subplots_adjust(wspace =0.1, hspace =-0.1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_feat_atten_maps(data=None, img_path=None, model=None, lk=0, sk=0):
    if data is None:
        im = Image.open(img_path).convert('RGB')
        x = transform(im).unsqueeze(0)
        im_list = [im]
    else:
        im_list = []
        for i in range(len(data['names'])):
            img_path = os.path.join(dataset.image_dir, data1['names'][i][:-4] + '.JPEG')
            with open(img_path, 'rb') as f:
                im = Image.open(img_path).convert('RGB')
                im_list.append(im)

    with torch.no_grad():
        outputs = model(data['images'].cuda(), return_cam=True, vis=True)
        if len(outputs) > 4:
            x_logits_pt, x_logits_ct, cams, feature_map, attn_weights = outputs
        else:
            x_logits_pt, cams, feature_map, attn_weights = outputs
        n, c, h, w = feature_map.shape
        new_num_patches = h * w
        if len(outputs) > 4:
            ct0_attns = attn_weights.sum(0)[:, 0, 1:new_num_patches + 1].reshape([n, h, w]).unsqueeze(1)
            cls_attns = attn_weights.sum(0)[:, new_num_patches + 1:, 1:new_num_patches + 1].reshape([n, c, h, w])
            cls_attns_multi_blocks = attn_weights[:, :, new_num_patches + 1:, 1:new_num_patches + 1].reshape(
                [attn_weights.size(0), n, c, h, w])
            cams = cls_attns * feature_map
            cams_topk, preds_topk, logits_topk = get_cls_maps_topk(x_logits_pt, cams, largest_k=lk, smallest_k=sk)
            feats_topk, *_ = get_cls_maps_topk(x_logits_pt, feature_map, largest_k=lk, smallest_k=sk)
            ct0_attns_topk, *_ = get_cls_maps_topk(x_logits_pt, ct0_attns.expand(-1, 1000, -1, -1), largest_k=lk,
                                                   smallest_k=sk)
            cls_attns_topk, *_ = get_cls_maps_topk(x_logits_pt, cls_attns, largest_k=lk, smallest_k=sk)
            cls_attns_ct_topk, preds_ct_topk, logits_ct_topk = get_cls_maps_topk(x_logits_ct, cls_attns, largest_k=lk,
                                                                                 smallest_k=sk)
            cls_attns_multi_blocks_topk = []
            for i in range(attn_weights.size(0)):
                cls_attns_multi_blocki_topk, *_ = get_cls_maps_topk(x_logits_pt, cls_attns_multi_blocks[i],
                                                                    largest_k=lk, smallest_k=sk)
                cls_attns_multi_blocks_topk.append(cls_attns_multi_blocki_topk)
            cls_attns_multi_blocks_topk = torch.stack(cls_attns_multi_blocks_topk, dim=0)
            outputs = {
                'logits_pt': x_logits_pt,
                'logits_ct': x_logits_ct,
                'cams': cams,
                'feats': feature_map,
                'ct0_attns': ct0_attns,
                'cls_attns': cls_attns,
                'cls_attns_multi_blocks': cls_attns_multi_blocks
            }
            outputs_pt_topk = {
                'preds': preds_topk,
                'logits': logits_topk,
                'cams': cams_topk,
                'feats': feats_topk,
                'ct0_attns': ct0_attns_topk,
                'cls_attns': cls_attns_topk,
                'cls_attns_multi_blocks_topk': cls_attns_multi_blocks_topk
            }
            outputs_ct_topk = {
                'preds': preds_ct_topk,
                'logits': logits_ct_topk,
                'cls_attns': cls_attns_ct_topk,
            }
            return im_list, outputs, outputs_pt_topk, outputs_ct_topk
        else:
            ct0_attns = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            cams = ct0_attns * feature_map
            cams_topk, preds_topk, logits_topk = get_cls_maps_topk(x_logits_pt, cams, largest_k=lk, smallest_k=sk)
            feats_topk, *_ = get_cls_maps_topk(x_logits_pt, feature_map, largest_k=lk, smallest_k=sk)
            ct0_attns_topk, *_ = get_cls_maps_topk(x_logits_pt, ct0_attns.expand(-1, 200, -1, -1), largest_k=lk,
                                                   smallest_k=sk)
            outputs = {
                'logits_pt': x_logits_pt,
                'cams': cams,
                'feats': feature_map,
                'ct0_attns': ct0_attns
            }
            outputs_pt_topk = {
                'cams': cams_topk,
                'preds': preds_topk,
                'logits': logits_topk,
                'feats': feats_topk,
                'ct0_attns': ct0_attns_topk,
            }
            return im_list, outputs, outputs_pt_topk


def get_cls_maps_topk(x_logits, x_cls_maps, largest_k=0, smallest_k=0):
    with torch.no_grad():
        x_probs = F.softmax(x_logits, dim=1)
        largest_values, largest_indics = x_probs.topk(largest_k, dim=1, largest=True)
        smallest_values, smallest_indics = x_probs.topk(smallest_k, dim=1, largest=False)
        if largest_k > 0 and smallest_k > 0:
            indics = torch.cat((largest_indics, smallest_indics), dim=1)
            values = torch.cat((largest_values, smallest_values), dim=1)
        elif largest_k > 0:
            indics = largest_indics
            values = largest_values
        elif smallest_k > 0:
            indics = smallest_indics
            values = smallest_values
        else:
            assert NotImplementedError
        x_cls_maps_topk = []
        batch_size = x_cls_maps.size(0)
        for i in range(batch_size):
            x_cls_maps_topk.append(x_cls_maps[i, indics[i]])
        x_cls_maps_topk = torch.stack(x_cls_maps_topk)

        return x_cls_maps_topk, indics, values


config_file = './configs/ILSVRC/deit_tscam_small_patch16_224.yaml'
cfg_from_file(config_file)
cfg.BASIC.ROOT_DIR = './'
cfg.TEST.BATCH_SIZE = 16

dataset = ImageNetDataset(root=os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR), cfg=cfg, is_train=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS, pin_memory=True)
dataiter = iter(dataloader)
print(len(dataset))
print(len(dataloader))

images, labels, gt_bboxes, names, one_hot_labels = dataiter.next()
print(images.size())
# plt_gt_bboxes(names, dataset)
data1 = {
    'images':images,
    'labels':labels,
    'gt_bboxes':gt_bboxes,
    'names':names,
    'one_hot_labels':one_hot_labels,
}

model1 = create_deit_model('deit_cls_attn_cam_v4_4_small_patch16_224', pretrained=True, num_classes=cfg.DATA.NUM_CLASSES, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None,)
model1 = model1.cuda()
checkpoint1 = torch.load('./ckpt/ImageNet/deit_cls_attn_cam_v4_4_small_patch16_224_CAM-NORMAL_SEED26_SUM-CAM-THR0.12_BS64/PT_2021-09-04-11-52-T3/ckpt/model_epoch20.pth')
pretrained_dict1 = {k[7:]: v for k, v in checkpoint1['state_dict'].items()}
model1.load_state_dict(pretrained_dict1)
model1.eval()

m1_im_list, m1_outputs, m1_outputs_pt_topk, m1_outputs_ct_topk = get_feat_atten_maps(data=data1, model=model1, lk=6, sk=6)

bs_i = 8
im = m1_im_list[bs_i]
plt_heatmap_list(F.relu(m1_outputs_pt_topk['cams'][bs_i]), im, map_name=['Cams_Cls{}_Top'.format(label) for label in m1_outputs_pt_topk['preds'][bs_i].tolist()], on_img=True)
plt_heatmap_list(F.relu(m1_outputs_pt_topk['feats'][bs_i]), im, map_name=['Feats_Cls{}_Top'.format(label) for label in m1_outputs_pt_topk['preds'][bs_i].tolist()], on_img=True)
plt_heatmap_list(m1_outputs_pt_topk['ct0_attns'][bs_i], im, map_name=['Ct0Attns_Cls{}_Top'.format(label) for label in m1_outputs_pt_topk['preds'][bs_i].tolist()], on_img=True)
plt_heatmap_list(m1_outputs_pt_topk['cls_attns'][bs_i], im, map_name=['ClsAttns_Cls{}_Top'.format(label) for label in m1_outputs_pt_topk['preds'][bs_i].tolist()], on_img=True)
plt_heatmap_list(m1_outputs_ct_topk['cls_attns'][bs_i], im, map_name=['ClsAttnsCt_Cls{}_Top'.format(label) for label in m1_outputs_ct_topk['preds'][bs_i].tolist()], on_img=True)