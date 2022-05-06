import torch
import torch.nn as nn
from functools import partial


from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    'deit_tscam_v6_5_tiny_patch16_224', 'deit_tscam_v6_5_small_patch16_224', 'deit_tscam_v6_5_base_patch16_224',
]



class TSCAM_V6_5(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = 6
        self.m = 0.999
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer("extra_cls_token", torch.randn(12, self.num_classes, self.embed_dim))
        # self.extra_cls_token = nn.Parameter(torch.zeros(12, self.num_classes, self.embed_dim), requires_grad=False)
        # trunc_normal_(self.extra_cls_token, std=.02)
        self.head.apply(self._init_weights)
        self.activations = {}
        for blk_i, blk in enumerate(self.blocks):
            layer_name = 'block{}.attn.qkv'.format(blk_i)
            self.activations[layer_name] = None
            blk.attn.qkv.register_forward_hook(partial(self.save_activation, layer_name))


    def forward_features(self, x, label=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        x_list = []
        for layer_name in self.activations.keys():
            self.activations[layer_name] = None
        for blk_i, blk in enumerate(self.blocks):
            x_list.append(x)
            x, weights = blk(x)
            # attn_weights.append(weights)
            if self.training:
                self._momentum_update_key_token(x_list[blk_i], self.activations['block{}.attn.qkv'.format(blk_i)], blk_i, label, blk.attn.scale)
            else:
                extra_cls_token = self.extra_cls_token[blk_i].unsqueeze(0).expand(B, -1, -1)
                weights_extra = self._extra_token_decoder(x_list[blk_i], extra_cls_token, blk, blk.attn)
                weights = torch.cat((weights, weights_extra), dim=2)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, label=None, return_cam=False, vis=False, cfg=None):
        x_cls, x_patch, attn_weights = self.forward_features(x,label)
        n, p, c = x_patch.shape
        patch_size = self.patch_embed.patch_size
        P_H, P_W = x.shape[2]//patch_size[0], x.shape[3]//patch_size[1]
        x_patch = torch.reshape(x_patch, [n, P_H, P_W, c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        if self.training:
            return x_logits
        else:
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            new_num_patches = P_H * P_W
            #cams = attn_weights.mean(0).mean(1)[:, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights[:-1].sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights.sum(0)[:][:, 1:, 1:].sum(1).reshape([n, h, w]).unsqueeze(1)
            cls_attns = attn_weights.sum(0)[:, new_num_patches+1:, 1:new_num_patches+1] # B * C * 196
            ct0_attns = attn_weights.sum(0)[:, 0, 1:new_num_patches+1].unsqueeze(1) # B * 1 * 196
            patch_attns = attn_weights.sum(0)[:, 1:new_num_patches+1:, 1:new_num_patches+1] # B * 196 * 196

            if cfg.TEST.CAM_TYPE.find('Attn') > 0:
                attn_type = cfg.TEST.CAM_TYPE[:cfg.TEST.CAM_TYPE.find('Attn')]
                if attn_type == 'Ct0':
                    attn_map = ct0_attns.reshape([n, 1, P_H, P_W]).expand(-1,self.num_classes,-1,-1)
                elif attn_type == 'Cls':
                    attn_map = cls_attns.reshape([n, self.num_classes, P_H, P_W])
                elif attn_type == 'Patch':
                    attn_map = torch.mean(patch_attns, dim=1, keepdim=True).reshape([n, 1, P_H, P_W]).expand(-1,self.num_classes,-1,-1)
                elif attn_type == 'Ct0Patch':
                    attn_map = torch.einsum('bci,bij->bcj', (ct0_attns, patch_attns)).reshape([n, 1, P_H, P_W]).expand(-1,self.num_classes,-1,-1)
                elif attn_type == 'ClsPatch':
                    attn_map = torch.einsum('bci,bij->bcj', (cls_attns, patch_attns)).reshape([n, self.num_classes, P_H, P_W])
            if 'AttnFeatCam' in cfg.TEST.CAM_TYPE:
                cams = attn_map * feature_map
            elif 'Feat' in cfg.TEST.CAM_TYPE:
                cams = feature_map
            elif 'Attn' in cfg.TEST.CAM_TYPE:
                cams = attn_map

            if vis:
                # return x_logits, cams, feature_map, attn_weights
#                 return {'logits': x_logits, 'cams': cams, 'attns': ct0_attns.expand(-1, self.num_classes, -1, -1), 'feats': feature_map}
                return {'logits': x_logits, 'cams': cams, 'ct0_attns': ct0_attns.expand(-1, self.num_classes, -1, -1), 'feats': feature_map}

            return x_logits, cams#, feature_map, ct0_attns.expand(-1,self.num_classes,-1,-1)

    def save_activation(self, attr_name, module, input, output):
        activation = output
        self.activations[attr_name] = activation.detach().clone()

    @torch.no_grad()
    def _momentum_update_key_token(self, x, qkv, blk_idx, token_idx, norm_scale):
        B, N, C = x.shape
        v_x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)[:,:,1:,:] #Bx6x196x64
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * norm_scale
        attn_cls_patch = attn[:, :, 0, 1:].unsqueeze(2).softmax(dim=-1)  # Bx6x1x196
        x_cls = (attn_cls_patch @ v_x).transpose(1, 2).reshape(B, 1, C)[:,0,:]
        self.extra_cls_token[blk_idx, token_idx] = self.extra_cls_token[blk_idx, token_idx] * self.m + x_cls * (1. - self.m)

    @torch.no_grad()
    def _extra_token_decoder(self, x_orig, x_extra, blk, attn):
        x_cat = blk.norm1(torch.cat((x_orig, x_extra), dim=1))
        B, N, C = x_cat.shape
        qkv = attn.qkv(x_cat).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_orig, k_orig, v_orig = q[:, :, :-self.num_classes, :], k[:, :, :-self.num_classes, :], v[:, :, :-self.num_classes, :]
        q_extra, k_extra, v_extra = q[:, :, -self.num_classes:, :], k[:, :, -self.num_classes:, :], v[:, :, -self.num_classes:, :]
        attn_extra = (q_extra @ k_orig.transpose(-2, -1)) * attn.scale
        attn_extra = attn_extra.softmax(dim=-1)
        return attn_extra



@register_model
def deit_tscam_v6_5_tiny_patch16_224(pretrained=False, **kwargs):
    model = TSCAM_V6_5(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_tscam_v6_5_small_patch16_224(pretrained=False, **kwargs):
    model = TSCAM_V6_5(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_tscam_v6_5_base_patch16_224(pretrained=False, **kwargs):
    model = TSCAM_V6_5(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model





