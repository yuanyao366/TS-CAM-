import torch
import torch.nn as nn
from functools import partial


from .vision_transformer import VisionTransformer, _cfg, Mlp, Block, Attention
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    'deit_cls_attn_cam_v4_8_tiny_patch16_224', 'deit_cls_attn_cam_v4_8_small_patch16_224', 'deit_cls_attn_cam_v4_8_base_patch16_224',
]


class Attention_ExtraTokenDecoder(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False, num_extra_token=0):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, vis=vis)
        self.num_extra_token = num_extra_token

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q_orig, k_orig, v_orig = q[:, :, :-self.num_extra_token, :], k[:, :, :-self.num_extra_token, :], v[:, :, :-self.num_extra_token, :]
        q_extra, k_extra, v_extra = q[:, :, -self.num_extra_token:, :], k[:, :, -self.num_extra_token:, :], v[:, :, -self.num_extra_token:, :]

        attn_orig = (q_orig @ k_orig.transpose(-2, -1)) * self.scale
        attn_orig = attn_orig.softmax(dim=-1)
        weights_orig = attn_orig
        attn_orig = self.attn_drop(attn_orig)
        x_orig = (attn_orig @ v_orig).transpose(1, 2).reshape(B, N-self.num_extra_token, C)

        attn_extra_orig = (q_extra @ k_orig.detach().transpose(-2, -1)) * self.scale #Bx6x200x197
        attn_extra_extra = (q_extra @ k_extra.transpose(-2, -1)) * self.scale #Bx6x200x200
        attn_extra_extra_diag = torch.diagonal(attn_extra_extra, dim1=-2, dim2=-1) #Bx6x200
        attn_extra = torch.cat((attn_extra_orig, attn_extra_extra_diag.unsqueeze(dim=-1)), dim=-1) #Bx6x200x198
        attn_extra = attn_extra.softmax(dim=-1)
        weights_extra = attn_extra[:, :, :, :-1]
        attn_extra = self.attn_drop(attn_extra)
        v_orig_expand = v_orig.detach().unsqueeze(dim=2).expand(-1,-1,self.num_extra_token,-1,-1) #Bx6x200x197x64
        v_orig_extra = torch.cat((v_orig_expand, v_extra.unsqueeze(dim=3)), dim=3) #Bx6x200x198x64
        x_extra = torch.einsum('bhij,bhijk->bhik', (attn_extra, v_orig_extra)).transpose(1, 2).reshape(B, self.num_extra_token, C)

        x = torch.cat((x_orig, x_extra), dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        weights = torch.cat((weights_orig, weights_extra), dim=2)
        return x, weights


class Block_ExtraTokenDecoder(Block):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False, num_extra_token=0):
        super().__init__(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                         drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, vis=vis)
        self.attn = Attention_ExtraTokenDecoder(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis, num_extra_token=num_extra_token)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class TSAttnCAM_V4_8(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, vis=vis)

        self.num_cls_attn = self.num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_ExtraTokenDecoder(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis, num_extra_token=self.num_cls_attn)
            for i in range(depth)])

        self.head_pt = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head_ct = nn.Linear(self.embed_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_cls_attn, self.embed_dim))

        trunc_normal_(self.extra_cls_token, std=.02)
        self.head_pt.apply(self._init_weights)
        self.head_ct.apply(self._init_weights)


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1) != x.shape[1]:
            temp_pos_embed, new_num_patches = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        else:
            temp_pos_embed = self.pos_embed
            new_num_patches = self.patch_embed.num_patches

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        extra_cls_token = self.extra_cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + temp_pos_embed
        x = torch.cat((x, extra_cls_token), dim=1)
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 1:new_num_patches+1], x[:, new_num_patches+1:], attn_weights

    def forward(self, x, return_cam=False, vis=False):
        x_patch, x_cls, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        patch_size = self.patch_embed.patch_size
        P_H, P_W = x.shape[2]//patch_size[0], x.shape[3]//patch_size[1]
        x_patch = torch.reshape(x_patch, [n, P_H, P_W, c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head_pt(x_patch)
        x_logits_pt = self.avgpool(x_patch).squeeze(3).squeeze(2)

        x_logits_ct = self.head_ct(x_cls).squeeze(-1)

        if self.training:
            return x_logits_pt, x_logits_ct
        else:
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            new_num_patches = h * w
            #cams = attn_weights.mean(0).mean(1)[:, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights[:-1].sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights.sum(0)[:][:, 1:, 1:].sum(1).reshape([n, h, w]).unsqueeze(1)
            cls_attns = attn_weights.sum(0)[:, new_num_patches+1:, 1:new_num_patches+1].reshape([n, self.num_cls_attn, h, w])
            cams = cls_attns * feature_map                           # B * C * 14 * 14

            if vis:
                return x_logits_pt, x_logits_ct, cams, feature_map, attn_weights

            return x_logits_pt, x_logits_ct, cams

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        print('InterpolateInitPosEmbed')
        embedding_size = pos_embed.shape[-1]
        num_patches = self.patch_embed.num_patches
        num_cls_tokens = pos_embed.shape[-2] - num_patches
        orig_size = int(num_patches ** 0.5)

        cls_pos_embed = pos_embed[:, :num_cls_tokens, :]
        patch_pos_embed = pos_embed[:, num_cls_tokens:, :]
        patch_pos_embed = patch_pos_embed.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

        patch_size = self.patch_embed.patch_size
        new_P_H, new_P_W = img_size[0]//patch_size[0], img_size[1]//patch_size[1]
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

        return scale_pos_embed, new_P_H*new_P_W


@register_model
def deit_cls_attn_cam_v4_8_tiny_patch16_224(pretrained=False, **kwargs):
    model = TSAttnCAM_V4_8(
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
def deit_cls_attn_cam_v4_8_small_patch16_224(pretrained=False, **kwargs):
    model = TSAttnCAM_V4_8(
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
        for k,v in checkpoint.items():
            if k not in model_dict:
                print(k)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_cls_attn_cam_v4_8_base_patch16_224(pretrained=False, **kwargs):
    model = TSAttnCAM_V4_8(
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





