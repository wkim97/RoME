"""ViT-B/16 backbone with LoRA-MoE adapters, adapted from timm."""

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .lora_moe_linear import LoRAMoEAdapterLinear
from .lora_moe_mlp import LoRAMoEAdapterMLP

_logger = logging.getLogger(__name__)

VIT_B16_CFG = {
    'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/'
           'v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .9, 'interpolation': 'bicubic',
    'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
    'first_conv': 'patch_embed.proj', 'classifier': 'head',
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., adapter_cls=LoRAMoEAdapterMLP,
                 **adapter_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_adapter = adapter_cls(dim, dim * 3, **adapter_kwargs)
        self.proj_adapter = adapter_cls(dim, dim, **adapter_kwargs)

    def forward(self, x, block_index):
        B, N, C = x.shape

        qkv_lora, qkv_gate = self.qkv_adapter(x, block_index)
        qkv = self.qkv(x) + qkv_lora
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        proj_lora, proj_gate = self.proj_adapter(x, block_index)
        x = self.proj_drop(self.proj(x) + proj_lora)
        return x, [qkv_gate, proj_gate]


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, **adapter_kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, **adapter_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, block_index):
        attn, gate = self.attn(self.norm1(x), block_index)
        x = x + self.drop_path(attn)
        x = x + self.mlp(self.norm2(x))
        return x, gate


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f'Input image size ({H}*{W}) does not match model ({self.img_size[0]}*{self.img_size[1]}).'
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, r=16, lora_alpha=16, num_lora=4, s=4.0, b=6.0,
                 mlp_gate=True, gate_proj_dim=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        adapter_cls = LoRAMoEAdapterMLP if mlp_gate else LoRAMoEAdapterLinear
        adapter_kwargs = dict(adapter_cls=adapter_cls, r=r, lora_alpha=lora_alpha,
                              num_lora=num_lora, depth=depth, s=s, b=b,
                              gate_proj_dim=gate_proj_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, **adapter_kwargs)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        gates = []
        for i, blk in enumerate(self.blocks):
            x, gate = blk(x, i)
            gates.append(gate)
        return self.norm(x)[:, 0], gates

    def forward(self, x):
        """Returns ``(logits, gates)``, one ``[qkv_aux, proj_aux]`` pair per block."""
        feats, gates = self.forward_features(x)
        return self.head(feats), gates


def resize_pos_embed(posemb, posemb_new):
    ntok_new = posemb_new.shape[1] - 1
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old, gs_new = int(math.sqrt(len(posemb_grid))), int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


def _filter_pretrained(state_dict, model, patch_size):
    state_dict = state_dict.get('model', state_dict)
    out = {}
    for k, v in state_dict.items():
        if k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed)
        elif k == 'patch_embed.proj.weight' and v.shape != model.patch_embed.proj.weight.shape:
            if v.shape[2] % patch_size == 0:
                v = v.reshape(*v.shape[:2], patch_size, v.shape[2] // patch_size,
                              patch_size, v.shape[3] // patch_size).sum(dim=[3, 5])
            else:
                v = F.interpolate(v, model.patch_embed.proj.weight.shape[-1])
        out[k] = v
    return out


def vit_base_patch16_224(pretrained=False, patch_size=16, **kwargs):
    kwargs.setdefault('embed_dim', 768)
    kwargs.setdefault('depth', 12)
    kwargs.setdefault('num_heads', 12)
    model = VisionTransformer(patch_size=patch_size, **kwargs)
    model.default_cfg = VIT_B16_CFG
    if pretrained:
        load_pretrained(
            model, VIT_B16_CFG, num_classes=kwargs.get('num_classes', 1000),
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=partial(_filter_pretrained, patch_size=patch_size),
            strict=False)
    return model
