# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.layer.stnorm import STnorm
from .kan import MoKLayer

class Result:

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


def interpolate_pos_encoding(pos_embed, H, W):
    num_patches = H * W

    N = pos_embed.shape[1]
    if num_patches == N or W == H:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(H, W),
        mode='bicubic',
        align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = MoKLayer(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = MoKLayer(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel( 
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = MoKLayer(dim, dim, bias=qkv_bias)
        self.k_proj = MoKLayer(dim, dim, bias=qkv_bias)
        self.v_proj = MoKLayer(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MoKLayer(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

        self.spatial_norm = STnorm(input_dim=dim,eps=1e-5)
        # self.spatial_norm = RevDain(input_dim=dim)

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        new_x = self.spatial_norm(new_x.permute(0, 2, 1))
        new_x = new_x.permute(0, 2, 1)

        return new_x, attn_dict


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = MoKLayer(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = MoKLayer(dim, dim, bias=qkv_bias)
            self.k_proj = MoKLayer(dim, dim, bias=qkv_bias)
            self.v_proj = MoKLayer(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = MoKLayer(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In GroupViT setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
    """

    def __init__(self,
                 dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 zero_init_group_token=False):

        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint

        self.group_projector = group_projector

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)
            else:
                cat_x = blk(cat_x)

        x, group_token = self.split_x(cat_x)

        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)

        return x, group_token, attn_dict


class PatchEmbed(nn.Module):

    def __init__(self, input_size=(32,32), kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        #input_size = to_2tuple(input_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.input_size = input_size
        self.patches_resolution = (
            int((input_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((input_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.input_size[0] and W == self.input_size[1], \
                f"Input size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


class GlobalSemanticEncoder(nn.Module):

    def __init__(self,
                 input_size=32,
                 patch_size=2,
                 in_chans=10,
                 embed_dim=128,
                 depths=8,
                 num_heads=4,
                 num_group_tokens=18,
                 num_output_groups=18,
                 hard_assignment=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pos_embed_type='simple'):
        super().__init__()
        self.num_layers = depths
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        assert pos_embed_type in ['simple', 'fourier']

        norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
            input_size=input_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if pos_embed_type == 'simple':
            self.pos_embed = self.build_simple_position_embedding()
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        num_input_token = num_patches
        num_output_token = num_input_token
        # build layers
        out_dim = embed_dim
        downsample = GroupingBlock(
            dim=embed_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_group_token=num_group_tokens,
            num_output_group=num_output_groups,
            norm_layer=norm_layer,
            hard=hard_assignment,
            gumbel=hard_assignment)

        group_projector = None
        self.layer = GroupingLayer(
                dim=embed_dim,
                num_input_token=num_input_token,
                depth=depths,
                num_heads=num_heads,
                num_group_token=num_group_tokens,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr,
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None)

        self.norm = norm_layer(self.num_features)
        self.head = MoKLayer(self.num_features, embed_dim)

        self.apply(self._init_weights)


    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False
        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed

    def forward_head(self, x):
        """

        Args:
            x: shape [B, L, C]

        Returns:

        """
        # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_features(self, x, *, return_attn=False):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)
        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)

        group_token = None
        x, group_token, attn_dict = self.layer(x, group_token, return_attn=return_attn)

        x = self.norm(x)

        return x, group_token, attn_dict


    def forward(self, x, *, return_feat=False, return_attn=False, as_dict=False):
        x, group_token, attn_dicts = self.forward_features(x, return_attn=return_attn)
        x_feat = x if return_feat else None

        outs = Result(as_dict=as_dict)
        outs.append(self.forward_head(x), name='x')
        if return_feat:
            outs.append(x_feat, name='feat')

        if return_attn:
            outs.append(attn_dicts, name='attn_dicts')

        return outs.as_return()
