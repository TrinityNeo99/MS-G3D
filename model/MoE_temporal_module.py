#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
import copy


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# Transformer 原始位置编码
class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_sequence_length, self.d_model,
                         dtype=torch.float)  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None,
                                                                                   :]  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
        pe = pe.unsqueeze(0)
        return torch.tensor(pe).cuda()


def temporal_window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)  # B, nW, W, c
    windows = x.view(-1, window_size, C)  # B*nW, W, c
    return windows


def temporal_window_reverse(windows, window_size, N):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (N / window_size))
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.view(B, N, -1)
    return x


class TemporalWindowAttention(nn.Module):  # attention block
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((2 * window_size - 1), num_heads)))  # 2*W-1, nH

        coords = torch.arange(self.window_size)
        relative_distance = torch.transpose(torch.arange(self.window_size).unsqueeze(0), 0, 1)
        reversed_coords = torch.flip(coords, [0])
        _, relative_index = torch.meshgrid([reversed_coords, reversed_coords])
        relative_position_index = relative_index + relative_distance  # W, W

        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print("the shape of attention is: ", attn.shape)
        # TODO this attention
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # W,W,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalWindowTransformerBlock(nn.Module):  # transformer block
    def __init__(self, dim, num_frames, num_heads, window_size=10, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = TemporalWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            N = self.num_frames
            img_mask = torch.zeros((1, N, 1))  # 1 H W 1
            slices = (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for s in slices:
                img_mask[:, s, :] = cnt
                cnt += 1

            mask_windows = temporal_window_partition(img_mask, self.window_size)  # nW, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)  # nW, w_s
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW w_s w_s
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            # partition windows
            x_windows = temporal_window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = temporal_window_partition(shifted_x, self.window_size)  # nW*B,  window_size, C

        x_windows = x_windows.view(-1, self.window_size, C)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows, attention_data = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = temporal_window_reverse(attn_windows, self.window_size, N)  # B N C
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            shifted_x = temporal_window_reverse(attn_windows, self.window_size, N)  # B N C
            x = shifted_x
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attention_data


class TemporalWindowTransformerLayer(nn.Module):
    def __init__(self, dim, num_frames, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            TemporalWindowTransformerBlock(dim=dim, num_frames=num_frames,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        self.window_attentions = None

    def forward(self, x):
        self.window_attentions = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, att = blk(x)
                self.window_attentions.append(att.cpu())
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Temporal_unit(nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, residual=True, num_frames=100, depth=2, window_size=10,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, temporal_merge=False):
        super().__init__()
        assert num_frames % window_size == 0, "num_frames can not divide window size"
        self.mlp_ratio = mlp_ratio
        self.ape = ape
        self.patch_norm = patch_norm
        self.in_channels = in_channels
        self.window_size = window_size
        self.out_channels = out_channels

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_frames, in_channels))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.trans = TemporalWindowTransformerLayer(
            dim=in_channels,
            num_frames=num_frames,
            depth=depth,
            num_heads=heads,
            window_size=window_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            # drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            norm_layer=norm_layer,
            downsample=TemporalWindowPatchMerging if temporal_merge else None,
            use_checkpoint=use_checkpoint
        )

        self.norm = norm_layer(out_channels)
        self.apply(self._init_weights)
        # self.linear = nn.Linear(in_channels, out_channels) # 使用temporal_merge以后不需要使用线性层进行升维
        self.pos_drop = nn.Dropout(p=drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        assert x.size()[1] % self.window_size == 0, "current num_frames can not be divided by window size"
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.trans(x)

        x = self.norm(x)  # B L C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.linear(x)
        return x


class TemporalWindowPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        B, L, C = x.shape
        assert L % 2 == 0, f"x length ({L}) are not even."

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


class MoE_temporal_module(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_heads=3, residual=True,
                 dropout=0.1, temporal_merge=False, expert_windows_size=[8, 32], num_frames=256, temporal_depth=2,
                 expert_weights=[0.5, 0.5, 0.5], isLearnable=True, channelDivide=False,
                 temporal_ape=False, use_zloss=0, topK=-1):
        super().__init__()
        self.channelDivide = channelDivide
        self.expert_weights_learnable = isLearnable
        if len(expert_windows_size) == 1:
            expert_weights = [1.0]
        # assert len(expert_weights) == len(
        #     expert_windows_size), "the numbers of expert weights and their windows size are not equal"
        if not isLearnable:
            expert_weights = torch.tensor(expert_weights)
            self.register_buffer("expert_weights", expert_weights)
        self.num_experts = len(expert_windows_size)
        self.temporal_merge = temporal_merge
        self.expert_linear = nn.Linear(in_channels, self.num_experts)
        self.expert_softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            self.experts.append(
                Temporal_unit(in_channels // self.num_experts if self.channelDivide else in_channels,
                              out_channels // self.num_experts if self.channelDivide else out_channels,
                              residual=residual,
                              heads=temporal_heads,
                              temporal_merge=temporal_merge, window_size=expert_windows_size[i],
                              num_frames=num_frames,
                              ape=temporal_ape, depth=temporal_depth))

        # TODO ape
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)  # TODO drop out org 0.3
        self.bn2 = nn.BatchNorm2d(out_channels)
        bn_init(self.bn2, 1)
        if not residual:
            self.residual = lambda x: 0
        elif temporal_merge == True:
            self.residual = TemporalWindowPatchMerging(in_channels)
        elif temporal_merge == False:
            self.residual = lambda x: x
        self.use_zloss = use_zloss
        self.expert_attention = None
        self.topK = topK
        self.window_attention = None

    def forward(self, x):
        B, C, T, V = x.size()
        tx = rearrange(x, "b c t v -> (b v) t c", t=T, v=V)
        if self.expert_weights_learnable:
            logit = self.expert_linear(tx)
            expert_weights = self.expert_softmax(logit)
            self.expert_attention = expert_weights
        else:
            expert_weights = self.expert_weights
        if self.channelDivide:
            atx = self.T_multi_expert_channels(tx, B, C, T, V, expert_weights)
        elif self.expert_weights_learnable:
            atx = self.T_multi_expert_learn(tx, B, C, T, V, expert_weights)
        elif self.expert_weights_learnable == False:
            atx = self.T_multi_expert(tx, B, C, T, V, expert_weights)
        stx = rearrange(atx, "(b v) t c -> b c t v", v=V)
        stx = self.bn2(stx)
        stx = self.drop(stx)
        rx = self.residual(tx)
        rx = rearrange(rx, "(b v) t c -> b c t v", v=V)
        stx = stx + rx
        return self.relu(stx)

    def T_multi_expert_learn(self, x, B, C, T, V, expert_weights):
        if self.temporal_merge:
            atx = torch.zeros(B * V, T // 2, C * 2)
        else:
            atx = torch.zeros(B * V, T, C)  # get experts' embedding dimension
        atx = atx.cuda(x.device)
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            if self.temporal_merge:
                expert_weights_i = ((expert_weights[:, 0::2, i] + expert_weights[0, 1::2, i]) / 2).unsqueeze(-1)
                atx += torch.mul(expert_output, expert_weights_i)
            elif self.topK == -1:
                atx += torch.mul(expert_output, expert_weights[:, :, i].unsqueeze(-1))
            elif self.topK != -1:
                expert_weights_onehot_index = torch.argmax(expert_weights, dim=-1)
                expert_weights_onehot = torch.zeros_like(expert_weights)
                expert_weights_onehot[expert_weights_onehot_index] = 1
                atx += torch.mul(expert_output, expert_weights_onehot[:, :, i].unsqueeze(-1))
        return atx

    def T_multi_expert(self, x, B, C, T, V, expert_weights):
        # x : B T C
        if self.temporal_merge:
            atx = torch.zeros(B * V, T // 2, C * 2)
        else:
            atx = torch.zeros(B * V, T, C)  # get experts' embedding dimension
        atx = atx.cuda(x.device)
        for i in range(self.num_experts):
            atx = expert_weights[i] * self.experts[i](x)
        return atx

    def T_multi_expert_channels(self, x, B, C, T, V, expert_weights):
        # x : B T C
        if self.temporal_merge:
            atx = torch.zeros(B * V, T // 2, C * 2)
        else:
            atx = torch.zeros(B * V, T, C)  # get experts' embedding dimension
        logit = self.expert_linear(x.reshape(-1, C))
        expert_weights = self.expert_softmax(logit)
        atx = atx.cuda(x.device)
        for i in range(self.num_experts):
            atx[:, :, i::self.num_experts] = expert_weights[i] * self.experts[i](x[:, :, i::self.num_experts])
        return atx

    def get_window_attention(self):
        self.window_attention = []
        for i in range(self.num_experts):
            self.window_attention.append(self.experts[i].trans.window_attentions)
        print(len(self.window_attention))
        return self.window_attention


if __name__ == '__main__':
    x = torch.zeros((1, 24, 256, 17)).to(0)
    model = MoE_temporal_module(in_channels=24, out_channels=48, temporal_merge=True).to(0)
    model_2 = MoE_temporal_module(in_channels=48, out_channels=48, num_frames=256 // 2).to(0)
    x = model(x)
    x = model_2(x)
    print("module successful!")
