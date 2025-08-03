## Our PoseFormer model was revised from https://github.com/QitaoZhao/PoseFormerV2/blob/main/common/model_poseformer.py

from ctypes import sizeof
from functools import partial
from einops import rearrange

import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm.models.layers import DropPath


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


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask: (B, N), True represents padding
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)


class PoseTransformerV2(nn.Module):
    def __init__(self, num_classes, num_joints=17, in_chans=3,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_ratio = 8    # spatial embedding dim ratio
        depth = 4   # number of transformer blocks
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        self.out_dim = num_classes    #### output dimension is num_classes
        # self.num_frame_kept = args.number_of_kept_frames
        self.num_coeff_kept = 400
        self.max_frame_len = 400
        ### spatial patch embedding
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans*num_joints, embed_dim)
        self.embed_dim = embed_dim

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.max_frame_len, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim * 2)  # Fix: doubled dimension


        # ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=self.max_frame_len, kernel_size=1)
        # self.weighted_mean_ = torch.nn.Conv1d(in_channels=self.max_frame_len, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, self.out_dim),
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def mask(self, x):
        mask = (~mask).float().unsqueeze(-1)  # True for valid, (B, T, 1)
        x = x * mask
        return x

    def _lengths_to_mask(self, lengths, max_len):
        # lengths: tensor on device, returns mask: (B, max_len), True for padding positions
        device = lengths.device
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return mask

    def Spatial_forward_features(self, x, mask):
        b, f, p, _ = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = self.Joint_embedding(x.view(b * f, p, -1))
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        # Fix: Apply mask properly without zeroing entire features
        if mask is not None:
            mask_expanded = mask.view(b * f, 1, 1).expand(-1, p, x.size(-1))
            x = x * (~mask_expanded).float()

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        return x

    def forward_features(self, x, Spatial_feature, mask=None):
        b, T, p, _ = x.shape
        if self.num_coeff_kept > T:
            num_coeff_kept = T
        else:
            num_coeff_kept = self.num_coeff_kept

        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]    # shape: (B, P, D, T_new)
        # T_new = x.shape[3]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)  # shape: (B, T_new, P*D)
        x = self.Freq_embedding(x) 
        
        Spatial_feature += self.Temporal_pos_embed[:, :T, :]
        x += self.Temporal_pos_embed_[:, :x.size(1), :]

        if x.size(1) != Spatial_feature.size(1):
            # Interpolate frequency features to match spatial features length
            x = F.interpolate(x.permute(0, 2, 1), size=Spatial_feature.size(1), mode='linear', align_corners=True).permute(0, 2, 1)
        
        x = torch.cat((x, Spatial_feature), dim=-1)  # shape: (B, T_new + F, embed_dim)

        if mask is not None:
            # Since we now have same sequence length, use original mask
            full_mask = mask  # (B, T)
        else:
            full_mask = None

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, mask=full_mask)

        x = self.Temporal_norm(x)
        return x

    def forward(self, x, lengths):
        B, T, p, _ = x.shape
        if self.num_coeff_kept > T:
            num_coeff_kept = T
        else:
            num_coeff_kept = self.num_coeff_kept
        x_ = x.clone()

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=x.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long)
            
        mask = self._lengths_to_mask(lengths, T)  # shape: (B, T)

        Spatial_feature = self.Spatial_forward_features(x, mask)
        x = self.forward_features(x_, Spatial_feature, mask=mask)

        # x2 = self.masked_weighted_mean(x[:, num_coeff_kept:], mask)
        # x = torch.cat((x1, x2), dim=-1)
        mask = (~mask).float().unsqueeze(-1)  # True for valid, (B, T, 1)
        x = x * mask
        # print("x shape before head:", x.shape)  # (B, num_coeff_kept + F, embed_dim)
        # x = torch.cat((x1, x2), dim=-1)    
        # (B, T, embed_dim*2)
        # print("x shape before head:", x.shape)
        x = self.head(x)
        # print("x shape after head:", x.shape)
        # x = x.view(B, -1, self.out_dim)
        # print("x shape before softmax:", x.shape)
        # x = self.log_softmax(x)
        # print("x shape after softmax:", x.shape)
        return self.log_softmax(x)