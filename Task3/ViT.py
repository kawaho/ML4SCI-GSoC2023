import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, num_hiddens):
        super().__init__()
        self.conv = nn.Conv2d(2, num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.flatten(start_dim=2).transpose(1, 2)

class ViTBlock(nn.Module):
    def __init__(self, norm_shape, num_hiddens, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False, use_cls=True):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout, bias=use_bias, batch_first=True)
        
        self.num_heads = num_heads

        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, mlp_num_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_num_hiddens, num_hiddens),
            nn.Dropout(dropout))

        self.ln1 = nn.LayerNorm(num_hiddens)
        self.ln2 = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        X_norm = self.ln1(X)
        attn_score, attn_map = self.attention(X_norm, X_norm, X_norm)

        X = X + attn_score
        return X + self.mlp(self.ln2(X))

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, blk_dropout, use_cls=True, use_bias=False):
        super().__init__()

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embedding = PatchEmbedding(patch_size, num_hiddens)

        self.use_cls = use_cls
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_hiddens))

        if use_cls:
          self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, num_hiddens))
        else:
          self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, num_hiddens))

        self.blocks = nn.Sequential()
        for i in range(num_blks):
            self.blocks.append(ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias, use_cls))

        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),nn.Linear(num_hiddens, 1))

    def forward(self, X):
        X = self.patch_embedding(X)
        if self.use_cls:
            X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = X + self.pos_embedding
        X = self.blocks(X)
        if self.use_cls:
            X = X[:, 0]
        else:
            X = X.mean(1) 
        return self.head(X)
