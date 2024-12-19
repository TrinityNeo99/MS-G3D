#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import torch
import torch.nn as nn
from thop import profile


# 定义标准 Transformer
class StandardTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(StandardTransformer, self).__init__()
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


# 定义滑动窗口 Transformer
class SlidingWindowTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length, window_size):
        super(SlidingWindowTransformer, self).__init__()
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.window_size = window_size
        self.transformer_encoder = nn.ModuleList(
            [encoder_layer for _ in range(num_encoder_layers)]
        )

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = []
        for i in range(0, src.size(0), self.window_size):
            chunk = src[i:i + self.window_size]
            for layer in self.transformer_encoder:
                chunk = layer(chunk)
            output.append(chunk)
            break
        return torch.cat(output, dim=0)


# 参数
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_length = 1024
window_size = 16

# 创建模型
standard_model = StandardTransformer(d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length)
sliding_window_model = SlidingWindowTransformer(d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length,
                                                window_size)

# 随机输入
src = torch.randint(0, max_seq_length, (max_seq_length, 1))

# 计算参数量和 FLOPs（标准 Transformer）
standard_model_params = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)
standard_flops, _ = profile(standard_model, inputs=(src,))
print(f"Standard Transformer 参数量: {standard_model_params: e}")
print(f"Standard Transformer FLOPs: {standard_flops: e}")

# 计算参数量和 FLOPs（滑动窗口 Transformer）
sliding_window_model_params = sum(p.numel() for p in sliding_window_model.parameters() if p.requires_grad)
sliding_window_flops, _ = profile(sliding_window_model, inputs=(src,))
print(f"Sliding Window Transformer 参数量: {sliding_window_model_params: e}")
print(f"Sliding Window Transformer FLOPs: {sliding_window_flops: e}")
