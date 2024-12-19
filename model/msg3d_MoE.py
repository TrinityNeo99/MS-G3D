#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import sys

sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory
from model.MoE_temporal_module import MoE_temporal_module as MoE_Trans
from model.angular_feature import *
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
import sys

sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory
from model.MoE_temporal_module import MoE_temporal_module as MoE_Trans

torch.autograd.set_detect_anomaly(True)
from einops import rearrange


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 num_frame=256,
                 dropout=0.1,
                 angularType="none",
                 expert_windows_size=[8, 32], isThreeLayer=False):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.angularType = angularType
        self.angular_feature = Angular_feature()
        self.isThreeLayer = isThreeLayer

        # channels
        c1 = 48
        c2 = c1 * 2  # 192
        c3 = c2 * 2  # 384
        c4 = c3 * 2

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            # MS_TCN(c1, c1),
            # MoE_Trans(c1, c1, num_frames=num_frame, expert_windows_size=expert_windows_size),
            MoE_Trans(c1, c1, num_frames=num_frame, expert_windows_size=expert_windows_size),
        )
        self.sgcn1[-1].act = nn.Identity()
        # self.tcn1 = MS_TCN(c1, c1)
        self.tcn1 = MoE_Trans(c1, c1, dropout=dropout, num_frames=num_frame, expert_windows_size=expert_windows_size)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            # MS_TCN(c1, c2, stride=2),
            # MS_TCN(c2, c2),
            MoE_Trans(c1, c2, temporal_merge=True, num_frames=num_frame, expert_windows_size=expert_windows_size),
            # MoE_Trans(c2, c2),
            MoE_Trans(c2, c2, num_frames=num_frame // 2, expert_windows_size=expert_windows_size),
        )
        self.sgcn2[-1].act = nn.Identity()
        # self.tcn2 = MS_TCN(c2, c2)
        self.tcn2 = MoE_Trans(c2, c2, num_frames=num_frame // 2, dropout=dropout,
                              expert_windows_size=expert_windows_size)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            # MS_TCN(c2, c3, stride=2),
            # MS_TCN(c3, c3),
            MoE_Trans(c2, c3, temporal_merge=True, num_frames=num_frame // 2, expert_windows_size=expert_windows_size),
            # MoE_Trans(c3, c3),
            MoE_Trans(c3, c3, num_frames=num_frame // 4, expert_windows_size=expert_windows_size)
        )
        self.sgcn3[-1].act = nn.Identity()
        if isThreeLayer:
            self.tcn3 = MS_TCN(c3, c3)
        else:
            self.tcn3 = MoE_Trans(c3, c3, num_frames=num_frame // 4, dropout=dropout,
                                  expert_windows_size=expert_windows_size)
        if not isThreeLayer:
            self.gcn3d4 = MultiWindow_MS_G3D(c3, c4, A_binary, num_g3d_scales, window_stride=2)
            self.sgcn4 = nn.Sequential(
                MS_GCN(num_gcn_scales, c3, c3, A_binary, disentangled_agg=True),
                # MS_TCN(c2, c3, stride=2),
                # MS_TCN(c3, c3),
                MoE_Trans(c3, c4, temporal_merge=True, num_frames=num_frame // 4,
                          expert_windows_size=expert_windows_size),
                # MoE_Trans(c3, c3),
                MoE_Trans(c4, c4, num_frames=num_frame // 8, expert_windows_size=expert_windows_size)
            )
            self.sgcn4[-1].act = nn.Identity()
            self.tcn4 = MS_TCN(c4, c4)
            # self.tcn4 = MoE_Trans(c4, c4, num_frames=num_frame // 8, dropout=dropout)
        if isThreeLayer:
            self.fc = nn.Linear(c3, num_class)
        else:
            self.fc = nn.Linear(c4, num_class)

    def forward(self, x):
        if self.angularType == "p2a":
            x = self.angular_feature.preprocessing_pingpong_coco(
                x)  # add 9 channels with original 3 channels, total 12 channels, all = 12

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        # x = F.relu(self.sgcn1(x), inplace=False)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        # x = F.relu(self.sgcn2(x), inplace=False)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        # x = F.relu(self.sgcn3(x), inplace=False)
        x = self.tcn3(x)

        if not self.isThreeLayer:
            # extra-layer
            x = F.relu(self.sgcn4(x) + self.gcn3d4(x), inplace=True)
            # x = F.relu(self.sgcn3(x), inplace=False)
            x = self.tcn4(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)  # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)  # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out

    def expert_attention(self):
        layer1, layer2, layer3 = {}, {}, {}
        layer1['block1'] = self.sgcn1[2].expert_attention[0].cpu()
        layer2['block1'] = self.sgcn2[1].expert_attention[0].cpu()
        layer2['block2'] = self.sgcn2[2].expert_attention[0].cpu()
        layer3['block1'] = self.sgcn3[1].expert_attention[0].cpu()
        layer3['block2'] = self.sgcn3[2].expert_attention[0].cpu()
        expert_attention = {"layer1": layer1, "layer2": layer2, "layer3": layer3}
        return expert_attention

    def window_attention(self):
        window_attention = self.sgcn1[2].get_window_attention()
        for e in window_attention:
            for l in e:
                # print("window attention shape: ", l.shape)
                pass
        ep1 = window_attention[0][0]
        ep2 = window_attention[1][1]

        ep1 = self.average_joint_attention(ep1)
        ep2 = self.average_joint_attention(ep2)

        # print(ep1.shape)
        # print(ep2.shape)
        ep1 = ep1.numpy()
        ep2 = ep2.numpy()
        return ep1, ep2

    def average_joint_attention(self, data):
        # data  (num, c, ws, ws)
        joint_num = 17
        window_num = data.shape[0] // joint_num
        data = rearrange(data, "(n1 n2) c ws ws1 -> n1 n2 c ws ws1", n1=17, n2=window_num)
        data = torch.mean(data, dim=0)  # n2 c ws ws1
        data = torch.mean(data, dim=1)
        return data


if __name__ == "__main__":
    # For debugging purposes
    import sys

    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))
