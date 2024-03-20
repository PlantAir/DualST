import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class STnorm(nn.Module):
    def __init__(self, mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=128,eps = 1e-8):
        super(STnorm, self).__init__()

        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = eps

    def forward(self, x):
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg

        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std


        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.reshape(gate.size(0), gate.size(1), 1)
        x = x * gate

        return x
    