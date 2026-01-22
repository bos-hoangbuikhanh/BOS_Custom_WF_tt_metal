import torch


class SELayer(torch.nn.Module):
    def __init__(self, channels,
                 act_layer=torch.nn.ReLU,
                 gate_layer=torch.nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = torch.nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = torch.nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)