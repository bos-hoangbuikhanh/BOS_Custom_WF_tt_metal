import torch
import torch.nn as nn

from bos_metal import op
"from https://github.com/Kashu7100/TokenLearner/blob/main/model.py"

class MlpBlock_Torch(nn.Module):
    """Simple MLP block with GELU activation and dropout."""
    def __init__(self, input_dim, mlp_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        # op.save_tensor(x, "pts/mlp_fc1.pt")
        x = self.gelu(x)
        # op.save_tensor(x, "pts/mlp_gelu.pt")
        x = self.fc2(x)
        # op.save_tensor(x, "pts/mlp_fc2.pt")
        return x

class TokenLearnerV11_Torch(nn.Module):
    """TokenLearner module Version 1.1 for PyTorch."""
    def __init__(self, num_tokens, in_channels, bottleneck_dim=64, dropout_rate=0.):
        super(TokenLearnerV11_Torch, self).__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.mlp = MlpBlock_Torch(input_dim=in_channels, mlp_dim=self.bottleneck_dim, output_dim=self.num_tokens, dropout_rate=self.dropout_rate)

    def forward(self, inputs, deterministic=True):
        """
        Args:
            inputs: Inputs of shape `[B, HW, C]` or `[B, C, H, W]`.

        Returns:
            [B, num_token, C]
        """
        if inputs.dim() == 4:
            n, c, h, w = inputs.shape
            inputs = inputs.view(n, c, h * w).permute(0,2,1)
        norm = self.layer_norm(inputs.permute(0,2,1)).permute(0,2,1)
        # op.save_tensor(norm, "pts/groupnorm.pt")
        selected = self.mlp(norm)
        # op.save_tensor(selected, "pts/mlpblock.pt")

        # Softmax normalization
        # selected [B, num_token, HW]
        selected = selected.view(inputs.shape[0], self.num_tokens, -1).softmax(dim=-1)
        # op.save_tensor(selected, "pts/softmax.pt")

        # Weighted sum based on the selected tokens
        # feat [B, HW, C]
        feat = inputs.view(inputs.shape[0], -1, inputs.shape[-1])
        outputs = torch.einsum('bsi,bic->bsc', selected, feat)
        # op.save_tensor(outputs, "pts/tokenlearner.pt")
        # outputs [B, num_token, C]
        return outputs, selected