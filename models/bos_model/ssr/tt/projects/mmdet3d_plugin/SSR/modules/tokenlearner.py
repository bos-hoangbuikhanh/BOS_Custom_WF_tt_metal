from bos_metal import op
from tt.projects.configs.ops_config import MyDict

import ttnn

from tt.projects.mmdet3d_plugin.SSR.utils import GELU


class MlpBlock(op.BaseModule):
    """Simple MLP block with GELU activation and dropout."""

    def __init__(self, input_dim, mlp_dim, output_dim, dropout_rate=0.1, device=None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.fc1 = op.Linear(input_dim, mlp_dim)  # 512 -> 64
        self.fc2 = op.Linear(mlp_dim, output_dim)  # 64 -> 16
        self.dropout = op.Identity()
        self.dropout2 = op.Identity()
        self.gelu = GELU()

    def forward(self, x, memory_config=MyDict(), program_config=MyDict()):
        x = self.fc1(
            x,
            memory_config=memory_config["mlp"]["fc1"].value,
            program_config=program_config["mlp"]["fc1"].value,
            # activation="gelu", # already in program_config["mlp"]["fc1"]["fused_activation"]
        )

        x = self.fc2(
            x,
            memory_config=memory_config["mlp"]["fc2"].value,
            program_config=program_config["mlp"]["fc2"].value,
        )
        return x


class TokenLearnerV11(op.BaseModule):
    def __init__(
        self,
        num_tokens: int,
        in_channels: int,
        bottleneck_dim: int = 64,
        num_out_blocks: int = 16,
        dropout_rate: float = 0.0,
        device=None,
    ):
        super(TokenLearnerV11, self).__init__()

        self.num_tokens = num_tokens
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.num_out_blocks = num_out_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = op.LayerNorm(in_channels, eps=1e-6)

        self.mlp = MlpBlock(
            input_dim=self.in_channels,
            mlp_dim=self.bottleneck_dim,
            output_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
        )
        self.device = device

    def forward(self, inputs, memory_config=None, program_config=None):
        selected = self.layer_norm(inputs, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, 1, 10_000, 512]
        selected = ttnn.to_memory_config(selected, memory_config["mlp"]["input"].value)
        selected = self.mlp(selected, memory_config=memory_config, program_config=program_config)

        selected = ttnn.sharded_to_interleaved(selected, ttnn.L1_MEMORY_CONFIG)
        selected = ttnn.reshape(selected, (inputs.shape[0], self.num_tokens, -1), memory_config=ttnn.L1_MEMORY_CONFIG)
        selected = ttnn.permute(selected, (0, 2, 1))
        selected = ttnn.softmax(selected, dim=1)
        selected = ttnn.permute(selected, (0, 2, 1))

        feat = ttnn.reshape(inputs, (inputs.shape[0], -1, inputs.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Weighted sum based on the selected tokens
        # feat [B, HW, C]
        outputs = ttnn.matmul(
            selected,
            feat,  # [1, 16, 10000] @ [1, 10000, 512] = [1, 16, 512]
        )

        return outputs

