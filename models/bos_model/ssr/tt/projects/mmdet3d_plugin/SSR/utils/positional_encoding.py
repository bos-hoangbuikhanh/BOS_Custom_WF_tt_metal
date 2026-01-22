# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Nhat Nguyen
# ---------------------------------------------
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING

from bos_metal import op
import ttnn


@POSITIONAL_ENCODING.register_module(name="LearnedPositionalEncoding_tt")
class LearnedPositionalEncoding(op.Operation):
    """Position embedding with learnable embedding weights. Mimicked from MMDet."""

    def __init__(
        self,
        num_feats,
        row_num_embed=50,
        col_num_embed=50,
        embed_cfg=op.EmbeddingConfig(),
        *,
        device=None,
        init_cfg=None,
        **kwargs,
    ):
        super(LearnedPositionalEncoding, self).__init__(
            device=device, init_config=init_cfg, **kwargs
        )
        self.embed_cfg = op.EmbeddingConfig.build(embed_cfg)

        self.row_embed = op.Embedding(
            row_num_embed, num_feats, device=self.device, config=self.embed_cfg
        )
        self.col_embed = op.Embedding(
            col_num_embed, num_feats, device=self.device, config=self.embed_cfg
        )

        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def __preprocess__(self, *inputs, **kwargs):
        return self._as_tuple(inputs)

    def forward(self, mask_shape, deallocate=True):
        """Forward function for `LearnedPositionalEncoding`.

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        bs, h, w = mask_shape
        x = ttnn.arange(0, w, device=self.device)
        y = ttnn.arange(0, h, device=self.device)

        x_embed = self.col_embed(x, deallocate=deallocate)  # 1, w, num_feats
        y_embed = self.row_embed(y, deallocate=deallocate)  # 1, h, num_feats

        x_embed = ttnn.repeat_interleave(
            ttnn.repeat_interleave(x_embed, h // 2, 0), 2, 0
        )

        y_embed = ttnn.repeat_interleave(
            ttnn.repeat_interleave(ttnn.permute(y_embed, (1, 0, 2)), w // 2, 1), 2, 1
        )

        pos = ttnn.concat([x_embed, y_embed], dim=-1)  # h, w, num_feats

        pos = ttnn.permute(pos, (2, 0, 1))
        pos = ttnn.repeat_interleave(ttnn.unsqueeze(pos, 0), bs, 0)

        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"row_num_embed={self.row_num_embed}, "
        repr_str += f"col_num_embed={self.col_num_embed})"
        return repr_str