import ttnn
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS

from bos_metal import op

__all__ = ["ReLU", "Sigmoid", "GELU"]


@ACTIVATION_LAYERS.register_module(name="ReLU_tt")
class ReLU(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.relu(inputs, memory_config=memory_config)


@ACTIVATION_LAYERS.register_module(name="Sigmoid_tt")
class Sigmoid(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.sigmoid(inputs, memory_config=memory_config)


@ACTIVATION_LAYERS.register_module(name="GELU_tt")
class GELU(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.gelu(inputs, memory_config=memory_config)