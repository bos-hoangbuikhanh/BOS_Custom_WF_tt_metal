from bos_metal import op
from test.common import ReLU, Sigmoid, MyDict


class SELayer(op.BaseModule):
    def __init__(self, channels, act_layer=ReLU, gate_layer=Sigmoid):
        super().__init__()
        self.mlp_reduce = op.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = op.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se,
                memory_config=MyDict(), program_config=MyDict()):
        x_se = self.mlp_reduce(
            x_se,
            memory_config=memory_config["mlp_reduce"].value,
            program_config=program_config["mlp_reduce"].value,
        )
        x_se = self.act1(x_se)

        x_se = self.mlp_expand(
            x_se,
            memory_config=memory_config["mlp_expand"].value,
            program_config=program_config["mlp_expland"].value,
        )

        return x * self.gate(x_se)