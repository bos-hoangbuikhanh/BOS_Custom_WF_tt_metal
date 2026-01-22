import ttnn

from .Config import CORE_GRID_16


def ResnetLinear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    output_mem_config,
    model_config,
    device,
    batch_size,
    compute_kernel_config,
):
    """
    Returns a function for linear operation in resnet with bias.
    """

    def linear_(act):
        output = ttnn.linear(
            act,
            weight.reshape(weight.shape.to_rank(4)),
            bias=bias.reshape(bias.shape.to_rank(4)),
            memory_config=output_mem_config,
            dtype=model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
            core_grid=CORE_GRID_16,
        )
        return output

    return linear_
