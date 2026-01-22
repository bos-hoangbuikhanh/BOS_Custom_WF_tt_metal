#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

#include "ttnn-pybind/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
// #include "cpp/pybind11/json_class.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/bos/bos_deformable_attention/bos_deformable_attention.hpp"
#include "ttnn/operations/bos/bos_deformable_attention/bos_deformable_attention_pybind.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::bos::bos_deformable_attention {
    void bind_bos_deformable_attention_operation(py::module& module) {
        bind_registered_operation(
            module,
            ttnn::bos_deformable_attention,
            R"doc(
            Multiscale Deformable Attention

            Implements the multiscale deformable attention mechanism, efficiently aggregating information from multiple spatial locations across different feature levels.

            Args:
                value (Tensor): The input value tensor containing features to be attended.
                value_spatial_shapes (Tensor): Tensor describing the spatial shapes of each feature level.
                sampling_locations (Tensor): Tensor specifying the sampling locations for attention.
                attention_weights (Tensor): Weights for each sampling location.
                use_fp32 (bool, optional): If True, attention_weights and sampling_locations are typecast to fp32 before computation to preserve high PCC. The output dtype will still follow the input value dtype.
                memory_config (MemoryConfig, optional): Specifies the memory configuration for the output tensor.
                compute_kernel_config (ComputeKernelConfig, optional): Configuration for the compute kernel. When use_fp32=True, set kernel to HiFi4. fp32_dest_accum is always set to True for this operation.

            Returns:
                Tensor: Output tensor after applying multiscale deformable attention.

            Example:
                out_tt = ttnn.bos_deformable_attention(
                    value_tt,
                    value_spatial_shapes_tt,
                    sampling_locations_tt,
                    attention_weights_tt,
                    use_fp32=True
                )

            Notes:
                - When use_fp32=True, only attention_weights and sampling_locations are cast to fp32. The output tensor's dtype matches the input value tensor.
                - For best accuracy (high PCC), enable use_fp32 and ensure compute_kernel_config uses HiFi4 with fp32_dest_accum=True.
                - memory_config and compute_kernel_config allow fine-tuning of performance and precision for deployment scenarios.
            )doc",
            ttnn::pybind_overload_t{
                [](decltype(ttnn::bos_deformable_attention)& self,
                   const ttnn::Tensor& input,
                   const Tensor& spatial_shapes,
                   const Tensor& sampling_locations,
                   const Tensor& attention_weights,
                   const bool use_fp32,
                   const bool is_denormed_grid,
                   const std::optional<Tensor> bilinear_weight_hash,
                   const std::optional<const MemoryConfig>& memory_config,
                   const std::optional<Tensor> optional_output_tensor,
                   const std::optional<DeviceComputeKernelConfig>& compute_kernel_config
                ) -> ttnn::Tensor {
                    return self(
                        input,
                        spatial_shapes,
                        sampling_locations,
                        attention_weights,
                        use_fp32,
                        is_denormed_grid,
                        bilinear_weight_hash,
                        memory_config,
                        optional_output_tensor,
                        compute_kernel_config
                    );
                },
                py::arg("input"),
                py::arg("spatial_shapes"),
                py::arg("sampling_locations"),
                py::arg("attention_weights"),
                py::kw_only(),
                py::arg("use_fp32") = false,
                py::arg("is_denormed_grid") = false,
                py::arg("bilinear_weight_hash") = std::nullopt,
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("compute_kernel_config") = std::nullopt
            }
        );
    }
}
