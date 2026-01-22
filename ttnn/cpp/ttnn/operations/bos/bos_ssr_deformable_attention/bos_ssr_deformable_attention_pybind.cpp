#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

#include "ttnn-pybind/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
// #include "cpp/pybind11/json_class.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/bos/bos_ssr_deformable_attention/bos_ssr_deformable_attention.hpp"
#include "ttnn/operations/bos/bos_ssr_deformable_attention/bos_ssr_deformable_attention_pybind.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::bos::bos_ssr_deformable_attention {
    void bind_bos_ssr_deformable_attention_operation(py::module& module) {
        bind_registered_operation(
            module,
            ttnn::bos_ssr_deformable_attention,
            R"doc(
            Multiscale Deformable Attention (BOS SSR)

            Implements a multiscale deformable attention mechanism that efficiently aggregates information 
            from multiple spatial locations across different feature levels.

            **Args:**
                value (Tensor): Input tensor containing the features to attend.
                value_spatial_shapes (Tensor): Tensor describing the spatial dimensions of each feature level.
                sampling_locations (Tensor): Tensor specifying the sampling positions used for attention.
                attention_weights (Tensor): Attention weights corresponding to each sampling location.
                use_fp32 (bool, optional): 
                    If True, `attention_weights` and `sampling_locations` are cast to fp32 before computation 
                    to preserve numerical precision (high PCC). The output dtype remains the same as the input `value` dtype.
                memory_config (MemoryConfig, optional): 
                    Configuration for the memory allocation of the output tensor.
                compute_kernel_config (ComputeKernelConfig, optional): 
                    Kernel configuration for computation. When `use_fp32=True`, the HiFi4 kernel is recommended, 
                    with `fp32_dest_accum=True` automatically enabled for improved precision.
                num_queries (int, optional): Number of query elements (used for shape inference and performance tuning).
                num_levels (int, optional): Number of feature levels used in attention.
                num_points (int, optional): Number of sampling points per query.
                is_QHB (bool, optional): 
                    Internal flag indicating whether the sampling_locations and attention_weights tensors follow the QHB layout.

            **Returns:**
                Tensor: Output tensor after applying the multiscale deformable attention mechanism.

            **Example:**
                ```python
                out_tt = ttnn.bos_deformable_attention(
                    value_tt,
                    value_spatial_shapes_tt,
                    sampling_locations_tt,
                    attention_weights_tt,
                    use_fp32=True,
                    num_queries=num_queries,
                    num_levels=num_levels,
                    num_points=num_points,
                    is_QHB=False
                )
                ```

            **Notes:**
                - When using `is_QHB=True`, ensure that `sampling_locations` and `attention_weights` tensors are in QHB format.
                - When `use_fp32=True`, only `attention_weights` and `sampling_locations` are cast to fp32. 
                The output dtype matches the input `value` tensor.
                - For best accuracy (high PCC), enable `use_fp32` and configure `compute_kernel_config` 
                with HiFi4 and `fp32_dest_accum=True`.
                - `memory_config` and `compute_kernel_config` provide advanced control over performance and precision 
                for deployment scenarios.
            )doc",
            ttnn::pybind_overload_t{
                [](decltype(ttnn::bos_ssr_deformable_attention)& self,
                   const ttnn::Tensor& input,
                   const Tensor& spatial_shapes,
                   const Tensor& sampling_locations,
                   const Tensor& attention_weights,
                   const bool use_fp32,
                   const bool is_denormed_grid,
                   const std::optional<Tensor> bilinear_weight_hash,
                   const std::optional<const MemoryConfig>& memory_config,
                   const std::optional<Tensor> optional_output_tensor,
                   const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
                   const std::optional<uint32_t> num_queries,
                    const std::optional<uint32_t> num_levels,
                    const std::optional<uint32_t> num_points,
                    const std::optional<bool> is_QHB
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
                        compute_kernel_config,
                        num_queries,
                        num_levels,
                        num_points,
                        is_QHB
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
                py::arg("compute_kernel_config") = std::nullopt,
                py::arg("num_queries") = std::nullopt,
                py::arg("num_levels") = std::nullopt,
                py::arg("num_points") = std::nullopt,
                py::arg("is_QHB") = std::nullopt
            }
        );
    }
}
