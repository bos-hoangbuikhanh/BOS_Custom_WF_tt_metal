#include "bos_grid_sample_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "bos_grid_sample.hpp"

namespace ttnn::operations::bos::bos_grid_sample {

void bind_bos_grid_sample_operation(py::module& module) {
    const auto doc = R"doc( dcm)doc";
    bind_registered_operation(
        module,
        ttnn::bos_grid_sample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("grid"),
            py::kw_only(),
            py::arg("align_corners") = false,
            py::arg("mode") = "bilinear",
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::bos::bos_grid_sample
