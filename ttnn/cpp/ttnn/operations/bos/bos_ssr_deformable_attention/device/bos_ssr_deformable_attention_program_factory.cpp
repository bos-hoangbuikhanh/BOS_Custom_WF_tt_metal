#include <algorithm>
#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"



using namespace tt::tt_metal;

namespace ttnn::operations::bos::bos_ssr_deformable_attention {
    using namespace tt::constants;
    std::uint32_t round_up_to_mul4(std::uint32_t val) { return ((val & 3) == 0) ? val : (val | 3) + 1; }
    operation::ProgramWithCallbacks bos_ssr_deformable_attention_multi_core_interleaved(
        const Tensor& value,
        const Tensor& flipped_spatial_shapes,
        const Tensor& sampling_locations,
        const Tensor& attention_weights,
        const Tensor& output,
        const std::optional<Tensor>& bilinear_weight_hash,
        const uint32_t batch_size,
        const uint32_t num_keys,
        const uint32_t num_heads,
        const uint32_t num_queries,
        const uint32_t num_levels,
        const uint32_t num_points,
        const uint32_t original_embed_dims,
        const DeviceComputeKernelConfig compute_kernel_config,
        const bool use_fp32,
        const bool is_QHB,
        const bool use_bilinear_weight_hash,
        const bool is_padded_input,
        const bool is_denormed_grid
    ){
        //** 1. Initialize program and get device */
        tt::tt_metal::Program program{};
        tt::tt_metal::IDevice* device = value.device();


        //** 2. Input, output info */
        // format
        tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value.dtype());
        tt::DataFormat location_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sampling_locations.dtype());
        tt::DataFormat weight_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(attention_weights.dtype());
        tt::DataFormat shapes_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(flipped_spatial_shapes.dtype());
        tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

        uint32_t MAX_TILES_PER_REDUCTION = (value_cb_data_format == tt::DataFormat::Float32) ? 4 : 8;
        uint32_t MAX_POINTS_PER_REDUCTION = 4;

        // size
        uint32_t value_esize = value.element_size();
        uint32_t grid_esize = sampling_locations.element_size();
        uint32_t weight_esize = attention_weights.element_size();
        uint32_t shape_esize = flipped_spatial_shapes.element_size();
        uint32_t output_esize = output.element_size();

        const auto& value_shape = value.logical_shape(); // B, num_keys, num_heads, embed_dims
        const auto& output_shape = output.logical_shape();   // B, num_queries, num_heads, embed_dims

        // buffers
        auto value_buffer = value.buffer();
        auto grid_buffer = sampling_locations.buffer();
        auto weights_buffer = attention_weights.buffer();
        auto flipped_shape_buffer = flipped_spatial_shapes.buffer();
        auto dst_buffer = output.buffer();

        // check dram
        bool value_is_dram = value_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool grid_is_dram = grid_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool weights_is_dram = weights_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool shapes_is_dram = flipped_shape_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
        bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

        // optional tensors
        ttnn::Tensor bilinear_weight_hash_;
        ttnn::Shape bilinear_weight_hash_shape;
        uint32_t bilinear_steps_x = 0, bilinear_steps_y = 0, bilinear_weight_hash_esize = 0;
        tt::tt_metal::Buffer* bilinear_weight_hash_buffer = nullptr;
        uint32_t bilinear_weight_hash_addr = 0;
        uint32_t bilinear_weight_hash_unit_size = 1;
        bool bilinear_weight_hash_is_dram = false;
        if (use_bilinear_weight_hash) {
            bilinear_weight_hash_ = bilinear_weight_hash.value();
            [[maybe_unused]] tt::DataFormat bilinear_weight_hash_cb_data_format =
                tt::tt_metal::datatype_to_dataformat_converter(bilinear_weight_hash_.dtype());
            bilinear_weight_hash_esize = bilinear_weight_hash_.element_size();
            bilinear_weight_hash_shape = bilinear_weight_hash_.logical_shape(); // 1, steps * steps * 4
            bilinear_steps_x = std::sqrt(bilinear_weight_hash_shape[1] / 4);
            bilinear_steps_y = bilinear_steps_x;
            bilinear_weight_hash_buffer = bilinear_weight_hash_.buffer();
            bilinear_weight_hash_is_dram = bilinear_weight_hash_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
            bilinear_weight_hash_unit_size = 4 * bilinear_weight_hash_esize;   // bfloat16
            bilinear_weight_hash_addr = bilinear_weight_hash_buffer->address();
        }

        //** 3. Split work to cores */
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_units = (attention_weights.logical_shape()[0] + 31) / 32;
        auto [
            num_cores, all_cores,
            core_group_1, core_group_2,
            num_units_per_core_group_1, num_units_per_core_group_2
        ] = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_units);
        auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

        //** 4. Circular Buffers Allocation */
        uint32_t next_cb_index = tt::CBIndex::c_0;
        auto CreateCB = [&program, &all_cores, &next_cb_index](
            uint32_t page_size, uint32_t num_pages, tt::DataFormat df, Buffer* buffer
        ) {
            return tt::tt_metal::create_cb(
                next_cb_index++, program, all_cores,
                page_size, num_pages, df, buffer
            );
        };

        // Constants
        const uint32_t num_total_sticks = attention_weights.logical_shape()[0];
        const uint32_t num_accum_points = attention_weights.logical_shape()[1];
        const uint32_t buffering_factor = (num_accum_points + MAX_POINTS_PER_REDUCTION - 1)/MAX_POINTS_PER_REDUCTION;
        uint32_t tile_shape = TILE_HW;

        // Value
        uint32_t in_ntiles_c = (uint32_t)std::ceil((float)value_shape[-1] / tt::constants::TILE_WIDTH);
        uint32_t out_ntiles_c = in_ntiles_c;
        uint32_t num_input_width_blocks =
                std::ceil((float)(value_shape[-1]) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH));
        uint32_t value_stick_nbytes = value_shape[-1] * value_esize;
        uint32_t value_cb_pagesize = std::min(tt::constants::TILE_WIDTH * value_esize * MAX_TILES_PER_REDUCTION, value_stick_nbytes);
        uint32_t output_stick_nbytes = output_shape[-1] * output_esize;

        //because in reader, all 4x points are reset to 0s at the same time no matter what num_accum_points is
        uint32_t value_num_pages = num_accum_points > MAX_POINTS_PER_REDUCTION ? 4*round_up_to_mul4(num_accum_points) : 4*num_accum_points;
        auto [value_cbi, value_cb] = CreateCB(value_cb_pagesize, value_num_pages*2, value_cb_data_format, nullptr);
        auto [value_zero_cbi, value_zero_cb] = CreateCB(value_cb_pagesize, 4*MAX_POINTS_PER_REDUCTION, value_cb_data_format, nullptr);

        // Grid
        uint32_t num_units_grid = tile_shape;
        uint32_t grid_unit_size = num_units_grid * grid_esize;
        auto [grid_cbi, grid_cb] = CreateCB(grid_unit_size, 1, location_cb_data_format, nullptr);
        auto [denormed_cbi, denormed_cb] = CreateCB(grid_unit_size, 1, location_cb_data_format, nullptr);
        auto [floored_grid_cbi, floored_grid_cb] = CreateCB(grid_unit_size, 1, location_cb_data_format, nullptr);
        auto [delta_cbi, delta_cb] = CreateCB(grid_unit_size, 1, location_cb_data_format, nullptr);
        auto [interm_floored_grid_cbi, interm_floored_grid_cb] = CreateCB(grid_unit_size, 1, location_cb_data_format, nullptr);

        // Weight
        uint32_t num_units_weight = tile_shape;
        uint32_t weight_unit_size = num_units_weight * weight_esize;
        auto [weight_tensor_cbi, weight_tensor_cb] = CreateCB(weight_unit_size, 1, weight_cb_data_format, nullptr);
        auto [weight_cbi, weight_cb] = CreateCB(weight_unit_size, buffering_factor, weight_cb_data_format, nullptr);

        // Spatial shapes
        uint32_t num_units_shapes = tile_shape;
        uint32_t shapes_unit_size = num_units_shapes * shape_esize;
        auto [flipped_shapes_cbi, flipped_shapes_cb] = CreateCB(shapes_unit_size, 1, shapes_cb_data_format, nullptr);

        // bilinear_weight
        uint32_t num_bilinear_weight_units = tile_shape ;  // 4 for bilinear, but because of tilize in compute kernel, use tile_shape instead
        uint32_t bilinear_weight_unit_size = num_bilinear_weight_units * weight_esize;  // assume same format with weight
        // scaler must be in the same data format with value.
        auto [scaler_cbi, scaler_cb] = CreateCB(tile_shape*value_esize, buffering_factor, value_cb_data_format, nullptr);
        auto [bilinear_weight_cbi, bilinear_weight_cb] = CreateCB(bilinear_weight_unit_size, buffering_factor, weight_cb_data_format, nullptr);
        // Output
        uint32_t num_units_out = value_shape[-1];
        uint32_t out_unit_size = num_units_out * output_esize;
        auto [out_cbi, out_cb] = CreateCB(out_unit_size, 2, output_cb_data_format, nullptr);

        //** 5. Compile-time arguments */
        std::vector<uint32_t> reader_compile_time_args;
        if (use_bilinear_weight_hash) {
            reader_compile_time_args = {
                // prepcompute CBs
                grid_cbi,
                flipped_shapes_cbi,
                // precomputed result CBs
                delta_cbi,
                floored_grid_cbi,
                // bilinear CBs
                value_cbi,
                value_zero_cbi,
                weight_cbi,
                bilinear_weight_cbi,
                //support CBs
                weight_tensor_cbi,
                // buffer addresses
                value_is_dram,
                shapes_is_dram,
                grid_is_dram,
                weights_is_dram,
                bilinear_weight_hash_is_dram,
                // sizes
                value_cb_pagesize,
                value_stick_nbytes,
                bilinear_weight_hash_unit_size,
                // metadata
                batch_size,
                num_queries,
                num_heads,
                num_keys,
                num_levels,
                num_points,
                bilinear_steps_x,
                bilinear_steps_y,
                value_num_pages,
                num_accum_points,
                num_total_sticks
            };
        } else {
            reader_compile_time_args = {
                // prepcompute CBs
                grid_cbi,
                flipped_shapes_cbi,
                // precomputed result CBs
                delta_cbi,
                floored_grid_cbi,
                // bilinear CBs
                value_cbi,
                weight_cbi,
                bilinear_weight_cbi,
                //support CBs
                weight_tensor_cbi,
                // buffer addresses
                value_is_dram,
                grid_is_dram,
                weights_is_dram,
                shapes_is_dram,
                // buffer size
                value_cb_pagesize,
                value_stick_nbytes,
                // metadata
                batch_size,
                num_queries,
                num_heads,
                num_keys,
                num_levels,
                num_points,
                value_num_pages,
                num_accum_points,
                num_total_sticks
            };
        }

        std::vector<uint32_t> writer_compile_time_args;
        if (use_bilinear_weight_hash) {
            uint32_t bilinear_page_size = round_up_to_mul32(bilinear_weight_hash_shape[1]*bilinear_weight_hash_esize);
            auto [tmp_bilinear_weight_cbi, tmp_bilinear_weight_cb] = CreateCB(bilinear_page_size, 1, weight_cb_data_format, nullptr);
            writer_compile_time_args = {
                // prepcompute CBs
                grid_cbi,
                flipped_shapes_cbi,
                // precomputed result CBs
                delta_cbi,
                floored_grid_cbi,
                // bilinear CBs
                value_cbi,
                value_zero_cbi,
                weight_cbi,
                bilinear_weight_cbi,
                tmp_bilinear_weight_cbi,
                //support CBs
                weight_tensor_cbi,
                // output CBs
                out_cbi,
                // buffer addresses
                value_is_dram,
                shapes_is_dram,
                grid_is_dram,
                weights_is_dram,
                bilinear_weight_hash_is_dram,
                dst_is_dram,
                // sizes
                value_cb_pagesize,
                value_stick_nbytes,
                bilinear_weight_hash_unit_size,
                num_input_width_blocks,
                output_stick_nbytes,
                // metadata
                batch_size,
                num_queries,
                num_heads,
                num_keys,
                num_levels,
                num_points,
                bilinear_steps_x,
                bilinear_steps_y,
                num_accum_points,
                num_total_sticks
            };
        } else {
            writer_compile_time_args = {
                out_cbi,
                dst_is_dram,
                //metadata
                value_cb_pagesize,
                num_input_width_blocks,
                output_stick_nbytes,
                batch_size,
                num_queries,
                num_heads,
                num_total_sticks
            };
        }

        std::vector<uint32_t> compute_compile_time_args = {
            // prepcompute CBs
            grid_cbi,
            flipped_shapes_cbi,
            //interm CBs
            interm_floored_grid_cbi,
            // precomputed result CBs
            denormed_cbi,
            floored_grid_cbi,
            delta_cbi,
            // bilinear
            value_cbi,
            weight_cbi,
            bilinear_weight_cbi,
            scaler_cbi,
            // bilineared result CB
            out_cbi,
            // metadata
            in_ntiles_c,
            out_ntiles_c,
            num_input_width_blocks,
            batch_size,
            bilinear_steps_x,
            bilinear_steps_y,
            num_accum_points,
            num_total_sticks
        };

        // defines
        std::map<std::string, std::string> reader_defines, writer_defines, compute_defines;
        // compute
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(value.device()->arch(), compute_kernel_config);
        auto reduce_op = ReduceOpMath::SUM;
        auto reduce_dim = ReduceOpDim::H;
        compute_defines = reduce_op_utils::get_defines(reduce_op, reduce_dim);
        if (use_fp32) {
            reader_defines["use_fp32"] = "1";
            writer_defines["use_fp32"] = "1";
            compute_defines["FLOOR_TILE"] = "floor_tile_float32";
        }
        else{
            compute_defines["FLOOR_TILE"] = "floor_tile";
        }
        if (use_bilinear_weight_hash) {
            reader_defines["use_bilinear_weight_hash"] = "1";
            writer_defines["use_bilinear_weight_hash"] = "1";
            compute_defines["use_bilinear_weight_hash"] = "1";
        }
        if (is_padded_input) {
            writer_defines["is_padded_input"] = "1";
        }
        if (is_denormed_grid) {
            compute_defines["is_denormed_grid"] = "1";
        }
        if (is_QHB) {
            reader_defines["is_QHB"] = "1";
            writer_defines["is_QHB"] = "1";
            compute_defines["is_QHB"] = "1";
        }
        // readers
        std::string reader_kernel_path = use_bilinear_weight_hash ?
            "ttnn/cpp/ttnn/operations/bos/bos_ssr_deformable_attention/device/kernels/dataflow/bos_ssr_reader_deformable_attention_hash_interleaved.cpp" :
            "ttnn/cpp/ttnn/operations/bos/bos_ssr_deformable_attention/device/kernels/dataflow/bos_ssr_reader_deformable_attention_interleaved.cpp";
        tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            reader_kernel_path,
            all_cores,
            ReaderDataMovementConfig(reader_compile_time_args, reader_defines)
        );

        // writers
        std::string writer_kernel_path = use_bilinear_weight_hash ?
            "ttnn/cpp/ttnn/operations/bos/bos_ssr_deformable_attention/device/kernels/dataflow/bos_ssr_writer_deformable_attention_hash_interleaved.cpp" :
            "ttnn/cpp/ttnn/operations/bos/bos_ssr_deformable_attention/device/kernels/dataflow/bos_ssr_writer_deformable_attention_interleaved.cpp";
        tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            writer_kernel_path,
            all_cores,
            WriterDataMovementConfig(writer_compile_time_args, writer_defines)
        );

        auto compute_config = ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines};
        std::string compute_kernel_path = "ttnn/cpp/ttnn/operations/bos/bos_ssr_deformable_attention/device/kernels/compute/bos_ssr_compute_deformable_attention_tile_interleaved.cpp";
        KernelHandle compute_kernel_id = CreateKernel(
            program,
            compute_kernel_path,
            all_cores,
            compute_config);

        //** 6. Runtime arguments */
        uint32_t start_id = 0;
        for (uint32_t i = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores.at(i);
            uint32_t n_units_per_core = i < core_group_1.num_cores() ?
                                            num_units_per_core_group_1 : num_units_per_core_group_2;

            std::vector<uint32_t> reader_runtime_args;
            if (use_bilinear_weight_hash) {
                reader_runtime_args = {
                    value_buffer->address(),
                    grid_buffer->address(),
                    flipped_shape_buffer->address(),
                    weights_buffer->address(),
                    bilinear_weight_hash_addr,
                    start_id,
                    n_units_per_core,
                };
            } else {
                reader_runtime_args = {
                    value_buffer->address(),
                    grid_buffer->address(),
                    weights_buffer->address(),
                    flipped_shape_buffer->address(),
                    start_id,
                    n_units_per_core
                };
            }

            std::vector<uint32_t> writer_runtime_args;
            if (use_bilinear_weight_hash) {
                writer_runtime_args = {
                    value_buffer->address(),
                    grid_buffer->address(),
                    flipped_shape_buffer->address(),
                    weights_buffer->address(),
                    bilinear_weight_hash_addr,
                    dst_buffer->address(),
                    start_id,
                    n_units_per_core,
                };
            } else {
                writer_runtime_args = {
                    dst_buffer->address(),
                    start_id,
                    n_units_per_core
                };
            };

            std::vector<uint32_t> compute_runtime_args = {
                start_id,
                n_units_per_core,
            };

            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
            start_id += n_units_per_core;
        }

        auto override_runtime_args_callback = [reader_kernel_id, compute_kernel_id, writer_kernel_id, cores, core_group_1, num_units_per_core_group_1, num_units_per_core_group_2, use_bilinear_weight_hash] (
                                                    const void* operation,
                                                    const Program& program,
                                                    const std::vector<Tensor>& input_tensors,
                                                    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                    const std::vector<Tensor>& output_tensors) {
                auto value_buffer = input_tensors.at(0).buffer();
                auto flipped_shapes_buffer = input_tensors.at(1).buffer();
                auto grid_buffer = input_tensors.at(2).buffer();
                auto weight_buffer = input_tensors.at(3).buffer();
                auto dst_buffer = output_tensors.at(0).buffer();

                // Optional inputs
                uint32_t bilinear_weight_hash_addr = 0;
                if (use_bilinear_weight_hash) {
                    auto bilinear_weight_hash_buffer = optional_input_tensors.at(0).value().buffer();
                    bilinear_weight_hash_addr = bilinear_weight_hash_buffer->address();
                }

                uint32_t start_id = 0;
                for (uint32_t i = 0; i < cores.size(); i++){
                    auto n_units_per_core = i < core_group_1.num_cores() ? num_units_per_core_group_1 : num_units_per_core_group_2;

                    // Reader
                    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
                    if (use_bilinear_weight_hash) {
                        reader_runtime_args[0] = value_buffer->address();
                        reader_runtime_args[1] = grid_buffer->address();
                        reader_runtime_args[2] = flipped_shapes_buffer->address();
                        reader_runtime_args[3] = weight_buffer->address();
                        reader_runtime_args[4] = bilinear_weight_hash_addr;
                        reader_runtime_args[5] = start_id;
                        reader_runtime_args[6] = n_units_per_core;
                    } else {
                        reader_runtime_args[0] = value_buffer->address();
                        reader_runtime_args[1] = grid_buffer->address();
                        reader_runtime_args[2] = weight_buffer->address();
                        reader_runtime_args[3] = flipped_shapes_buffer->address();
                        reader_runtime_args[4] = start_id;
                        reader_runtime_args[5] = n_units_per_core;
                    }

                    // Compute
                    auto& compute_runtime_args = GetRuntimeArgs(program, compute_kernel_id, cores[i]);
                    compute_runtime_args[0] = start_id;
                    compute_runtime_args[1] = n_units_per_core;

                    // Writer
                    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
                    if (use_bilinear_weight_hash) {
                        writer_runtime_args[0] = value_buffer->address();
                        writer_runtime_args[1] = grid_buffer->address();
                        writer_runtime_args[2] = flipped_shapes_buffer->address();
                        writer_runtime_args[3] = weight_buffer->address();
                        writer_runtime_args[4] = bilinear_weight_hash_addr;
                        writer_runtime_args[5] = dst_buffer->address();
                        writer_runtime_args[6] = start_id;
                        writer_runtime_args[7] = n_units_per_core;
                    }
                    else {
                        writer_runtime_args[0] = dst_buffer->address();
                        writer_runtime_args[1] = start_id;
                        writer_runtime_args[2] = n_units_per_core;
                    }
                    start_id += n_units_per_core;
                }
            };

        return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
    }
}
