/*
This test will measure the speed diff btw non-masking dram cores and masking dram_cores
I'm suspecting that masking makes multicast faster. So If I multicast lots of data , we can see time diff
It should be implemented using FD

Host will write 0.75mb data to the L1, and device core (0,0) send data to the cores
*/

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <matmul_common/bmm_op.hpp>
#include <algorithm>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

void multicast_vec(std::vector<bfloat16>& a, uint32_t total_l1_size, uint32_t M, uint32_t N, IDevice* device) {
    /*
     * Setup program to execute along with its buffers and kernels to use
     * Core range is just single core
     */
    CommandQueue& cq = device->command_queue();
    Program program{};

    /*
        TEST FLOW

    1. set sharded l1 tensor to core 0,0
    2. multicast 2 others
    3. loooooooop
    4. measure time
    */

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);
    // uint32_t single_tile_size = 2 * 1024;

    //(0,0) to (4,3)
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreCoord start_core = {0, 0};
    CoreCoord core_range = {4, 3};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    // 근데 생각해보니까 자기자신을 제외하려면?
    // corerangeset을 사용하여, 차집합 사용
    CoreRangeSet all_cores_set(all_cores);
    CoreRangeSet core_to_exclude(CoreRange({0, 0}, {0, 0}));
    CoreRangeSet cores_without_0_0 = all_cores_set.subtract(core_to_exclude);

    ShardSpecBuffer shard_spec = ShardSpecBuffer(
        CoreRangeSet(start_core),
        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        ShardOrientation::ROW_MAJOR,
        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        {16, 32});

    auto shard_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = total_l1_size,
        .page_size = tt::constants::TILE_HW * sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_spec};
    auto buffer = CreateBuffer(shard_config);
    uint32_t in0_CB_size = total_l1_size;
    uint32_t src0_cb_index = CBIndex::c_0;  // 0
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
                                              .set_page_size(src0_cb_index, single_tile_size)
                                              .set_globally_allocated_address(*buffer);
    // we don't have to set cb all over the cores
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, start_core, cb_src0_config);
    auto in0_sender = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/multicast_pattern/kernel/reader_sender_in1.cpp",
        start_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto in0_receiver = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/multicast_pattern/kernel/reader_receiver_in1.cpp",
        cores_without_0_0,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
}

int main() {
    bool pass = true;

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Multicast Parameters Setup
        ////////////////////////////////////////////////////////////////////////////

        /* Create source data */
        // target for 1mb multicast
        constexpr uint32_t M = 512;   // user-defined
        constexpr uint32_t N = 1024;  // user-defined
        constexpr uint32_t K = 1;     // user-defined
        constexpr uint32_t B = 1;     // user-defined

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t total_l1_size = 512 / 32 * 1024 / 32 * single_tile_size;

        /* input vector */
        std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(total_l1_size, 1, 123, -0.4);

        multicast_vec(src0_vec, total_l1_size, M, N, device);
        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
