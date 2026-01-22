// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_metal.hpp>

/*
* 1. Host writes data to buffer in DRAM
* 2. dram_copy kernel on logical core {0, 0} BRISC copies data from buffer
*      in step 1. to buffer in L1 and back to another buffer in DRAM
* 3. Host reads from buffer written to in step 2.
*/

using namespace tt::tt_metal;

int main(int argc, char **argv) {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = false;

    try {
        /*
        * Silicon accelerator setup
        */
        constexpr int device_id = 0;
        IDevice *device =
            CreateDevice(device_id);

        /*
        * Setup program and command queue to execute along with its buffers and kernels to use
        */

        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/hugepage/kernels/pcie_host_rw.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0}
        );


        // void *host_hugepage_start = (void *)tt::Cluster::instance().host_dma_address(0, device_id, 0);
        // uint64_t host_hugepage_physic = tt::Cluster::instance().host_dma_physical_address(0, device_id, 0);

     //   tt::log_info(tt::LogTest, "Huge page address (virt): {}", (uint64_t) host_hugepage_start);
        // tt::log_info(tt::LogTest, "Huge page address (phys): {}", (uint64_t) host_hugepage_physic);

        // constexpr uint32_t data_size = 512;
        // uint32_t l1_buffer_addr0 = 400 * 1024;
        // uint32_t l1_buffer_addr1 = l1_buffer_addr0 + data_size;
        // uint32_t host_dram_addr = 0;
        // uint32_t test_value = 0xAB;

        // uint32_t* huge_ptr = static_cast<uint32_t*>(host_hugepage_start);
        // for (int i = 0; i < (data_size/sizeof(uint32_t)); ++i) {
        //     huge_ptr[i+host_dram_addr] = 0x0;
        // }

        /*
        * Create input data and runtime arguments, then execute
        */
        //std::vector<uint32_t> input_vec = create_constant_vector_of_bfloat16(data_size, 0);

        // const std::vector<uint32_t> runtime_args = {
        //     l1_buffer_addr0,
        //     l1_buffer_addr1,
        //     host_dram_addr,
        //     data_size,
        //     test_value
        // };

        // SetRuntimeArgs(
        //     program,
        //     dram_copy_kernel_id,
        //     core,
        //     runtime_args
        // );
        // //kmd code for DMA configuration

        // tt::tt_metal::detail::LaunchProgram(device, program);

        /*
        * Validation & Teardown
        */

        // for (int i = 0; i < (data_size/sizeof(uint32_t)); ++i) {
        //     if(huge_ptr[i+host_dram_addr] != test_value) pass=false;
        // }

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        // tt::log_error(tt::LogTest, "Test failed with exception!");
        // tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        // tt::log_info(tt::LogTest, "Test Passed");
    } else {
        // TT_THROW("Test Failed, please remake this");
    }

    return 0;
}
