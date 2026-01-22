add_library(ttnn_op_bos ${LIB_TYPE})
add_library(TTNN::Ops::Bos ALIAS ttnn_op_bos)

target_precompile_headers(ttnn_op_bos REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_bos)
set_target_properties(
    ttnn_op_bos
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    bos_grid_sample/device/kernels/*
    bos_deformable_attention/device/kernels/*
    bos_non_zero_indices/device/kernels/*
    bos_sharded/device/kernels/*
    bos_sharded/bos_reshard/device/kernels/*
    bos_setitem/device/kernels/*
    bos_getitem/device/bos_getitem_kernels/*
)
target_sources(
    ttnn_op_bos
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        # Grid Sample
        bos_grid_sample/device/bos_grid_sample_op.cpp
        bos_grid_sample/device/bos_grid_sample_program_factory.cpp
        bos_grid_sample/bos_grid_sample.cpp
        # Deformable Attention
        bos_deformable_attention/device/bos_deformable_attention_op.cpp
        bos_deformable_attention/device/bos_deformable_attention_program_factory.cpp
        bos_deformable_attention/bos_deformable_attention.cpp
        # Non Zero Indices
        bos_non_zero_indices/device/bos_non_zero_indices_op.cpp
        bos_non_zero_indices/device/bos_non_zero_indices_program_factory.cpp
        bos_non_zero_indices/bos_non_zero_indices.cpp
        # Reshard
        bos_sharded/bos_reshard/device/bos_reshard_op.cpp
        bos_sharded/bos_reshard/device/bos_reshard_program_factory.cpp
        bos_sharded/bos_reshard/device/bos_nd_reshard_program_factory.cpp
        bos_sharded/bos_reshard/bos_reshard.cpp
        bos_sharded/bos_sharded_common.cpp
        # Setitem
        bos_setitem/bos_setitem.cpp
        bos_setitem/device/bos_setitem_op.cpp
        bos_setitem/device/bos_setitem_program_factory.cpp
        # Getitem
        bos_getitem/bos_getitem.cpp
        bos_getitem/device/bos_getitem_device_operation.cpp
        bos_getitem/device/bos_getitem_rm_factory.cpp
        # Bos ssr deformable attention
        bos_ssr_deformable_attention/device/bos_ssr_deformable_attention_op.cpp
        bos_ssr_deformable_attention/device/bos_ssr_deformable_attention_program_factory.cpp
        bos_ssr_deformable_attention/bos_ssr_deformable_attention.cpp
)

target_include_directories(
    ttnn_op_bos 
    PRIVATE 
        ${FixmeOpIncDirs}
)
target_link_libraries(
    ttnn_op_bos
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_bos
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/bos
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_bos LIBRARY COMPONENT tar)
