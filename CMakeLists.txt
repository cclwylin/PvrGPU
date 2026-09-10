cmake_minimum_required(VERSION 3.20)

project(PvrGPUControlPlane VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(PVRGPU_ENABLE_DIAGNOSTICS
    "Compile optional model traces and diagnostic dumps (not correctness checks)" ON)
add_compile_definitions(
    PVRGPU_ENABLE_DIAGNOSTICS=$<BOOL:${PVRGPU_ENABLE_DIAGNOSTICS}>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(build_config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${build_config}" build_config_upper)
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${build_config_upper}"
        "${CMAKE_BINARY_DIR}/bin")
    set("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${build_config_upper}"
        "${CMAKE_BINARY_DIR}/lib")
endforeach()

find_package(PkgConfig REQUIRED)
pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
pkg_check_modules(LIBPNG REQUIRED IMPORTED_TARGET libpng>=1.6)

function(pvrgpu_target_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

if(UNIX)
    add_executable(model-diagnostics-test tests/model_diagnostics_test.cpp)
    target_include_directories(model-diagnostics-test PRIVATE
        "${CMAKE_SOURCE_DIR}/src/systemc")
    pvrgpu_target_warnings(model-diagnostics-test)
endif()

add_library(pvrgpu-rdc-runner-support STATIC
    src/rdc_runner/native_report.cpp
    src/rdc_runner/process.cpp
    src/rdc_runner/replay_range_audit.cpp
    src/rdc_runner/runtime_config.cpp
    src/rdc_runner/sha256.cpp
)
target_include_directories(pvrgpu-rdc-runner-support PUBLIC
    "${CMAKE_SOURCE_DIR}/src"
)
target_compile_definitions(pvrgpu-rdc-runner-support PRIVATE
    PVRGPU_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
pvrgpu_target_warnings(pvrgpu-rdc-runner-support)

add_executable(llvmpipe src/rdc_runner/main.cpp)
target_compile_definitions(llvmpipe PRIVATE PVRGPU_RDC_BACKEND="llvmpipe")
target_link_libraries(llvmpipe PRIVATE pvrgpu-rdc-runner-support)

add_executable(pvrgpu src/rdc_runner/main.cpp)
target_compile_definitions(pvrgpu PRIVATE PVRGPU_RDC_BACKEND="pvrgpu")
target_link_libraries(pvrgpu PRIVATE pvrgpu-rdc-runner-support)

pvrgpu_target_warnings(llvmpipe)
pvrgpu_target_warnings(pvrgpu)

install(TARGETS llvmpipe pvrgpu RUNTIME DESTINATION bin)

add_library(pvrgpu-memory-core STATIC
    src/systemc/cache_mmu/cache_array.cpp
    src/systemc/memory/dram_address_space.cpp
    src/systemc/memory/gpu_memory_system.cpp
)
set_target_properties(pvrgpu-memory-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pvrgpu-memory-core PUBLIC
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pvrgpu-memory-core)

# The archive is pulled only into executables that reference its symbols.
link_libraries(pvrgpu-memory-core)

add_library(pvrgpu-geometry-core STATIC src/systemc/common/geometry_emission.cpp)
set_target_properties(pvrgpu-geometry-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pvrgpu-geometry-core PUBLIC "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pvrgpu-geometry-core)
link_libraries(pvrgpu-geometry-core)

add_library(pvrgpu-tessellation-core STATIC
    src/systemc/common/tessellation.cpp
    src/systemc/common/tessellation_reference.cpp
)
set_target_properties(pvrgpu-tessellation-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pvrgpu-tessellation-core PUBLIC "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pvrgpu-tessellation-core)
link_libraries(pvrgpu-tessellation-core)

add_library(pvrgpu-tessellation-iss-core STATIC src/systemc/shader/tessellation_iss.cpp)
set_target_properties(pvrgpu-tessellation-iss-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pvrgpu-tessellation-iss-core PUBLIC
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pvrgpu-tessellation-iss-core)
link_libraries(pvrgpu-tessellation-iss-core)

set(PVRGPU_TESSELLATION_STAGE_SOURCES
    src/systemc/shader/tessellation_control_shader.cpp
    src/systemc/geometry/tessellator.cpp
    src/systemc/shader/tessellation_evaluation_shader.cpp
)

add_executable(pvrgpu-model-stub
    src/systemc/geometry/stream_output.cpp
    ${PVRGPU_TESSELLATION_STAGE_SOURCES}
    src/systemc/shader/geometry_shader.cpp
    src/systemc/shader/geometry_iss.cpp
    model_stub/driver_command.cpp
    model_stub/json_reporter.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/pvrgpu_model_stub.cpp
    model_stub/submitter.cpp
    src/support/png_writer.cpp
    src/systemc/cache_mmu/mixed_cache.cpp
    src/systemc/cache_mmu/mmu_bif.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/cache_mmu/texture_cache.cpp
    src/systemc/cache_mmu/usc_l2_cache.cpp
    src/systemc/compression/image_compression.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/data_master/compute_data_master.cpp
    src/systemc/data_master/domain_data_master.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/data_master/vertex_data_master.cpp
    src/systemc/data_master/two_d_data_master.cpp
    src/systemc/firmware/firmware_scheduler.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/host/control_register_bus.cpp
    src/systemc/host/soc_bus_interface.cpp
    src/systemc/host/xpu_interface.cpp
    src/systemc/memory/mem_fabric.cpp
    src/systemc/memory/on_chip_fabric.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/compute_shader.cpp
    src/systemc/shader/compute_iss.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(pvrgpu-model-stub PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pvrgpu-model-stub PRIVATE
    PkgConfig::LIBPNG
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(pvrgpu-model-stub)

add_library(pvrgpu-systemc-bridge SHARED
    src/systemc/geometry/stream_output.cpp
    ${PVRGPU_TESSELLATION_STAGE_SOURCES}
    src/systemc/shader/geometry_shader.cpp
    src/systemc/shader/geometry_iss.cpp
    model_stub/driver_command.cpp
    model_stub/json_reporter.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/pvrgpu_model_stub.cpp
    model_stub/pvrgpu_systemc_bridge.cpp
    model_stub/submitter.cpp
    src/support/png_writer.cpp
    src/systemc/cache_mmu/mixed_cache.cpp
    src/systemc/cache_mmu/mmu_bif.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/cache_mmu/texture_cache.cpp
    src/systemc/cache_mmu/usc_l2_cache.cpp
    src/systemc/compression/image_compression.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/data_master/compute_data_master.cpp
    src/systemc/data_master/domain_data_master.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/data_master/vertex_data_master.cpp
    src/systemc/data_master/two_d_data_master.cpp
    src/systemc/firmware/firmware_scheduler.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/host/control_register_bus.cpp
    src/systemc/host/soc_bus_interface.cpp
    src/systemc/host/xpu_interface.cpp
    src/systemc/memory/mem_fabric.cpp
    src/systemc/memory/on_chip_fabric.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/compute_shader.cpp
    src/systemc/shader/compute_iss.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
set_target_properties(pvrgpu-systemc-bridge PROPERTIES
    OUTPUT_NAME "pvrgpu_systemc_bridge"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_include_directories(pvrgpu-systemc-bridge PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pvrgpu-systemc-bridge PRIVATE
    PkgConfig::LIBPNG
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(pvrgpu-systemc-bridge)

# `pvrgpu` is the user-facing replay target. Building it also prepares the
# model fallback and the in-process driver-to-SystemC bridge it resolves at
# runtime.
add_dependencies(pvrgpu pvrgpu-model-stub pvrgpu-systemc-bridge)
install(TARGETS pvrgpu-model-stub pvrgpu-systemc-bridge
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Optional live dEQP integration.  VK-GL-CTS remains an external, pinned
# checkout, but its EGL/GLES packages, the PvrGPU Mesa frontend, and the
# SystemC bridge are exposed through one directly linked executable.
set(PVRGPU_DEQP_SOURCE_DIR "$ENV{PVRGPU_DEQP_SOURCE_DIR}" CACHE PATH
    "Path to a fetched VK-GL-CTS source checkout")
set(PVRGPU_DEQP_MESA_PREFIX "$ENV{PVRGPU_MESA_PVRGPU_PREFIX}" CACHE PATH
    "Mesa install prefix containing the PvrGPU Gallium driver")
set(PVRGPU_DEQP_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/pvrgpu-deqp-build"
    CACHE PATH "Out-of-tree VK-GL-CTS build used by pvrgpu-deqp")
set(pvrgpu_live_deqp_default OFF)
if(PVRGPU_DEQP_SOURCE_DIR AND PVRGPU_DEQP_MESA_PREFIX)
    set(pvrgpu_live_deqp_default ON)
endif()
option(PVRGPU_ENABLE_LIVE_DEQP
    "Build one live EGL/GLES dEQP executable linked to Mesa PvrGPU"
    ${pvrgpu_live_deqp_default})

if(PVRGPU_ENABLE_LIVE_DEQP)
    if(NOT EXISTS "${PVRGPU_DEQP_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "PVRGPU_DEQP_SOURCE_DIR is not a VK-GL-CTS checkout: "
            "${PVRGPU_DEQP_SOURCE_DIR}")
    endif()
    if(NOT EXISTS "${PVRGPU_DEQP_MESA_PREFIX}/include/EGL/egl.h")
        message(FATAL_ERROR
            "PVRGPU_DEQP_MESA_PREFIX has no EGL headers: "
            "${PVRGPU_DEQP_MESA_PREFIX}")
    endif()

    unset(PVRGPU_DEQP_EGL_LIBRARY CACHE)
    find_library(PVRGPU_DEQP_EGL_LIBRARY
        NAMES EGL libEGL
        PATHS
            "${PVRGPU_DEQP_MESA_PREFIX}/lib"
            "${PVRGPU_DEQP_MESA_PREFIX}/lib64"
        NO_DEFAULT_PATH REQUIRED)
    unset(PVRGPU_DEQP_GLES_LIBRARY CACHE)
    find_library(PVRGPU_DEQP_GLES_LIBRARY
        NAMES GLESv2 libGLESv2
        PATHS
            "${PVRGPU_DEQP_MESA_PREFIX}/lib"
            "${PVRGPU_DEQP_MESA_PREFIX}/lib64"
        NO_DEFAULT_PATH REQUIRED)

    include(ExternalProject)
    set(pvrgpu_deqp_build_type "${CMAKE_BUILD_TYPE}")
    if(NOT pvrgpu_deqp_build_type)
        set(pvrgpu_deqp_build_type Release)
    endif()
    get_filename_component(pvrgpu_cxx_compiler_dir
        "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(pvrgpu_cxx_compiler_name
        "${CMAKE_CXX_COMPILER}" NAME)
    if(pvrgpu_cxx_compiler_name MATCHES "^clang\\+\\+(\\.exe)?$")
        set(pvrgpu_deqp_c_compiler_names clang)
    elseif(pvrgpu_cxx_compiler_name MATCHES "^(g\\+\\+|gcc)(-[0-9.]+)?(\\.exe)?$")
        set(pvrgpu_deqp_c_compiler_names gcc)
    elseif(pvrgpu_cxx_compiler_name MATCHES "^cl(\\.exe)?$")
        set(pvrgpu_deqp_c_compiler_names cl)
    else()
        set(pvrgpu_deqp_c_compiler_names cc clang gcc cl)
    endif()
    find_program(PVRGPU_DEQP_C_COMPILER
        NAMES ${pvrgpu_deqp_c_compiler_names}
        HINTS "${pvrgpu_cxx_compiler_dir}"
        REQUIRED)
    set(PVRGPU_DEQP_CXX_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
        "C++ compiler used for the VK-GL-CTS composite executable")

    ExternalProject_Add(pvrgpu-deqp
        SOURCE_DIR "${PVRGPU_DEQP_SOURCE_DIR}"
        BINARY_DIR "${PVRGPU_DEQP_BUILD_DIR}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PATCH_COMMAND ""
        CMAKE_ARGS
            "-DCMAKE_BUILD_TYPE:STRING=${pvrgpu_deqp_build_type}"
            "-DCMAKE_C_COMPILER:FILEPATH=${PVRGPU_DEQP_C_COMPILER}"
            "-DCMAKE_CXX_COMPILER:FILEPATH=${PVRGPU_DEQP_CXX_COMPILER}"
            "-DCMAKE_PROJECT_INCLUDE:FILEPATH=${CMAKE_SOURCE_DIR}/cmake/PvrGPUDeqpSingleBinary.cmake"
            "-DDEQP_TARGET:STRING=nullws"
            "-DGLES_ALLOW_DIRECT_LINK:BOOL=ON"
            "-DSELECTED_BUILD_TARGETS:STRING=pvrgpu-deqp"
            "-DCMAKE_PREFIX_PATH:PATH=${PVRGPU_DEQP_MESA_PREFIX}"
            "-DEGL_LIBRARY:FILEPATH=${PVRGPU_DEQP_EGL_LIBRARY}"
            "-DGLES2_LIBRARY:FILEPATH=${PVRGPU_DEQP_GLES_LIBRARY}"
            "-DGLES3_LIBRARY:FILEPATH=${PVRGPU_DEQP_GLES_LIBRARY}"
            "-DEGL_INCLUDE_PATH:PATH=${PVRGPU_DEQP_MESA_PREFIX}/include"
            "-DGLES2_INCLUDE_PATH:PATH=${PVRGPU_DEQP_MESA_PREFIX}/include"
            "-DGLES3_INCLUDE_PATH:PATH=${PVRGPU_DEQP_MESA_PREFIX}/include"
            "-DGLES31_INCLUDE_PATH:PATH=${PVRGPU_DEQP_MESA_PREFIX}/include"
            "-DGLES32_INCLUDE_PATH:PATH=${PVRGPU_DEQP_MESA_PREFIX}/include"
            "-DPVRGPU_DEQP_RUNNER_MAIN:FILEPATH=${CMAKE_SOURCE_DIR}/src/deqp_runner/main.cpp"
            "-DPVRGPU_DEQP_OUTPUT_DIR:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            "-DPVRGPU_DEQP_SYSTEMC_BRIDGE:FILEPATH=$<TARGET_FILE:pvrgpu-systemc-bridge>"
            "-DPVRGPU_DEQP_SYSTEMC_API_INCLUDE_DIR:PATH=${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
        BUILD_COMMAND
            "${CMAKE_COMMAND}" --build <BINARY_DIR>
                --target pvrgpu-deqp --parallel
        INSTALL_COMMAND ""
        BUILD_ALWAYS TRUE
        BUILD_BYPRODUCTS
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pvrgpu-deqp${CMAKE_EXECUTABLE_SUFFIX}"
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
    )
    add_dependencies(pvrgpu-deqp pvrgpu-systemc-bridge)

    install(PROGRAMS
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pvrgpu-deqp${CMAKE_EXECUTABLE_SUFFIX}"
        DESTINATION bin OPTIONAL)
    install(DIRECTORY
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gles2"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gles3"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gles31"
        DESTINATION bin OPTIONAL)
endif()

enable_testing()
if(UNIX)
    add_test(NAME model-diagnostics-unit COMMAND model-diagnostics-test)
endif()
add_executable(memory-pool-test
    tests/memory_pool_test.cpp
    model_stub/memory_pool.cpp
)
target_include_directories(memory-pool-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
)
pvrgpu_target_warnings(memory-pool-test)
add_test(NAME memory-pool-unit COMMAND memory-pool-test)
add_executable(rdc-native-report-test tests/rdc_native_report_test.cpp)
target_link_libraries(rdc-native-report-test PRIVATE pvrgpu-rdc-runner-support)
pvrgpu_target_warnings(rdc-native-report-test)
add_test(NAME rdc-native-report-test COMMAND rdc-native-report-test)
add_executable(replay-range-audit-test tests/replay_range_audit_test.cpp)
target_link_libraries(replay-range-audit-test PRIVATE pvrgpu-rdc-runner-support)
pvrgpu_target_warnings(replay-range-audit-test)
add_test(NAME replay-range-audit-test COMMAND replay-range-audit-test)
if(UNIX)
    add_executable(rdc-gl-debug-scope-test tests/rdc_gl_debug_scope_test.cpp)
    target_include_directories(rdc-gl-debug-scope-test PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(rdc-gl-debug-scope-test PRIVATE ${CMAKE_DL_LIBS})
    pvrgpu_target_warnings(rdc-gl-debug-scope-test)
    add_test(NAME rdc-gl-debug-scope-test COMMAND rdc-gl-debug-scope-test)
endif()
add_test(NAME llvmpipe-native-cli-help COMMAND llvmpipe --help)
add_test(NAME pvrgpu-native-cli-help COMMAND pvrgpu --help)
if(PVRGPU_ENABLE_LIVE_DEQP)
    add_test(NAME pvrgpu-deqp-cli-help
        COMMAND
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pvrgpu-deqp${CMAKE_EXECUTABLE_SUFFIX}"
            --pvrgpu-help)
    add_test(NAME pvrgpu-deqp-clear-smoke
        COMMAND
            "${CMAKE_COMMAND}"
            "-DRUNNER=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pvrgpu-deqp${CMAKE_EXECUTABLE_SUFFIX}"
            "-DOUTPUT_DIR=${CMAKE_BINARY_DIR}/test-output/pvrgpu-deqp-clear-smoke"
            -P "${CMAKE_SOURCE_DIR}/cmake/PvrGPUDeqpSmoke.cmake")
    set_tests_properties(pvrgpu-deqp-clear-smoke PROPERTIES TIMEOUT 60)
endif()

add_executable(msaa-resolve-test
    tests/msaa_resolve_test.cpp
)
target_include_directories(msaa-resolve-test PRIVATE "${CMAKE_SOURCE_DIR}/src")
pvrgpu_target_warnings(msaa-resolve-test)
add_test(NAME msaa-resolve-unit COMMAND msaa-resolve-test)

add_executable(pco-iss-test
    tests/pco_iss_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-iss-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-iss-test)
add_test(NAME pco-iss-unit COMMAND pco-iss-test)

add_executable(pco-prepared-fragment-test
    tests/pco_prepared_fragment_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-prepared-fragment-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-prepared-fragment-test)
add_test(NAME pco-prepared-fragment-unit COMMAND pco-prepared-fragment-test)

add_executable(pco-texture-sequence-test
    tests/pco_texture_sequence_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-texture-sequence-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-texture-sequence-test)
add_test(NAME pco-texture-sequence-unit COMMAND pco-texture-sequence-test)

add_executable(pco-texture-gather-test
    tests/pco_texture_gather_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-texture-gather-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-texture-gather-test)
add_test(NAME pco-texture-gather-unit COMMAND pco-texture-gather-test)

add_executable(pco-array-shadow-gather-test
    tests/pco_array_shadow_gather_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-array-shadow-gather-test PRIVATE "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pco-array-shadow-gather-test)
add_test(NAME pco-array-shadow-gather-unit COMMAND pco-array-shadow-gather-test)

add_executable(pco-branch-continuation-test
    tests/pco_branch_continuation_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-branch-continuation-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-branch-continuation-test)
add_test(NAME pco-branch-continuation-unit COMMAND pco-branch-continuation-test)

add_executable(pco-front-face-test
    tests/pco_front_face_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-front-face-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-front-face-test)
add_test(NAME pco-front-face-unit COMMAND pco-front-face-test)

add_executable(pco-vertex-shift-test
    tests/pco_vertex_shift_test.cpp
    src/systemc/shader/pco_iss.cpp)
target_include_directories(pco-vertex-shift-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pco-vertex-shift-test)
add_test(NAME pco-vertex-shift-unit COMMAND pco-vertex-shift-test)

add_executable(pco-find-top-bit-test
    tests/pco_find_top_bit_test.cpp
    src/systemc/shader/pco_iss.cpp)
target_include_directories(pco-find-top-bit-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pco-find-top-bit-test)
add_test(NAME pco-find-top-bit-unit COMMAND pco-find-top-bit-test)

add_executable(fragment-output-mask-test tests/fragment_output_mask_test.cpp)
target_include_directories(fragment-output-mask-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(fragment-output-mask-test)
add_test(NAME fragment-output-mask-unit COMMAND fragment-output-mask-test)

# This checker takes a directory of binaries from the real Mesa-side
# pvrgpu_pco_fragment_output_test generator; it has no synthetic fallback.
add_executable(pco-fragment-output-contract-test
    tests/pco_fragment_output_contract_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-fragment-output-contract-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-fragment-output-contract-test)

add_executable(pco-multisample-texture-test
    tests/pco_multisample_texture_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-multisample-texture-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-multisample-texture-test)
add_test(NAME pco-multisample-texture-unit COMMAND pco-multisample-texture-test)

add_executable(pco-sequence-profiles-test
    tests/pco_sequence_profiles_test.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/memory_pool.cpp
    src/systemc/common/functional_types.cpp
)
target_include_directories(pco-sequence-profiles-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
)
pvrgpu_target_warnings(pco-sequence-profiles-test)
add_test(NAME pco-sequence-profiles-unit COMMAND pco-sequence-profiles-test)

add_executable(pco-test-select-mbyp-test
    tests/pco_test_select_mbyp_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-test-select-mbyp-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-test-select-mbyp-test)
add_test(NAME pco-test-select-mbyp-unit COMMAND pco-test-select-mbyp-test)

add_executable(driver-command-test
    tests/driver_command_test.cpp
    model_stub/driver_command.cpp
)
target_include_directories(driver-command-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
)
pvrgpu_target_warnings(driver-command-test)
add_test(NAME driver-command-unit COMMAND driver-command-test)

add_executable(systemc-api-bridge-test
    tests/systemc_api_bridge_test.cpp
)
target_include_directories(systemc-api-bridge-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(systemc-api-bridge-test PRIVATE
    pvrgpu-systemc-bridge
    PkgConfig::LIBPNG
)
pvrgpu_target_warnings(systemc-api-bridge-test)
add_test(NAME systemc-api-bridge-unit COMMAND systemc-api-bridge-test)

add_executable(systemc-readback-flush-test
    tests/systemc_readback_flush_test.cpp
)
target_include_directories(systemc-readback-flush-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(systemc-readback-flush-test PRIVATE
    pvrgpu-systemc-bridge
)
pvrgpu_target_warnings(systemc-readback-flush-test)
add_test(NAME systemc-readback-flush-unit COMMAND systemc-readback-flush-test)
add_test(NAME systemc-compute-first-interleave-unit
    COMMAND systemc-readback-flush-test compute-first)

add_executable(shader-stage-placeholders-test
    tests/shader_stage_placeholders_test.cpp
)
target_include_directories(shader-stage-placeholders-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(shader-stage-placeholders-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(shader-stage-placeholders-test)
add_test(NAME shader-stage-placeholders-unit COMMAND shader-stage-placeholders-test)

add_executable(geometry-emission-test tests/geometry_emission_test.cpp)
pvrgpu_target_warnings(geometry-emission-test)
add_test(NAME geometry-emission-unit COMMAND geometry-emission-test)

add_executable(tessellation-test tests/tessellation_test.cpp)
pvrgpu_target_warnings(tessellation-test)
add_test(NAME tessellation-unit COMMAND tessellation-test)

foreach(tess_test IN ITEMS tessellation-iss tessellation-module tessellation-shader tessellation-texture-iss)
    string(REPLACE "-" "_" tess_source "${tess_test}")
    add_executable(${tess_test}-test tests/${tess_source}_test.cpp)
    target_include_directories(${tess_test}-test PRIVATE
        "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/systemc")
    target_link_libraries(${tess_test}-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
    pvrgpu_target_warnings(${tess_test}-test)
    add_test(NAME ${tess_test}-unit COMMAND ${tess_test}-test)
endforeach()

add_executable(tessellation-texture-test tests/tessellation_texture_test.cpp)
target_include_directories(tessellation-texture-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(tessellation-texture-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(tessellation-texture-test)
foreach(tess_texture_mode IN ITEMS cache direct bypass reject-count reject-descriptor
        reject-stage reject-gather reject-shadow reject-dimension reject-identity)
    add_test(NAME tessellation-texture-${tess_texture_mode}
        COMMAND tessellation-texture-test ${tess_texture_mode})
endforeach()
add_test(NAME tessellation-shader-textures-unit COMMAND tessellation-shader-test textures)

add_executable(geometry-shader-test tests/geometry_shader_test.cpp)
target_include_directories(geometry-shader-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(geometry-shader-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(geometry-shader-test)
add_test(NAME geometry-shader-unit COMMAND geometry-shader-test)

add_executable(stream-output-test tests/stream_output_test.cpp)
target_include_directories(stream-output-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc" "${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(stream-output-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(stream-output-test)
foreach(stream_output_mode IN ITEMS direct bypass cache)
    add_test(NAME stream-output-${stream_output_mode}-unit
        COMMAND stream-output-test ${stream_output_mode})
endforeach()

add_executable(stream-output-api-test tests/stream_output_api_test.cpp)
target_include_directories(stream-output-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc" "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu")
target_link_libraries(stream-output-api-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(stream-output-api-test)
foreach(stream_output_mode IN ITEMS direct bypass cache)
    add_test(NAME stream-output-api-${stream_output_mode}-unit
        COMMAND stream-output-api-test ${stream_output_mode})
endforeach()

add_executable(fragment-image-api-test tests/fragment_image_api_test.cpp)
target_include_directories(fragment-image-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc" "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu")
target_link_libraries(fragment-image-api-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(fragment-image-api-test)
foreach(fragment_image_mode IN ITEMS direct bypass cache)
    add_test(NAME fragment-image-api-${fragment_image_mode}-unit
        COMMAND fragment-image-api-test ${fragment_image_mode})
endforeach()

add_executable(pco-decoder-boundary-test tests/pco_decoder_boundary_test.cpp)
target_include_directories(pco-decoder-boundary-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(pco-decoder-boundary-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(pco-decoder-boundary-test)
foreach(pco_decoder_boundary IN ITEMS valid reject65 reject65-point-size)
    add_test(NAME pco-decoder-boundary-${pco_decoder_boundary}
        COMMAND pco-decoder-boundary-test ${pco_decoder_boundary})
endforeach()

add_executable(geometry-frontend-test tests/geometry_frontend_test.cpp)
target_include_directories(geometry-frontend-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(geometry-frontend-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(geometry-frontend-test)
add_test(NAME geometry-frontend-unit COMMAND geometry-frontend-test)

add_executable(point-restart-test tests/point_restart_test.cpp)
target_include_directories(point-restart-test PRIVATE "${CMAKE_SOURCE_DIR}/src")
pvrgpu_target_warnings(point-restart-test)
add_test(NAME point-restart-unit COMMAND point-restart-test)

add_executable(indirect-draw-test tests/indirect_draw_test.cpp)
target_include_directories(indirect-draw-test PRIVATE "${CMAKE_SOURCE_DIR}/src")
pvrgpu_target_warnings(indirect-draw-test)
add_test(NAME indirect-draw-unit COMMAND indirect-draw-test)

add_executable(index-fetch-test tests/index_fetch_test.cpp)
target_include_directories(index-fetch-test PRIVATE "${CMAKE_SOURCE_DIR}/src")
pvrgpu_target_warnings(index-fetch-test)
add_test(NAME index-fetch-unit COMMAND index-fetch-test)

add_executable(vertex-fetch-test tests/vertex_fetch_test.cpp)
target_include_directories(vertex-fetch-test PRIVATE "${CMAKE_SOURCE_DIR}/src")
pvrgpu_target_warnings(vertex-fetch-test)
add_test(NAME vertex-fetch-unit COMMAND vertex-fetch-test)

add_executable(pco-systemc-api-bridge-test
    tests/pco_systemc_api_bridge_test.cpp
)
target_include_directories(pco-systemc-api-bridge-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pco-systemc-api-bridge-test PRIVATE
    pvrgpu-systemc-bridge
    PkgConfig::LIBPNG
)
pvrgpu_target_warnings(pco-systemc-api-bridge-test)
add_test(NAME pco-systemc-api-bridge-unit COMMAND pco-systemc-api-bridge-test)

add_executable(color-attachment-format-contract-test tests/color_attachment_format_contract_test.cpp)
target_include_directories(color-attachment-format-contract-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
pvrgpu_target_warnings(color-attachment-format-contract-test)
add_test(NAME color-attachment-format-contract-unit COMMAND color-attachment-format-contract-test)

add_executable(color-attachment-format-parser-test
    tests/color_attachment_format_parser_test.cpp
    model_stub/driver_command.cpp
)
target_include_directories(color-attachment-format-parser-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
pvrgpu_target_warnings(color-attachment-format-parser-test)
add_test(NAME color-attachment-format-parser-unit COMMAND color-attachment-format-parser-test)

add_executable(pco-color-attachment-format-api-test tests/pco_color_attachment_format_api_test.cpp)
target_include_directories(pco-color-attachment-format-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pco-color-attachment-format-api-test PRIVATE pvrgpu-systemc-bridge PkgConfig::LIBPNG)
pvrgpu_target_warnings(pco-color-attachment-format-api-test)
add_test(NAME pco-color-attachment-format-api-unit COMMAND pco-color-attachment-format-api-test)

add_executable(pco-cube-array-api-test tests/pco_cube_array_api_test.cpp)
target_include_directories(pco-cube-array-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc" "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu")
target_link_libraries(pco-cube-array-api-test PRIVATE pvrgpu-systemc-bridge PkgConfig::LIBPNG)
pvrgpu_target_warnings(pco-cube-array-api-test)
add_test(NAME pco-cube-array-api-unit COMMAND pco-cube-array-api-test)

add_executable(pco-geometry-api-bridge-test tests/pco_geometry_api_bridge_test.cpp)
target_include_directories(pco-geometry-api-bridge-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pco-geometry-api-bridge-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(pco-geometry-api-bridge-test)
add_test(NAME pco-geometry-api-bridge-unit COMMAND pco-geometry-api-bridge-test)

add_executable(pco-layered-api-test tests/pco_layered_api_test.cpp)
target_include_directories(pco-layered-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu")
target_link_libraries(pco-layered-api-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(pco-layered-api-test)
add_test(NAME pco-layered-api-direct COMMAND pco-layered-api-test direct)
add_test(NAME pco-layered-api-bypass COMMAND pco-layered-api-test bypass)
add_test(NAME pco-layered-api-cache COMMAND pco-layered-api-test cache)

add_executable(pco-tessellation-api-bridge-test tests/pco_tessellation_api_bridge_test.cpp)
target_include_directories(pco-tessellation-api-bridge-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(pco-tessellation-api-bridge-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(pco-tessellation-api-bridge-test)
add_test(NAME pco-tessellation-api-bridge-unit COMMAND pco-tessellation-api-bridge-test)
add_test(NAME pco-tessellation-incomplete-patch-unit
    COMMAND pco-tessellation-api-bridge-test cache incomplete)
foreach(tess_mode IN ITEMS direct bypass)
    add_test(NAME pco-tessellation-api-${tess_mode}-unit
        COMMAND pco-tessellation-api-bridge-test ${tess_mode})
endforeach()

add_executable(pco-tessellation-api-boundary-test tests/pco_tessellation_api_boundary_test.cpp)
target_include_directories(pco-tessellation-api-boundary-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu")
target_link_libraries(pco-tessellation-api-boundary-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(pco-tessellation-api-boundary-test)
add_test(NAME pco-tessellation-api-boundary-unit COMMAND pco-tessellation-api-boundary-test)

add_executable(pco-signed-metadata-test tests/pco_signed_metadata_test.cpp)
target_include_directories(pco-signed-metadata-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(pco-signed-metadata-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(pco-signed-metadata-test)
add_test(NAME pco-signed-metadata-unit COMMAND pco-signed-metadata-test)

foreach(tess_reject RANGE 1 7)
    add_test(NAME tessellation-module-reject-${tess_reject}
        COMMAND tessellation-module-test ${tess_reject})
endforeach()
foreach(tess_shader_reject RANGE 1 8)
    add_test(NAME tessellation-shader-reject-${tess_shader_reject}
        COMMAND tessellation-shader-test ${tess_shader_reject})
endforeach()

add_executable(driver-front-face-test tests/driver_front_face_test.cpp)
target_include_directories(driver-front-face-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(driver-front-face-test PRIVATE pvrgpu-systemc-bridge PkgConfig::SYSTEMC)
pvrgpu_target_warnings(driver-front-face-test)
add_test(NAME driver-front-face-unit COMMAND driver-front-face-test)

add_executable(graphics-stats-test tests/graphics_stats_test.cpp)
target_include_directories(graphics-stats-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/model_stub")
pvrgpu_target_warnings(graphics-stats-test)
add_test(NAME graphics-stats-unit COMMAND graphics-stats-test)

add_executable(compute-systemc-api-test tests/compute_systemc_api_test.cpp)
target_include_directories(compute-systemc-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(compute-systemc-api-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(compute-systemc-api-test)
foreach(compute_memory_mode RANGE 0 2)
    add_test(NAME compute-systemc-api-mode-${compute_memory_mode}
        COMMAND compute-systemc-api-test ${compute_memory_mode})
endforeach()

add_executable(pco-compute-decode-test
    tests/pco_compute_decode_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-compute-decode-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-compute-decode-test)
add_test(NAME pco-compute-decode-unit COMMAND pco-compute-decode-test)

add_executable(pco-compute-alu-test
    tests/pco_compute_alu_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-compute-alu-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(pco-compute-alu-test)
add_test(NAME pco-compute-alu-unit COMMAND pco-compute-alu-test)

add_executable(pco-compute-add64-test
    tests/pco_compute_add64_test.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/compute_iss.cpp
)
target_include_directories(pco-compute-add64-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pco-compute-add64-test PRIVATE pvrgpu-memory-core)
pvrgpu_target_warnings(pco-compute-add64-test)
add_test(NAME pco-compute-add64-unit COMMAND pco-compute-add64-test)

add_executable(compute-shader-fifo-test
    tests/compute_shader_fifo_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/compute_data_master.cpp
    src/systemc/shader/compute_shader.cpp
    src/systemc/shader/compute_iss.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(compute-shader-fifo-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(compute-shader-fifo-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(compute-shader-fifo-test)
foreach(compute_memory_mode RANGE 0 2)
    add_test(NAME compute-shader-fifo-mode-${compute_memory_mode}
        COMMAND compute-shader-fifo-test ${compute_memory_mode})
endforeach()

add_executable(compute-shared-fifo-test
    tests/compute_shared_fifo_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/compute_data_master.cpp
    src/systemc/shader/compute_shader.cpp
    src/systemc/shader/compute_iss.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(compute-shared-fifo-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(compute-shared-fifo-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(compute-shared-fifo-test)
foreach(compute_memory_mode RANGE 0 2)
    add_test(NAME compute-shared-fifo-mode-${compute_memory_mode}
        COMMAND compute-shared-fifo-test ${compute_memory_mode})
endforeach()

add_executable(compute-texture-fifo-test
    tests/compute_texture_fifo_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/data_master/compute_data_master.cpp
    src/systemc/shader/compute_shader.cpp
    src/systemc/shader/compute_iss.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/texture/texture_unit.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/astc_decoder.cpp)
target_include_directories(compute-texture-fifo-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(compute-texture-fifo-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(compute-texture-fifo-test)
foreach(compute_memory_mode RANGE 0 2)
    add_test(NAME compute-texture-fifo-mode-${compute_memory_mode}
        COMMAND compute-texture-fifo-test ${compute_memory_mode})
endforeach()

add_executable(compute-image-api-test tests/compute_image_api_test.cpp)
target_include_directories(compute-image-api-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu")
target_link_libraries(compute-image-api-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(compute-image-api-test)
foreach(compute_memory_mode RANGE 0 2)
    add_test(NAME compute-image-api-mode-${compute_memory_mode}
        COMMAND compute-image-api-test ${compute_memory_mode})
endforeach()

add_executable(pco-sequence-integer-load-test
    tests/pco_sequence_integer_load_test.cpp
)
target_include_directories(pco-sequence-integer-load-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pco-sequence-integer-load-test PRIVATE
    pvrgpu-systemc-bridge
)
pvrgpu_target_warnings(pco-sequence-integer-load-test)
add_test(NAME pco-sequence-integer-load-unit
    COMMAND pco-sequence-integer-load-test)

add_executable(pco-initial-color-load-test
    tests/pco_initial_color_load_test.cpp
)
target_include_directories(pco-initial-color-load-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pco-initial-color-load-test PRIVATE
    pvrgpu-systemc-bridge
)
pvrgpu_target_warnings(pco-initial-color-load-test)
add_test(NAME pco-initial-color-load-unit COMMAND pco-initial-color-load-test)
add_test(NAME pco-uniform-buffer-bypass-unit
    COMMAND pco-initial-color-load-test ubo-bypass)
add_test(NAME pco-uniform-buffer-cache-unit
    COMMAND pco-initial-color-load-test ubo-cache)

add_executable(pco-initial-depth-load-test tests/pco_initial_depth_load_test.cpp)
target_include_directories(pco-initial-depth-load-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/systemc"
    "${CMAKE_SOURCE_DIR}/src/gallium/drivers/pvrgpu"
)
target_link_libraries(pco-initial-depth-load-test PRIVATE pvrgpu-systemc-bridge)
pvrgpu_target_warnings(pco-initial-depth-load-test)
add_test(NAME pco-initial-depth-load-unit COMMAND pco-initial-depth-load-test)

add_executable(json-reporter-opcode-test
    tests/json_reporter_opcode_test.cpp
)
target_include_directories(json-reporter-opcode-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(json-reporter-opcode-test PRIVATE
    pvrgpu-systemc-bridge
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(json-reporter-opcode-test)
add_test(NAME json-reporter-opcode-unit COMMAND json-reporter-opcode-test)

add_executable(cache-array-test
    tests/cache_array_test.cpp
)
target_include_directories(cache-array-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(cache-array-test)
add_test(NAME cache-array-unit COMMAND cache-array-test)

add_executable(gpu-memory-system-test
    tests/gpu_memory_system_test.cpp
)
target_include_directories(gpu-memory-system-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(gpu-memory-system-test)
add_test(NAME gpu-memory-system-unit COMMAND gpu-memory-system-test)

add_executable(dram-address-space-contains-test
    tests/dram_address_space_contains_test.cpp
)
target_include_directories(dram-address-space-contains-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(dram-address-space-contains-test)
add_test(NAME dram-address-space-contains-unit COMMAND dram-address-space-contains-test)

add_executable(cache-write-allocation-test
    tests/cache_write_allocation_test.cpp)
target_include_directories(cache-write-allocation-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/systemc" "${CMAKE_SOURCE_DIR}/model_stub")
pvrgpu_target_warnings(cache-write-allocation-test)
add_test(NAME cache-write-allocation-unit COMMAND cache-write-allocation-test)

add_executable(cache-controller-test
    tests/cache_controller_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/cache_mmu/mixed_cache.cpp
    src/systemc/cache_mmu/texture_cache.cpp
    src/systemc/cache_mmu/usc_l2_cache.cpp
    src/systemc/common/pipeline_state.cpp
)
target_include_directories(cache-controller-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(cache-controller-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(cache-controller-test)
add_test(NAME cache-controller-unit COMMAND cache-controller-test)

add_executable(texture-unit-test
    tests/texture_unit_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(texture-unit-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(texture-unit-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-unit-test)
add_test(NAME texture-unit-unit COMMAND texture-unit-test)
add_test(NAME texture-unit-cache-readinto-unit COMMAND texture-unit-test cache)
add_test(NAME texture-unit-bypass-readinto-unit COMMAND texture-unit-test bypass)

add_executable(texture-gather-test
    tests/texture_gather_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(texture-gather-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(texture-gather-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-gather-test)
foreach(gather_case IN ITEMS
        default 1x1 odd linear lod-window
        rgba-default rgba-1x1 rgba-odd rgba-linear rgba-min-linear
        rgba-mag-linear rgba-lod-window rgba-direct rgba-bypass
        reject-flag reject-implicit reject-lod reject-negative-zero-lod
        reject-normalized reject-fcnorm reject-component reject-coordinate-count
        reject-coordinate-z reject-offset reject-address reject-sno reject-bias
        reject-stage reject-nan reject-infinity reject-sampler-offset
        reject-repeat reject-compare reject-base reject-layers reject-dimension
        reject-format reject-red-swizzle reject-gather-descriptor
        reject-gather-word1 reject-image-state reject-mixed-batch
        reject-rgba-red-swizzle
        array-default array-1x1 array-odd array-linear array-direct array-bypass
        array-rgba-default array-rgba-1x1 array-rgba-odd array-rgba-linear
        array-rgba-direct array-rgba-bypass
        compare-0 compare-1 compare-2 compare-3 compare-4 compare-5 compare-6 compare-7
        array-compare-0 array-compare-1 array-compare-2 array-compare-3
        array-compare-4 array-compare-5 array-compare-6 array-compare-7
        reject-array-stride reject-array-depth reject-array-unaligned
        reject-array-below-base reject-array-past-last reject-array-missing-address
        reject-array-msaa reject-array-mips)
    add_test(NAME texture-gather-${gather_case}
             COMMAND texture-gather-test ${gather_case})
endforeach()

add_executable(texture-multisample-test
    tests/texture_multisample_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(texture-multisample-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(texture-multisample-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-multisample-test)
add_test(NAME texture-multisample-unit COMMAND texture-multisample-test)

add_executable(texture-cube-nonfinite-test
    tests/texture_cube_nonfinite_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(texture-cube-nonfinite-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(texture-cube-nonfinite-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-cube-nonfinite-test)
add_test(NAME texture-cube-nonfinite-unit COMMAND texture-cube-nonfinite-test)

add_executable(pco-cube-array-test tests/pco_cube_array_test.cpp src/systemc/shader/pco_iss.cpp)
target_include_directories(pco-cube-array-test PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(pco-cube-array-test)
add_test(NAME pco-cube-array-unit COMMAND pco-cube-array-test)

add_executable(texture-cube-array-test
    tests/texture_cube_array_test.cpp
    model_stub/memory_pool.cpp model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp src/systemc/common/pipeline_state.cpp
    src/systemc/texture/astc_decoder.cpp src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp)
target_include_directories(texture-cube-array-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(texture-cube-array-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-cube-array-test)
add_test(NAME texture-cube-array-unit COMMAND texture-cube-array-test)
foreach(refusal offset gather bias face-address low-address end-address word4 raw-depth shadow)
  add_test(NAME texture-cube-array-reject-${refusal} COMMAND texture-cube-array-test ${refusal})
endforeach()

# Native Mesa-emitted BIAS/PPLOD, real USC/TextureUnit FIFOs, and mip data.
# The private Mesa producer script also runs this fixture on freshly compiled
# binaries, independently of the checked-in bytes used by ordinary CTest.
add_executable(texture-bias-test
    tests/texture_bias_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(texture-bias-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(texture-bias-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-bias-test)
add_test(NAME texture-bias-unit COMMAND texture-bias-test)

add_executable(centroid-geometry-test tests/centroid_geometry_test.cpp)
target_include_directories(centroid-geometry-test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(centroid-geometry-test)
add_test(NAME centroid-geometry-unit COMMAND centroid-geometry-test)

add_executable(shader-image-payload-test tests/shader_image_payload_test.cpp)
target_include_directories(shader-image-payload-test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/model_stub")
pvrgpu_target_warnings(shader-image-payload-test)
add_test(NAME shader-image-payload-unit COMMAND shader-image-payload-test)

add_executable(astc-decoder-test
    tests/astc_decoder_test.cpp
    src/systemc/texture/astc_decoder.cpp
)
target_include_directories(astc-decoder-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(astc-decoder-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(astc-decoder-test)
add_test(NAME astc-decoder-unit COMMAND astc-decoder-test)

add_executable(texture-filter-test
    tests/texture_filter_test.cpp
    model_stub/memory_pool.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/texture/texture_filter.cpp
)
target_include_directories(texture-filter-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(texture-filter-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-filter-test)
add_test(NAME texture-filter-unit COMMAND texture-filter-test)

add_executable(texture-lod-invariance-test tests/texture_lod_invariance_test.cpp
    model_stub/memory_pool.cpp src/systemc/common/functional_types.cpp
    src/systemc/texture/texture_filter.cpp)
target_include_directories(texture-lod-invariance-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src/systemc")
pvrgpu_target_warnings(texture-lod-invariance-test)
add_test(NAME texture-lod-invariance-unit COMMAND texture-lod-invariance-test)

add_executable(texture-lod-independent-test tests/texture_lod_independent_test.cpp
    model_stub/memory_pool.cpp model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp src/systemc/common/pipeline_state.cpp
    src/systemc/texture/astc_decoder.cpp src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp)
target_include_directories(texture-lod-independent-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(texture-lod-independent-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(texture-lod-independent-test)
add_test(NAME texture-lod-independent-unit COMMAND texture-lod-independent-test)
foreach(lod_case IN ITEMS half next wide filters miplinear)
  add_test(NAME texture-lod-dependent-reject-${lod_case}
      COMMAND texture-lod-independent-test ${lod_case})
endforeach()

add_executable(usc-uniform-buffer-memory-test
    tests/usc_uniform_buffer_memory_test.cpp
)
target_include_directories(usc-uniform-buffer-memory-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(usc-uniform-buffer-memory-test)
add_test(NAME usc-uniform-buffer-memory-unit COMMAND usc-uniform-buffer-memory-test)

add_executable(uniform-buffer-contract-test
    tests/uniform_buffer_contract_test.cpp
)
target_include_directories(uniform-buffer-contract-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
pvrgpu_target_warnings(uniform-buffer-contract-test)
add_test(NAME uniform-buffer-contract-unit COMMAND uniform-buffer-contract-test)

add_executable(usc-cluster-texture-continuation-test
    tests/usc_cluster_texture_continuation_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(usc-cluster-texture-continuation-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(usc-cluster-texture-continuation-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(usc-cluster-texture-continuation-test)
add_test(
    NAME usc-cluster-texture-continuation-unit
    COMMAND usc-cluster-texture-continuation-test
)
add_test(
    NAME usc-cluster-texture-request-limit-fail-closed
    COMMAND usc-cluster-texture-continuation-test request-limit
)
add_test(
    NAME usc-cluster-texture-request-boundary-unit
    COMMAND usc-cluster-texture-continuation-test request-boundary
)
foreach(samples 10 16)
    add_test(
        NAME usc-cluster-texture-sequence-${samples}-unit
        COMMAND usc-cluster-texture-continuation-test sequence${samples}
    )
endforeach()
add_test(
    NAME usc-cluster-texture-response-order-fail-closed
    COMMAND usc-cluster-texture-continuation-test response-order
)
foreach(field count steps mask)
    add_test(
        NAME usc-cluster-texture-continuation-${field}-fail-closed
        COMMAND usc-cluster-texture-continuation-test continuation-${field}
    )
endforeach()

foreach(scratch_failure request-bytes response-bytes continuation-bytes response-count)
    add_test(
        NAME usc-cluster-texture-scratch-${scratch_failure}-fail-closed
        COMMAND usc-cluster-texture-continuation-test scratch-${scratch_failure}
    )
endforeach()

add_executable(usc-fragment-residency-test
    tests/usc_fragment_residency_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(usc-fragment-residency-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(usc-fragment-residency-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(usc-fragment-residency-test)
add_test(NAME usc-fragment-residency-unit COMMAND usc-fragment-residency-test)
foreach(residency_invalid_state
        duplicate-task duplicate-lane missing-lane missing-quad
        lane-out-of-range task-out-of-range overlapping-mask
        high-coverage-mask high-helper-mask high-write-mask empty-mask
        partial-derivative-quad coefficient-offset coefficient-span output-collision
        facing-noncanonical helper-facing-mismatch visible-facing-mismatch
        continuation-facing continuation-facing-valid
        program-stage program-bytes program-groups program-instructions
        program-vertex-input program-vertex-output program-pixel-output
        program-early-hsr program-depth program-derivatives program-ends-task
        program-handle-slot program-handle-generation)
    add_test(NAME usc-fragment-residency-reject-${residency_invalid_state}
        COMMAND usc-fragment-residency-test ${residency_invalid_state})
    set_tests_properties(usc-fragment-residency-reject-${residency_invalid_state}
        PROPERTIES TIMEOUT 30
        PASS_REGULAR_EXPRESSION "reject ${residency_invalid_state} PASS diagnostic=")
endforeach()

add_executable(pbe-blend-test
    tests/pbe_blend_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pbe-blend-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pbe-blend-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(pbe-blend-test)
add_test(NAME pbe-blend-unit COMMAND pbe-blend-test)

add_executable(pbe-packed-unorm-test
    tests/pbe_packed_unorm_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/pbe.cpp
)
target_include_directories(pbe-packed-unorm-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pbe-packed-unorm-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(pbe-packed-unorm-test)
add_test(NAME pbe-packed-unorm-unit COMMAND pbe-packed-unorm-test)
foreach(pbe_packed_invalid_state IN ITEMS float integer srgb format load-size)
    add_test(NAME pbe-packed-unorm-reject-${pbe_packed_invalid_state}
        COMMAND pbe-packed-unorm-test ${pbe_packed_invalid_state})
endforeach()

add_test(NAME pbe-packed-unorm-mixed COMMAND pbe-packed-unorm-test mixed)
foreach(pbe_mixed_invalid_state IN ITEMS mixed-count mixed-count-overflow mixed-target0
        mixed-inactive mixed-enum mixed-legacy mixed-float mixed-integer mixed-srgb
        legacy-geometry legacy-tessellation)
    add_test(NAME pbe-packed-unorm-reject-${pbe_mixed_invalid_state}
        COMMAND pbe-packed-unorm-test ${pbe_mixed_invalid_state})
endforeach()

add_executable(pbe-nonfinite-output-test
    tests/pbe_nonfinite_output_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pbe-nonfinite-output-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pbe-nonfinite-output-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(pbe-nonfinite-output-test)
add_test(NAME pbe-nonfinite-output-unit COMMAND pbe-nonfinite-output-test)
foreach(pbe_invalid_state IN ITEMS output-mask load-size identity clear-nan blend-constant-nan)
    add_test(NAME pbe-nonfinite-reject-${pbe_invalid_state}
        COMMAND pbe-nonfinite-output-test ${pbe_invalid_state})
endforeach()

add_executable(pbe-sample-counter-test
    tests/pbe_sample_counter_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/pbe.cpp
)
target_include_directories(pbe-sample-counter-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pbe-sample-counter-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(pbe-sample-counter-test)
add_test(NAME pbe-sample-counter-unit COMMAND pbe-sample-counter-test)

add_executable(late-depth-stencil-test
    tests/late_depth_stencil_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
)
target_include_directories(late-depth-stencil-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(late-depth-stencil-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(late-depth-stencil-test)
add_test(NAME late-depth-stencil-unit COMMAND late-depth-stencil-test)

add_executable(isp-depth-attachment-test
    tests/isp_depth_attachment_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/isp.cpp
)
target_include_directories(isp-depth-attachment-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(isp-depth-attachment-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(isp-depth-attachment-test)
add_test(NAME isp-depth-attachment-unit COMMAND isp-depth-attachment-test)

add_executable(polygon-offset-test
    tests/polygon_offset_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/fragment/isp.cpp
)
target_include_directories(polygon-offset-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(polygon-offset-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(polygon-offset-test)
add_test(NAME polygon-offset-unit COMMAND polygon-offset-test)

add_executable(fragment-frontend-test
    tests/fragment_frontend_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/fragment_frontend.cpp
)
target_include_directories(fragment-frontend-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(fragment-frontend-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(fragment-frontend-test)
add_test(NAME fragment-frontend-unit COMMAND fragment-frontend-test)

add_executable(ordered-blend-pipeline-test
    tests/ordered_blend_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(ordered-blend-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(ordered-blend-pipeline-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(ordered-blend-pipeline-test)
add_test(NAME ordered-blend-pipeline-unit COMMAND ordered-blend-pipeline-test)

add_executable(indexed-geometry-test
    tests/indexed_geometry_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(indexed-geometry-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(indexed-geometry-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(indexed-geometry-test)
add_test(NAME indexed-geometry-unit COMMAND indexed-geometry-test)

add_executable(nonindexed-triangle-geometry-test
    tests/nonindexed_triangle_geometry_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(nonindexed-triangle-geometry-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(nonindexed-triangle-geometry-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(nonindexed-triangle-geometry-test)
add_test(
    NAME nonindexed-triangle-geometry-unit
    COMMAND nonindexed-triangle-geometry-test
)
add_test(
    NAME nonindexed-triangle-count-fail-closed
    COMMAND nonindexed-triangle-geometry-test invalid-count
)
add_test(
    NAME nonindexed-triangle-front-cull-reflection
    COMMAND nonindexed-triangle-geometry-test front-cull
)
add_test(
    NAME nonindexed-triangle-homogeneous-clip
    COMMAND nonindexed-triangle-geometry-test clip-w
)
add_test(
    NAME nonindexed-triangle-driver-viewport-fma
    COMMAND nonindexed-triangle-geometry-test viewport-fma
)
add_test(
    NAME nonindexed-triangle-driver-clipped-setup-precision
    COMMAND nonindexed-triangle-geometry-test clipped-setup
)
add_test(
    NAME nonindexed-triangle-driver-subpixel-rne
    COMMAND nonindexed-triangle-geometry-test subpixel-rne
)

add_executable(all-culled-geometry-test
    tests/all_culled_geometry_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(all-culled-geometry-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(all-culled-geometry-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(all-culled-geometry-test)
add_test(NAME all-culled-geometry-unit COMMAND all-culled-geometry-test)

add_executable(half-culled-geometry-test
    tests/half_culled_geometry_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(half-culled-geometry-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(half-culled-geometry-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(half-culled-geometry-test)
add_test(NAME half-culled-geometry-unit COMMAND half-culled-geometry-test)

add_executable(attribute-fetch-geometry-test
    tests/attribute_fetch_geometry_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(attribute-fetch-geometry-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(attribute-fetch-geometry-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(attribute-fetch-geometry-test)
add_test(
    NAME attribute-fetch-geometry-unit
    COMMAND attribute-fetch-geometry-test
)
add_test(
    NAME attribute-fetch-two-attribute-geometry-unit
    COMMAND attribute-fetch-geometry-test two
)
add_test(
    NAME attribute-fetch-four-attribute-geometry-unit
    COMMAND attribute-fetch-geometry-test four
)
add_test(
    NAME attribute-fetch-eight-attribute-geometry-unit
    COMMAND attribute-fetch-geometry-test eight
)
add_test(
    NAME vertex-input-overlap-fail-closed
    COMMAND attribute-fetch-geometry-test invalid-overlap
)
add_test(
    NAME vertex-input-stride-fail-closed
    COMMAND attribute-fetch-geometry-test invalid-stride
)
add_test(
    NAME vertex-input-offset-fail-closed
    COMMAND attribute-fetch-geometry-test invalid-offset
)

add_executable(varyings-shader-one-pipeline-test
    tests/varyings_shader_one_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/submitter.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(varyings-shader-one-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-one-pipeline-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-one-pipeline-test)
add_test(
    NAME varyings-shader-one-pipeline-unit
    COMMAND varyings-shader-one-pipeline-test
)

# Exercise the same complete production module chain with native derivative
# bytes and no user varyings/texture. Keep module coverage in step with the
# existing full-pipeline target above, without linking another test main.
get_target_property(pvrgpu_derivative_pipeline_sources
    varyings-shader-one-pipeline-test SOURCES)
list(REMOVE_ITEM pvrgpu_derivative_pipeline_sources
    tests/varyings_shader_one_pipeline_test.cpp)
add_executable(pco-derivative-pipeline-test
    tests/pco_derivative_pipeline_test.cpp
    ${pvrgpu_derivative_pipeline_sources})
target_include_directories(pco-derivative-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub" "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc")
target_link_libraries(pco-derivative-pipeline-test PRIVATE PkgConfig::SYSTEMC)
pvrgpu_target_warnings(pco-derivative-pipeline-test)
foreach(mode IN ITEMS front back changed-classification)
    add_test(NAME pco-derivative-pipeline-${mode}
        COMMAND pco-derivative-pipeline-test ${mode})
endforeach()

add_executable(varyings-shader-two-pipeline-test
    tests/varyings_shader_two_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/submitter.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/data_master/vertex_data_master.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(varyings-shader-two-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-two-pipeline-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-two-pipeline-test)
add_test(
    NAME varyings-shader-two-pipeline-unit
    COMMAND varyings-shader-two-pipeline-test
)

add_executable(varyings-shader-two-perspective-test
    tests/varyings_shader_two_perspective_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(varyings-shader-two-perspective-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-two-perspective-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-two-perspective-test)
add_test(
    NAME varyings-shader-two-perspective-unit
    COMMAND varyings-shader-two-perspective-test
)

add_executable(varyings-shader-four-pipeline-test
    tests/varyings_shader_four_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/submitter.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/data_master/vertex_data_master.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(varyings-shader-four-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-four-pipeline-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-four-pipeline-test)
add_test(
    NAME varyings-shader-four-pipeline-unit
    COMMAND varyings-shader-four-pipeline-test
)

add_executable(varyings-shader-four-perspective-test
    tests/varyings_shader_four_perspective_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(varyings-shader-four-perspective-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-four-perspective-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-four-perspective-test)
add_test(
    NAME varyings-shader-four-perspective-unit
    COMMAND varyings-shader-four-perspective-test
)

add_executable(varyings-shader-eight-pipeline-test
    tests/varyings_shader_eight_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pco_sequence_profiles.cpp
    model_stub/submitter.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/data_master/vertex_data_master.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/pbe_write_back.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
    src/systemc/pds/vertex_pds_engine.cpp
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
    src/systemc/texture/astc_decoder.cpp
    src/systemc/texture/texture_filter.cpp
    src/systemc/texture/texture_unit.cpp
)
target_include_directories(varyings-shader-eight-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-eight-pipeline-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-eight-pipeline-test)
add_test(
    NAME varyings-shader-eight-pipeline-unit
    COMMAND varyings-shader-eight-pipeline-test
)

add_executable(varyings-shader-eight-perspective-test
    tests/varyings_shader_eight_perspective_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(varyings-shader-eight-perspective-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-eight-perspective-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(varyings-shader-eight-perspective-test)
add_test(
    NAME varyings-shader-eight-perspective-unit
    COMMAND varyings-shader-eight-perspective-test
)

add_executable(parameter-buffer-perspective-test
    tests/parameter_buffer_perspective_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(parameter-buffer-perspective-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(parameter-buffer-perspective-test PRIVATE
    PkgConfig::SYSTEMC
)
pvrgpu_target_warnings(parameter-buffer-perspective-test)
add_test(
    NAME parameter-buffer-perspective-unit
    COMMAND parameter-buffer-perspective-test
)
foreach(plane_case IN ITEMS
        empty-depth empty-varying empty-reciprocal rollback covered zero-mask
        scissor-empty scissor-boundary framebuffer-clipped
        msaa-covered msaa-masked msaa-disabled-covered msaa-disabled-empty
        msaa-enabled-scissored msaa-2-empty msaa-4-empty msaa-8-empty
        invalid-reciprocal invalid-layout invalid-setup invalid-samples
        invalid-multisample top-left-covered bottom-left-empty)
    add_test(
        NAME parameter-buffer-plane-${plane_case}
        COMMAND parameter-buffer-perspective-test ${plane_case}
    )
endforeach()

add_test(
    NAME pvrgpu-model-stub-smoke
    COMMAND pvrgpu-model-stub --frames 2 --width 64 --height 64 --case fill_solid
)
set_tests_properties(
    pvrgpu-model-stub-smoke
    PROPERTIES PASS_REGULAR_EXPRESSION "pvrgpu.counter.v1"
)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME drawlist-replay-cli-test
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_drawlist_replay_cli.py")
    add_test(NAME drawlist-checkpoint-integration-test
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_drawlist_checkpoint_integration.py")
    add_test(NAME build-renderdoc-snapshot-test
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_build_renderdoc_snapshot.py")
    add_test(NAME inspect-drawlist-snapshot-test
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_inspect_drawlist_snapshot.py")
    add_test(NAME drawlist-snapshot-rejection-test
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_drawlist_snapshot_rejection.py")
    add_test(NAME rdc-native-runner-process-test
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONDONTWRITEBYTECODE=1"
                "PVRGPU_NATIVE_RUNNER_TEST_BIN=$<TARGET_FILE:pvrgpu>"
                "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_rdc_native_runner.py")
    add_test(
        NAME systemc-module-layout
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_SOURCE_DIR}/tests/check_systemc_module_layout.py"
                --root "${CMAKE_SOURCE_DIR}"
    )
    add_test(
        NAME counter-protocol-tests
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_SOURCE_DIR}/tests" -v
    )
    add_test(
        NAME systemc-pipeline-integration
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_SOURCE_DIR}/tests/check_model_pipeline.py"
                "$<TARGET_FILE:pvrgpu-model-stub>"
    )
    set_tests_properties(
        counter-protocol-tests
        PROPERTIES ENVIRONMENT
            "PYTHONDONTWRITEBYTECODE=1;PYTHONPATH=${CMAKE_SOURCE_DIR}/tools"
    )
endif()
