cmake_minimum_required(VERSION 3.20)

project(PvrGPUControlPlane VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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

add_library(pvrgpu-rdc-runner-support STATIC
    src/rdc_runner/process.cpp
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

add_executable(pvrgpu-model-stub
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
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
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
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
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

add_executable(pco-sequence-profiles-test
    tests/pco_sequence_profiles_test.cpp
    model_stub/pco_sequence_profiles.cpp
)
target_include_directories(pco-sequence-profiles-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
)
pvrgpu_target_warnings(pco-sequence-profiles-test)
add_test(NAME pco-sequence-profiles-unit COMMAND pco-sequence-profiles-test)

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

add_executable(usc-cluster-texture-continuation-test
    tests/usc_cluster_texture_continuation_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
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
    NAME usc-cluster-texture-response-order-fail-closed
    COMMAND usc-cluster-texture-continuation-test response-order
)

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
