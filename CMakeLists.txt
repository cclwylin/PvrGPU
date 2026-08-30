cmake_minimum_required(VERSION 3.20)

project(PvrGPUControlPlane VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(PkgConfig REQUIRED)
pkg_check_modules(SYSTEMC REQUIRED IMPORTED_TARGET systemc)
pkg_check_modules(LIBPNG REQUIRED IMPORTED_TARGET libpng>=1.6)

add_executable(pvrgpu-model-stub
    model_stub/json_reporter.cpp
    model_stub/mesa_command.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/pvrgpu_model_stub.cpp
    model_stub/submitter.cpp
    src/support/png_writer.cpp
    src/systemc/cache_mmu/cache_array.cpp
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
    src/systemc/data_master/two_d_data_master.cpp
    src/systemc/firmware/firmware_scheduler.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
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
target_compile_options(pvrgpu-model-stub PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(pco-iss-test
    tests/pco_iss_test.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pco-iss-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_compile_options(pco-iss-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pco-iss-unit COMMAND pco-iss-test)

add_executable(cache-array-test
    tests/cache_array_test.cpp
    src/systemc/cache_mmu/cache_array.cpp
)
target_include_directories(cache-array-test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_compile_options(cache-array-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME cache-array-unit COMMAND cache-array-test)

add_executable(cache-controller-test
    tests/cache_controller_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/cache_mmu/cache_array.cpp
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
target_compile_options(cache-controller-test PRIVATE -Wall -Wextra -Wpedantic)
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
target_compile_options(texture-unit-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME texture-unit-unit COMMAND texture-unit-test)

add_executable(pbe-blend-test
    tests/pbe_blend_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/shader/pco_iss.cpp
)
target_include_directories(pbe-blend-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(pbe-blend-test PRIVATE PkgConfig::SYSTEMC)
target_compile_options(pbe-blend-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pbe-blend-unit COMMAND pbe-blend-test)

add_executable(ordered-blend-pipeline-test
    tests/ordered_blend_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/pds/pds_engine.cpp
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
target_compile_options(ordered-blend-pipeline-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(indexed-geometry-test PRIVATE
    -Wall -Wextra -Wpedantic
)
add_test(NAME indexed-geometry-unit COMMAND indexed-geometry-test)

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
target_compile_options(all-culled-geometry-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(half-culled-geometry-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(attribute-fetch-geometry-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
    src/systemc/shader/pco_decoder.cpp
    src/systemc/shader/pco_iss.cpp
    src/systemc/shader/usc_cluster.cpp
    src/systemc/shader/usc_slot.cpp
)
target_include_directories(varyings-shader-one-pipeline-test PRIVATE
    "${CMAKE_SOURCE_DIR}/model_stub"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/systemc"
)
target_link_libraries(varyings-shader-one-pipeline-test PRIVATE
    PkgConfig::SYSTEMC
)
target_compile_options(varyings-shader-one-pipeline-test PRIVATE
    -Wall -Wextra -Wpedantic
)
add_test(
    NAME varyings-shader-one-pipeline-unit
    COMMAND varyings-shader-one-pipeline-test
)

add_executable(varyings-shader-two-pipeline-test
    tests/varyings_shader_two_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/submitter.cpp
    src/systemc/cache_mmu/cache_array.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
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
target_compile_options(varyings-shader-two-pipeline-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(varyings-shader-two-perspective-test PRIVATE
    -Wall -Wextra -Wpedantic
)
add_test(
    NAME varyings-shader-two-perspective-unit
    COMMAND varyings-shader-two-perspective-test
)

add_executable(varyings-shader-four-pipeline-test
    tests/varyings_shader_four_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/submitter.cpp
    src/systemc/cache_mmu/cache_array.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
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
target_compile_options(varyings-shader-four-pipeline-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(varyings-shader-four-perspective-test PRIVATE
    -Wall -Wextra -Wpedantic
)
add_test(
    NAME varyings-shader-four-perspective-unit
    COMMAND varyings-shader-four-perspective-test
)

add_executable(varyings-shader-eight-pipeline-test
    tests/varyings_shader_eight_pipeline_test.cpp
    model_stub/memory_pool.cpp
    model_stub/model_types.cpp
    model_stub/submitter.cpp
    src/systemc/cache_mmu/cache_array.cpp
    src/systemc/cache_mmu/slc.cpp
    src/systemc/common/functional_types.cpp
    src/systemc/common/glbench_triangle_fixture.cpp
    src/systemc/common/glbench_texture_fixture.cpp
    src/systemc/common/pipeline_state.cpp
    src/systemc/data_master/pixel_data_master.cpp
    src/systemc/fragment/fragment_frontend.cpp
    src/systemc/fragment/isp.cpp
    src/systemc/fragment/pbe.cpp
    src/systemc/fragment/tile_scheduler.cpp
    src/systemc/geometry/clip_cull.cpp
    src/systemc/geometry/parameter_buffer.cpp
    src/systemc/geometry/tiler.cpp
    src/systemc/geometry/vdm.cpp
    src/systemc/geometry/vertex_fetch.cpp
    src/systemc/memory/dram_model.cpp
    src/systemc/pds/pds_engine.cpp
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
target_compile_options(varyings-shader-eight-pipeline-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(varyings-shader-eight-perspective-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
target_compile_options(parameter-buffer-perspective-test PRIVATE
    -Wall -Wextra -Wpedantic
)
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
                "${CMAKE_SOURCE_DIR}/scripts/check-systemc-module-layout.py"
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
