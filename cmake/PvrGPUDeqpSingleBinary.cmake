# Inject a desktop, multi-package dEQP executable into a VK-GL-CTS build.
#
# This file is loaded through CMAKE_PROJECT_INCLUDE.  It intentionally leaves
# the external VK-GL-CTS source checkout untouched and adds the target only
# after the upstream top-level CMakeLists.txt has created its package targets.

if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    return()
endif()

get_property(pvrgpu_deqp_injection_scheduled GLOBAL
    PROPERTY PVRGPU_DEQP_INJECTION_SCHEDULED)
if(pvrgpu_deqp_injection_scheduled)
    return()
endif()
set_property(GLOBAL PROPERTY PVRGPU_DEQP_INJECTION_SCHEDULED TRUE)

function(pvrgpu_add_deqp_single_binary)
    foreach(required_target IN ITEMS
            tcutil
            tcutil-platform
            deqp-gles2-package
            deqp-gles3-package
            deqp-gles31-package
            deqp-egl-package)
        if(NOT TARGET "${required_target}")
            message(FATAL_ERROR
                "PvrGPU live dEQP requires VK-GL-CTS target ${required_target}")
        endif()
    endforeach()

    foreach(required_value IN ITEMS
            PVRGPU_DEQP_RUNNER_MAIN
            PVRGPU_DEQP_OUTPUT_DIR
            PVRGPU_DEQP_SYSTEMC_BRIDGE
            PVRGPU_DEQP_SYSTEMC_API_INCLUDE_DIR)
        if(NOT DEFINED ${required_value} OR "${${required_value}}" STREQUAL "")
            message(FATAL_ERROR "${required_value} must be provided")
        endif()
    endforeach()

    if(NOT EXISTS "${PVRGPU_DEQP_RUNNER_MAIN}")
        message(FATAL_ERROR
            "PvrGPU dEQP runner main does not exist: ${PVRGPU_DEQP_RUNNER_MAIN}")
    endif()
    if(NOT EXISTS "${PVRGPU_DEQP_SYSTEMC_BRIDGE}")
        message(FATAL_ERROR
            "PvrGPU SystemC bridge does not exist: ${PVRGPU_DEQP_SYSTEMC_BRIDGE}")
    endif()

    # Add a callback to the upstream executor without changing the external
    # source checkout.  The callback updates PVRGPU_RDC_CASE_NAME and switches
    # artifact paths before each case's init/iterate/deinit sequence.
    set(executor_source
        "${CMAKE_SOURCE_DIR}/framework/common/tcuTestSessionExecutor.cpp")
    file(READ "${executor_source}" executor_contents)

    set(namespace_anchor "namespace tcu\n{\n")
    string(FIND "${executor_contents}" "${namespace_anchor}"
        namespace_anchor_offset)
    if(namespace_anchor_offset EQUAL -1)
        message(FATAL_ERROR
            "Unsupported VK-GL-CTS tcuTestSessionExecutor.cpp namespace layout")
    endif()
    set(callback_support [=[
namespace
{

using PvrGpuDeqpCaseCallback = void (*)(const char *casePath);
PvrGpuDeqpCaseCallback g_pvrgpuDeqpCaseCallback = nullptr;

} // anonymous namespace

extern "C" void pvrgpuDeqpSetCaseCallback(PvrGpuDeqpCaseCallback callback)
{
    g_pvrgpuDeqpCaseCallback = callback;
}

namespace tcu
{
]=])
    string(REPLACE "${namespace_anchor}" "${callback_support}"
        patched_executor_contents "${executor_contents}")

    set(enter_anchor [=[bool TestSessionExecutor::enterTestCase(TestCase *testCase, const std::string &casePath)
{
]=])
    string(FIND "${patched_executor_contents}" "${enter_anchor}"
        enter_anchor_offset)
    if(enter_anchor_offset EQUAL -1)
        message(FATAL_ERROR
            "Unsupported VK-GL-CTS TestSessionExecutor::enterTestCase layout")
    endif()
    set(enter_replacement [=[bool TestSessionExecutor::enterTestCase(TestCase *testCase, const std::string &casePath)
{
    if (g_pvrgpuDeqpCaseCallback)
        g_pvrgpuDeqpCaseCallback(casePath.c_str());
]=])
    string(REPLACE "${enter_anchor}" "${enter_replacement}"
        patched_executor_contents "${patched_executor_contents}")

    set(leave_anchor [=[    if (m_testCtx.getWatchDog())
        qpWatchDog_reset(m_testCtx.getWatchDog());
}

TestCase::IterateResult TestSessionExecutor::iterateTestCase]=])
    string(FIND "${patched_executor_contents}" "${leave_anchor}"
        leave_anchor_offset)
    if(leave_anchor_offset EQUAL -1)
        message(FATAL_ERROR
            "Unsupported VK-GL-CTS TestSessionExecutor::leaveTestCase layout")
    endif()
    set(leave_replacement [=[    if (m_testCtx.getWatchDog())
        qpWatchDog_reset(m_testCtx.getWatchDog());

    if (g_pvrgpuDeqpCaseCallback)
        g_pvrgpuDeqpCaseCallback(nullptr);
}

TestCase::IterateResult TestSessionExecutor::iterateTestCase]=])
    string(REPLACE "${leave_anchor}" "${leave_replacement}"
        patched_executor_contents "${patched_executor_contents}")

    set(generated_executor
        "${CMAKE_BINARY_DIR}/pvrgpu-generated/tcuTestSessionExecutor.cpp")
    get_filename_component(generated_executor_dir "${generated_executor}" DIRECTORY)
    file(MAKE_DIRECTORY "${generated_executor_dir}")
    file(WRITE "${generated_executor}" "${patched_executor_contents}")

    get_target_property(tcutil_sources tcutil SOURCES)
    set(patched_tcutil_sources)
    set(found_executor_source FALSE)
    foreach(source IN LISTS tcutil_sources)
        get_filename_component(source_name "${source}" NAME)
        if(source_name STREQUAL "tcuTestSessionExecutor.cpp")
            set(found_executor_source TRUE)
        else()
            list(APPEND patched_tcutil_sources "${source}")
        endif()
    endforeach()
    if(NOT found_executor_source)
        message(FATAL_ERROR
            "VK-GL-CTS tcutil target did not expose tcuTestSessionExecutor.cpp")
    endif()
    list(APPEND patched_tcutil_sources "${generated_executor}")
    set_property(TARGET tcutil PROPERTY SOURCES "${patched_tcutil_sources}")

    set(entry_points
        "${CMAKE_SOURCE_DIR}/modules/gles2/tes2TestPackageEntry.cpp"
        "${CMAKE_SOURCE_DIR}/modules/gles3/tes3TestPackageEntry.cpp"
        "${CMAKE_SOURCE_DIR}/modules/gles31/tes31TestPackageEntry.cpp"
        "${CMAKE_SOURCE_DIR}/modules/egl/teglTestPackageEntry.cpp"
    )

    add_executable(pvrgpu-deqp
        "${PVRGPU_DEQP_RUNNER_MAIN}"
        ${entry_points}
    )
    target_compile_features(pvrgpu-deqp PRIVATE cxx_std_17)
    target_include_directories(pvrgpu-deqp PRIVATE
        "${PVRGPU_DEQP_SYSTEMC_API_INCLUDE_DIR}"
    )
    target_compile_definitions(pvrgpu-deqp PRIVATE
        PVRGPU_DEQP_DEFAULT_ARCHIVE_DIR="${PVRGPU_DEQP_OUTPUT_DIR}"
        PVRGPU_DEQP_DEFAULT_SYSTEMC_API_LIB="${PVRGPU_DEQP_SYSTEMC_BRIDGE}"
    )
    target_link_libraries(pvrgpu-deqp PRIVATE
        tcutil-platform
        deqp-gles2-package
        deqp-gles3-package
        deqp-gles31-package
        deqp-egl-package
        "${PVRGPU_DEQP_SYSTEMC_BRIDGE}"
    )
    set_target_properties(pvrgpu-deqp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PVRGPU_DEQP_OUTPUT_DIR}"
    )

    add_dependencies(pvrgpu-deqp
        deqp-gles2-data
        deqp-gles3-data
        deqp-gles31-data
        deqp-egl-data
    )
    foreach(api IN ITEMS gles2 gles3 gles31)
        add_custom_command(TARGET pvrgpu-deqp POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${CMAKE_BINARY_DIR}/modules/${api}/${api}"
                "${PVRGPU_DEQP_OUTPUT_DIR}/${api}"
            VERBATIM
        )
    endforeach()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL pvrgpu_add_deqp_single_binary)
