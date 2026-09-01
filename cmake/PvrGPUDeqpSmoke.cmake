# Run one exact live dEQP case and require evidence from every layer of the
# dEQP -> Mesa/PvrGPU -> SystemC path. This catches stale Mesa/SystemC ABI
# combinations that can otherwise leave dEQP itself reporting Pass.

if(NOT DEFINED RUNNER OR RUNNER STREQUAL "")
    message(FATAL_ERROR "RUNNER must name the pvrgpu-deqp executable")
endif()
if(NOT EXISTS "${RUNNER}")
    message(FATAL_ERROR "pvrgpu-deqp does not exist: ${RUNNER}")
endif()
if(NOT DEFINED OUTPUT_DIR OR OUTPUT_DIR STREQUAL "")
    message(FATAL_ERROR "OUTPUT_DIR must be provided")
endif()

set(case_name "dEQP-GLES2.functional.prerequisite.clear_color")
set(case_dir "${OUTPUT_DIR}/cases/${case_name}")
file(REMOVE_RECURSE "${OUTPUT_DIR}")

execute_process(
    COMMAND
        "${RUNNER}"
        "--pvrgpu-output-dir=${OUTPUT_DIR}"
        "--deqp-case=${case_name}"
        "--deqp-gl-config-name=rgba8888d24s8ms0"
        "--deqp-log-images=disable"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_stdout
    ERROR_VARIABLE run_stderr
)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
        "pvrgpu-deqp returned ${run_result}\nstdout:\n${run_stdout}\nstderr:\n${run_stderr}")
endif()

set(required_files
    "${OUTPUT_DIR}/run.txt"
    "${OUTPUT_DIR}/results.qpa"
    "${case_dir}/driver-command.txt"
    "${case_dir}/driver-counter.txt"
    "${case_dir}/systemc.jsonl"
)
foreach(required_file IN LISTS required_files)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "missing live dEQP artifact: ${required_file}")
    endif()
endforeach()

file(READ "${OUTPUT_DIR}/results.qpa" qpa)
foreach(qpa_evidence IN ITEMS
        "#sessionInfo vendor \"PvrGPU\""
        "#sessionInfo renderer \"PvrGPU SystemC Gallium bring-up\""
        "<Result StatusCode=\"Pass\">Pass</Result>")
    string(FIND "${qpa}" "${qpa_evidence}" qpa_evidence_offset)
    if(qpa_evidence_offset EQUAL -1)
        message(FATAL_ERROR "results.qpa lacks evidence: ${qpa_evidence}")
    endif()
endforeach()

file(READ "${case_dir}/driver-command.txt" driver_command)
string(FIND "${driver_command}" "case=${case_name}" command_case_offset)
if(command_case_offset EQUAL -1)
    message(FATAL_ERROR "driver command did not preserve the dEQP case name")
endif()

file(READ "${case_dir}/driver-counter.txt" driver_counter)
string(FIND "${driver_counter}" "event=systemc_api_error" systemc_error_offset)
if(NOT systemc_error_offset EQUAL -1)
    message(FATAL_ERROR "driver counter contains a SystemC API error")
endif()
string(FIND "${driver_counter}"
    "event=systemc_api_done command=clear_color case=${case_name}"
    systemc_done_offset)
if(systemc_done_offset EQUAL -1)
    message(FATAL_ERROR "driver counter lacks the SystemC submit completion")
endif()

file(READ "${case_dir}/systemc.jsonl" systemc_jsonl)
string(FIND "${systemc_jsonl}"
    "\"driver_command_case\":\"${case_name}\""
    systemc_case_offset)
if(systemc_case_offset EQUAL -1)
    message(FATAL_ERROR "SystemC JSONL did not preserve the dEQP case name")
endif()
string(FIND "${systemc_jsonl}" "\"type\":\"done\"" systemc_done_record_offset)
if(systemc_done_record_offset EQUAL -1)
    message(FATAL_ERROR "SystemC JSONL has no terminal done record")
endif()

file(GLOB systemc_pngs "${case_dir}/systemc/*.png")
if(NOT systemc_pngs)
    message(FATAL_ERROR "SystemC produced no framebuffer PNG")
endif()

message(STATUS "live dEQP clear smoke passed with ${systemc_pngs}")
