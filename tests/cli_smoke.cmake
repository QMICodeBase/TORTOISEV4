execute_process(COMMAND "${PROGRAM}" RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
# These CLIs deliberately exit 1 after printing their no-argument usage menu.
if(NOT "${status}" STREQUAL "1" OR NOT output MATCHES "${EXPECTED}")
    message(FATAL_ERROR "CLI smoke check failed (${status}):\n${output}\n${error}")
endif()
