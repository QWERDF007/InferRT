# Read build-time dependency defaults from the shared runtime manifest.
# Installed packages may omit the source-tree manifest; callers then continue
# with environment and CMake prefix discovery.
function(inferrt_dependency_default dependency_name output_variable)
    set(_manifest "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/dependencies.yaml")
    if(DEFINED INFERRT_DEPENDENCY_MANIFEST AND NOT INFERRT_DEPENDENCY_MANIFEST STREQUAL "")
        set(_manifest "${INFERRT_DEPENDENCY_MANIFEST}")
    endif()

    set(_result "")
    if(EXISTS "${_manifest}")
        file(STRINGS "${_manifest}" _manifest_lines)
        set(_in_dependency OFF)
        foreach(_line IN LISTS _manifest_lines)
            string(REGEX MATCH
                "^[ ]*-[ ]+name:[ ]*([A-Za-z0-9_.-]+)[ ]*$"
                _name_match "${_line}")
            if(_name_match)
                set(_entry_name "${_name_match}")
                string(REGEX REPLACE "^[ ]*-[ ]+name:[ ]*"
                    "" _entry_name "${_entry_name}")
                if(_entry_name STREQUAL "${dependency_name}")
                    set(_in_dependency ON)
                else()
                    set(_in_dependency OFF)
                endif()
                continue()
            endif()

            if(_in_dependency)
                string(REGEX MATCH "^[ ]*default:[ ]*(.+)[ ]*$"
                    _default_match "${_line}")
                if(_default_match)
                    set(_result "${_default_match}")
                    string(REGEX REPLACE "^[ ]*default:[ ]*"
                        "" _result "${_result}")
                    string(REGEX REPLACE "^[ ]*[\"'](.*)[\"'][ ]*$"
                        "\\1" _result "${_result}")
                    break()
                endif()
            endif()
        endforeach()
    endif()

    set(${output_variable} "${_result}" PARENT_SCOPE)
endfunction()
