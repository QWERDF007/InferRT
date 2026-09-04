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

# CMake does not expose whether an existing cache entry came from an earlier
# find_package call or from the current command line.  Keep that provenance in
# project-owned metadata after the first resolution and use the cache type/help
# as the compatibility signal for legacy build trees.
function(inferrt_dependency_variable_is_explicit variable marker output_variable)
    set(_explicit OFF)
    set(_origin_key "${marker}_ORIGIN")
    set(_last_value_key "${marker}_LAST_VALUE")

    get_property(_cache_set CACHE "${variable}" PROPERTY TYPE SET)
    if(DEFINED ${variable})
        if(NOT _cache_set)
            # A normal variable supplied by the parent project is explicit.
            set(_explicit ON)
        else()
            get_property(_cache_value CACHE "${variable}" PROPERTY VALUE)
            if(NOT "${${variable}}" STREQUAL "${_cache_value}")
                # A normal variable shadows the cache entry in this scope.
                set(_explicit ON)
            elseif(DEFINED ${_origin_key} AND "${${_origin_key}}" STREQUAL "user")
                set(_explicit ON)
            elseif(DEFINED ${_last_value_key}
                   AND NOT "${${variable}}" STREQUAL "${${_last_value_key}}")
                # The cache was changed before this configure pass, normally
                # by a new -D value.
                set(_explicit ON)
            else()
                get_property(_cache_type CACHE "${variable}" PROPERTY TYPE)
                get_property(_cache_help CACHE "${variable}" PROPERTY HELPSTRING)
                if(_cache_type STREQUAL "UNINITIALIZED" OR _cache_help STREQUAL "")
                    # Untyped/undocumented cache entries are how command-line
                    # -D values are represented on a fresh build tree.
                    set(_explicit ON)
                endif()
            endif()
        endif()
    endif()

    set(${output_variable} "${_explicit}" PARENT_SCOPE)
endfunction()

# Resolve a dependency root with one shared precedence rule.  Cache entries
# created by an earlier configure pass are deliberately considered last so a
# new manifest default can repair a stale build tree without overriding a
# current command-line or parent-project selection.
function(inferrt_dependency_resolve_path output_variable origin_variable dependency_name)
    set(_options)
    set(_one_value_args)
    set(_multi_value_args VARIABLES ENVIRONMENT_VARIABLES PREFIX_PATHS REQUIRED_FILES)
    cmake_parse_arguments(
        _inferrt_resolve
        "${_options}"
        "${_one_value_args}"
        "${_multi_value_args}"
        ${ARGN}
    )

    set(_resolved_value)
    set(_resolved_origin)

    foreach(_variable IN LISTS _inferrt_resolve_VARIABLES)
        if(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL ""
           OR "${${_variable}}" MATCHES "-NOTFOUND$")
            continue()
        endif()
        # Provenance belongs to the cache variable, not the dependency alias.
        # A dependency can expose aliases such as tensorrt/TRT_ROOT, while
        # the resolved value is recorded under the canonical cache variable.
        string(TOUPPER "${_variable}" _variable_key)
        string(REGEX REPLACE "[^A-Z0-9_]" "_" _variable_key "${_variable_key}")
        inferrt_dependency_variable_is_explicit(
            "${_variable}"
            "INFERRT_DEPENDENCY_${_variable_key}"
            _variable_explicit)
        if(_variable_explicit)
            set(_resolved_value "${${_variable}}")
            set(_resolved_origin "user")
            break()
        endif()
    endforeach()

    if(NOT _resolved_value)
        foreach(_environment_variable IN LISTS _inferrt_resolve_ENVIRONMENT_VARIABLES)
            if(DEFINED ENV{${_environment_variable}}
               AND NOT "$ENV{${_environment_variable}}" STREQUAL ""
               AND NOT "$ENV{${_environment_variable}}" MATCHES "-NOTFOUND$")
                set(_resolved_value "$ENV{${_environment_variable}}")
                set(_resolved_origin "environment")
                break()
            endif()
        endforeach()
    endif()

    if(NOT _resolved_value AND _inferrt_resolve_PREFIX_PATHS
       AND _inferrt_resolve_REQUIRED_FILES)
        foreach(_prefix IN LISTS _inferrt_resolve_PREFIX_PATHS)
            if("${_prefix}" STREQUAL "")
                continue()
            endif()
            set(_prefix_valid ON)
            foreach(_required_file IN LISTS _inferrt_resolve_REQUIRED_FILES)
                if(NOT EXISTS "${_prefix}/${_required_file}")
                    set(_prefix_valid OFF)
                    break()
                endif()
            endforeach()
            if(_prefix_valid)
                get_filename_component(_resolved_value "${_prefix}" ABSOLUTE)
                set(_resolved_origin "cmake-prefix")
                break()
            endif()
        endforeach()
    endif()

    if(NOT _resolved_value)
        inferrt_dependency_default("${dependency_name}" _default_value)
        if(_default_value AND EXISTS "${_default_value}")
            get_filename_component(_resolved_value "${_default_value}" ABSOLUTE)
            set(_resolved_origin "project-default")
        endif()
    endif()

    if(NOT _resolved_value)
        foreach(_variable IN LISTS _inferrt_resolve_VARIABLES)
            if(DEFINED ${_variable} AND NOT "${${_variable}}" STREQUAL ""
               AND NOT "${${_variable}}" MATCHES "-NOTFOUND$")
                set(_resolved_value "${${_variable}}")
                set(_resolved_origin "legacy-cache")
                break()
            endif()
        endforeach()
    endif()

    set(${output_variable} "${_resolved_value}" PARENT_SCOPE)
    set(${origin_variable} "${_resolved_origin}" PARENT_SCOPE)
endfunction()

function(inferrt_dependency_set_internal key value description)
    get_property(_cache_set CACHE "${key}" PROPERTY TYPE SET)
    if(_cache_set)
        set_property(CACHE "${key}" PROPERTY VALUE "${value}")
        set_property(CACHE "${key}" PROPERTY TYPE INTERNAL)
    else()
        set("${key}" "${value}" CACHE INTERNAL "${description}")
    endif()
    set("${key}" "${value}" PARENT_SCOPE)
endfunction()

function(inferrt_dependency_record_value marker value origin)
    inferrt_dependency_set_internal(
        "${marker}_ORIGIN" "${origin}" "InferRT dependency value origin")
    inferrt_dependency_set_internal(
        "${marker}_LAST_VALUE" "${value}" "Last resolved InferRT dependency value")
endfunction()

function(inferrt_dependency_cache_set variable value type description marker origin)
    get_property(_cache_set CACHE "${variable}" PROPERTY TYPE SET)
    if(_cache_set)
        set_property(CACHE "${variable}" PROPERTY VALUE "${value}")
        set_property(CACHE "${variable}" PROPERTY TYPE "${type}")
    else()
        set("${variable}" "${value}" CACHE "${type}" "${description}")
    endif()
    set("${variable}" "${value}" PARENT_SCOPE)
    inferrt_dependency_record_value("${marker}" "${value}" "${origin}")
endfunction()
