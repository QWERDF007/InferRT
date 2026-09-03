# Use a versioned prefix only when CMake has not received a user prefix.
# Keep provenance in the cache so a project-owned default can follow version
# changes without rewriting a user-selected path.
set(_inferrt_default_install_prefix
    "${CMAKE_SOURCE_DIR}/${PROJECT_NAME}-${PROJECT_VERSION}")

if(NOT DEFINED INFERRT_INSTALL_PREFIX_ORIGIN)
    set(_inferrt_project_owns_prefix OFF)
    if(NOT DEFINED CMAKE_INSTALL_PREFIX)
        set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}"
            CACHE PATH "Install path prefix")
        set(_inferrt_project_owns_prefix ON)
    elseif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE
                     "${_inferrt_default_install_prefix}")
        set(_inferrt_project_owns_prefix ON)
    endif()

    if(_inferrt_project_owns_prefix)
        set(INFERRT_INSTALL_PREFIX_ORIGIN "project-default"
            CACHE INTERNAL "Origin of CMAKE_INSTALL_PREFIX")
        set(INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT
            "${_inferrt_default_install_prefix}"
            CACHE INTERNAL "Last project-generated install prefix")
    else()
        set(INFERRT_INSTALL_PREFIX_ORIGIN "user"
            CACHE INTERNAL "Origin of CMAKE_INSTALL_PREFIX")
    endif()
elseif(INFERRT_INSTALL_PREFIX_ORIGIN STREQUAL "project-default")
    if(DEFINED INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT
       AND DEFINED CMAKE_INSTALL_PREFIX)
        file(TO_CMAKE_PATH "${CMAKE_INSTALL_PREFIX}" _inferrt_current_prefix)
        file(TO_CMAKE_PATH "${INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT}"
             _inferrt_previous_prefix)
        if(WIN32)
            string(TOLOWER "${_inferrt_current_prefix}" _inferrt_current_prefix)
            string(TOLOWER "${_inferrt_previous_prefix}" _inferrt_previous_prefix)
        endif()
        string(REGEX REPLACE "/+$" "" _inferrt_current_prefix
               "${_inferrt_current_prefix}")
        string(REGEX REPLACE "/+$" "" _inferrt_previous_prefix
               "${_inferrt_previous_prefix}")

        if(_inferrt_current_prefix STREQUAL _inferrt_previous_prefix)
            set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE
                         "${_inferrt_default_install_prefix}")
            set_property(CACHE INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT PROPERTY VALUE
                         "${_inferrt_default_install_prefix}")
        else()
            set_property(CACHE INFERRT_INSTALL_PREFIX_ORIGIN PROPERTY VALUE "user")
        endif()
    else()
        set_property(CACHE INFERRT_INSTALL_PREFIX_ORIGIN PROPERTY VALUE "user")
    endif()
endif()

unset(_inferrt_current_prefix)
unset(_inferrt_default_install_prefix)
unset(_inferrt_previous_prefix)
unset(_inferrt_project_owns_prefix)
