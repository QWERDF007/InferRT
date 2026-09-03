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
        set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}")
        set(_inferrt_project_owns_prefix ON)
    elseif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}"
            CACHE PATH "Install path prefix" FORCE)
        set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}")
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
        if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
            file(TO_CMAKE_PATH "${_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX}"
                 _inferrt_last_install_prefix)
        endif()
        if(WIN32)
            string(TOLOWER "${_inferrt_current_prefix}" _inferrt_current_prefix)
            string(TOLOWER "${_inferrt_previous_prefix}" _inferrt_previous_prefix)
            if(DEFINED _inferrt_last_install_prefix)
                string(TOLOWER "${_inferrt_last_install_prefix}" _inferrt_last_install_prefix)
            endif()
        endif()
        string(REGEX REPLACE "/+$" "" _inferrt_current_prefix
               "${_inferrt_current_prefix}")
        string(REGEX REPLACE "/+$" "" _inferrt_previous_prefix
               "${_inferrt_previous_prefix}")
        if(DEFINED _inferrt_last_install_prefix)
            string(REGEX REPLACE "/+$" "" _inferrt_last_install_prefix
                   "${_inferrt_last_install_prefix}")
        endif()

        if(_inferrt_current_prefix STREQUAL _inferrt_previous_prefix
           OR (DEFINED _inferrt_last_install_prefix
               AND _inferrt_current_prefix STREQUAL _inferrt_last_install_prefix))
            set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}"
                CACHE PATH "Install path prefix" FORCE)
            set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}")
            set(INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT
                "${_inferrt_default_install_prefix}" CACHE INTERNAL
                "Last project-generated install prefix" FORCE)
        else()
            set_property(CACHE INFERRT_INSTALL_PREFIX_ORIGIN PROPERTY VALUE "user")
        endif()
    else()
        set_property(CACHE INFERRT_INSTALL_PREFIX_ORIGIN PROPERTY VALUE "user")
    endif()
elseif(INFERRT_INSTALL_PREFIX_ORIGIN STREQUAL "user"
       AND DEFINED INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT
       AND DEFINED CMAKE_INSTALL_PREFIX
       AND DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
    file(TO_CMAKE_PATH "${CMAKE_INSTALL_PREFIX}" _inferrt_current_prefix)
    file(TO_CMAKE_PATH "${_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX}"
         _inferrt_last_install_prefix)
    file(TO_CMAKE_PATH "${INFERRT_INSTALL_PREFIX_PROJECT_DEFAULT}"
         _inferrt_previous_prefix)
    file(TO_CMAKE_PATH "${_inferrt_default_install_prefix}"
         _inferrt_current_default_prefix)
    if(WIN32)
        string(TOLOWER "${_inferrt_current_prefix}" _inferrt_current_prefix)
        string(TOLOWER "${_inferrt_last_install_prefix}" _inferrt_last_install_prefix)
        string(TOLOWER "${_inferrt_previous_prefix}" _inferrt_previous_prefix)
        string(TOLOWER "${_inferrt_current_default_prefix}"
               _inferrt_current_default_prefix)
    endif()
    string(REGEX REPLACE "/+$" "" _inferrt_current_prefix
           "${_inferrt_current_prefix}")
    string(REGEX REPLACE "/+$" "" _inferrt_last_install_prefix
           "${_inferrt_last_install_prefix}")
    string(REGEX REPLACE "/+$" "" _inferrt_previous_prefix
           "${_inferrt_previous_prefix}")
    string(REGEX REPLACE "/+$" "" _inferrt_current_default_prefix
           "${_inferrt_current_default_prefix}")

    if(_inferrt_current_prefix STREQUAL _inferrt_last_install_prefix
       AND _inferrt_previous_prefix STREQUAL _inferrt_current_default_prefix)
        set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}"
            CACHE PATH "Install path prefix" FORCE)
        set(CMAKE_INSTALL_PREFIX "${_inferrt_default_install_prefix}")
        set(INFERRT_INSTALL_PREFIX_ORIGIN "project-default"
            CACHE INTERNAL "Origin of CMAKE_INSTALL_PREFIX" FORCE)
    endif()
endif()

unset(_inferrt_current_prefix)
unset(_inferrt_default_install_prefix)
unset(_inferrt_current_default_prefix)
unset(_inferrt_last_install_prefix)
unset(_inferrt_previous_prefix)
unset(_inferrt_project_owns_prefix)
