include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

set(Faiss_VERSION "1.7.4" CACHE STRING "Faiss version")

if(NOT DEFINED Faiss_HOME OR Faiss_HOME STREQUAL "")
    set(_faiss_root_hint)
    if(DEFINED Faiss_ROOT AND NOT Faiss_ROOT STREQUAL "")
        set(_faiss_root_hint "${Faiss_ROOT}")
    else()
        foreach(_faiss_root_environment IN ITEMS Faiss_HOME Faiss_ROOT FAISS_ROOT)
            if(DEFINED ENV{${_faiss_root_environment}} AND NOT "$ENV{${_faiss_root_environment}}" STREQUAL "")
                set(_faiss_root_hint "$ENV{${_faiss_root_environment}}")
                break()
            endif()
        endforeach()
    endif()

    if(_faiss_root_hint)
        set(Faiss_HOME "${_faiss_root_hint}" CACHE PATH
            "Faiss installation directory")
    else()
        find_path(_faiss_include_dir
            NAMES faiss/Index.h
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES include)
        if(_faiss_include_dir)
            get_filename_component(_faiss_detected_home "${_faiss_include_dir}" DIRECTORY)
            set(Faiss_HOME "${_faiss_detected_home}" CACHE PATH
                "Faiss installation directory")
        endif()
    endif()

    if(NOT DEFINED Faiss_HOME OR Faiss_HOME STREQUAL "")
        if(WIN32)
            inferrt_dependency_default(faiss _faiss_default_root)
            if(_faiss_default_root)
                set(Faiss_HOME "${_faiss_default_root}" CACHE PATH
                    "Faiss installation directory")
            endif()
        endif()
    endif()

    unset(_faiss_detected_home)
    unset(_faiss_include_dir CACHE)
    unset(_faiss_include_dir)
    unset(_faiss_root_hint)
    unset(_faiss_default_root)
endif()

if(DEFINED Faiss_HOME)
    set(Faiss_LIBRARY_DIR "${Faiss_HOME}/lib")
    set(Faiss_INCLUDE_DIRS "${Faiss_HOME}/include")
    set(Faiss_BIN_DIR "${Faiss_HOME}/bin")
endif()

if(NOT DEFINED MKL_ROOT)
    if(DEFINED ENV{MKL_ROOT})
        set(MKL_ROOT "$ENV{MKL_ROOT}" CACHE PATH "Intel MKL installation root")
    elseif(DEFINED MKLROOT)
        set(MKL_ROOT "${MKLROOT}" CACHE PATH "Intel MKL installation root")
    elseif(DEFINED ENV{MKLROOT})
        set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "Intel MKL installation root")
    endif()
endif()

if(DEFINED MKL_ROOT)
    set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib")
endif()

find_library(Faiss_LIB_RELEASE faiss HINTS ${Faiss_LIBRARY_DIR} PATH_SUFFIXES lib lib64)
find_library(Faiss_LIB_DEBUG faissd HINTS ${Faiss_LIBRARY_DIR} PATH_SUFFIXES lib lib64)

set(Faiss_DEPENDENCY_LIBS)
if (UNIX AND DEFINED MKL_LIBRARY_DIR)
    find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64 HINTS ${MKL_LIBRARY_DIR})
    find_library(MKL_GNU_THREAD_LIB mkl_gnu_thread HINTS ${MKL_LIBRARY_DIR})
    find_library(MKL_CORE_LIB mkl_core HINTS ${MKL_LIBRARY_DIR})

    foreach(_Faiss_DEPENDENCY_LIB IN ITEMS MKL_INTEL_LP64_LIB MKL_GNU_THREAD_LIB MKL_CORE_LIB)
        if(${_Faiss_DEPENDENCY_LIB})
            list(APPEND Faiss_DEPENDENCY_LIBS ${${_Faiss_DEPENDENCY_LIB}})
        endif()
    endforeach()
endif()

set(Faiss_LIBS)
if(Faiss_LIB_RELEASE AND Faiss_LIB_DEBUG)
    list(APPEND Faiss_LIBS debug ${Faiss_LIB_DEBUG} optimized ${Faiss_LIB_RELEASE} ${Faiss_DEPENDENCY_LIBS})
elseif(Faiss_LIB_RELEASE)
    list(APPEND Faiss_LIBS ${Faiss_LIB_RELEASE} ${Faiss_DEPENDENCY_LIBS})
elseif(Faiss_LIB_DEBUG)
    list(APPEND Faiss_LIBS ${Faiss_LIB_DEBUG} ${Faiss_DEPENDENCY_LIBS})
endif()

# Export a relocatable dependency target instead of embedding the build
# machine's absolute library paths in InferRTTargets.cmake.
if(NOT TARGET Faiss::faiss)
    if(WIN32)
        add_library(Faiss::faiss SHARED IMPORTED GLOBAL)
    else()
        add_library(Faiss::faiss UNKNOWN IMPORTED GLOBAL)
    endif()
    if(Faiss_LIB_RELEASE)
        if(WIN32)
            set_target_properties(Faiss::faiss PROPERTIES
                IMPORTED_LOCATION_RELEASE "${Faiss_BIN_DIR}/faiss.dll"
                IMPORTED_IMPLIB_RELEASE "${Faiss_LIB_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${Faiss_BIN_DIR}/faiss.dll"
                IMPORTED_IMPLIB_RELWITHDEBINFO "${Faiss_LIB_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${Faiss_BIN_DIR}/faiss.dll"
                IMPORTED_IMPLIB_MINSIZEREL "${Faiss_LIB_RELEASE}")
        else()
            set_target_properties(Faiss::faiss PROPERTIES
                IMPORTED_LOCATION_RELEASE "${Faiss_LIB_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${Faiss_LIB_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${Faiss_LIB_RELEASE}")
        endif()
        set_property(TARGET Faiss::faiss APPEND PROPERTY IMPORTED_CONFIGURATIONS
                     RELEASE RELWITHDEBINFO MINSIZEREL)
    endif()
    if(Faiss_LIB_DEBUG)
        if(WIN32)
            set_target_properties(Faiss::faiss PROPERTIES
                IMPORTED_LOCATION_DEBUG "${Faiss_BIN_DIR}/faissd.dll"
                IMPORTED_IMPLIB_DEBUG "${Faiss_LIB_DEBUG}")
        else()
            set_target_properties(Faiss::faiss PROPERTIES
                IMPORTED_LOCATION_DEBUG "${Faiss_LIB_DEBUG}")
        endif()
        set_property(TARGET Faiss::faiss APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    endif()
    if(Faiss_LIB_RELEASE AND NOT Faiss_LIB_DEBUG)
        set_target_properties(Faiss::faiss PROPERTIES MAP_IMPORTED_CONFIG_DEBUG RELEASE)
    elseif(Faiss_LIB_DEBUG AND NOT Faiss_LIB_RELEASE)
        set_target_properties(Faiss::faiss PROPERTIES
            MAP_IMPORTED_CONFIG_RELEASE DEBUG
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO DEBUG
            MAP_IMPORTED_CONFIG_MINSIZEREL DEBUG)
    endif()
    if(Faiss_INCLUDE_DIRS)
        set_target_properties(Faiss::faiss PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Faiss_INCLUDE_DIRS}")
    endif()
endif()
if(Faiss_LIBS AND Faiss_INCLUDE_DIRS)
    set(Faiss_FOUND TRUE)
else()
    set(Faiss_FOUND FALSE)
endif()
