include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

set(Faiss_VERSION "1.7.4" CACHE STRING "Faiss version")

# Prefer a native Faiss package when one is available.  Different Faiss
# distributions use different package/target spellings, so normalize them to
# the target name exported by InferRT before falling back to root discovery.
find_package(Faiss CONFIG QUIET)
find_package(faiss CONFIG QUIET)

if(NOT TARGET Faiss::faiss)
    if(TARGET faiss::faiss)
        add_library(Faiss::faiss ALIAS faiss::faiss)
    elseif(TARGET faiss)
        add_library(Faiss::faiss ALIAS faiss)
    endif()
endif()

if(TARGET Faiss::faiss)
    get_target_property(_faiss_interface_includes Faiss::faiss INTERFACE_INCLUDE_DIRECTORIES)
    if(_faiss_interface_includes AND NOT Faiss_INCLUDE_DIRS)
        set(Faiss_INCLUDE_DIRS "${_faiss_interface_includes}")
    endif()
    set(Faiss_LIBS Faiss::faiss)
    set(Faiss_FOUND TRUE)
    unset(_faiss_interface_includes)
    return()
endif()

inferrt_dependency_resolve_path(
    _faiss_resolved_root _faiss_resolved_origin faiss
    VARIABLES Faiss_HOME Faiss_ROOT
    ENVIRONMENT_VARIABLES Faiss_HOME Faiss_ROOT FAISS_ROOT
    PREFIX_PATHS ${CMAKE_PREFIX_PATH}
    REQUIRED_FILES include/faiss/Index.h
)
if(_faiss_resolved_root)
    inferrt_dependency_cache_set(
        Faiss_HOME "${_faiss_resolved_root}" PATH
        "Faiss installation directory"
        INFERRT_DEPENDENCY_FAISS_HOME "${_faiss_resolved_origin}")
endif()

if(DEFINED Faiss_HOME)
    set(Faiss_LIBRARY_DIR "${Faiss_HOME}/lib")
    set(Faiss_INCLUDE_DIRS "${Faiss_HOME}/include")
    set(Faiss_BIN_DIR "${Faiss_HOME}/bin")
endif()

inferrt_dependency_resolve_path(
    _inferrt_mkl_root _inferrt_mkl_origin faiss-mkl-runtime
    VARIABLES MKL_ROOT MKLROOT
    ENVIRONMENT_VARIABLES MKL_ROOT MKLROOT
)
if(_inferrt_mkl_root)
    inferrt_dependency_cache_set(
        MKL_ROOT "${_inferrt_mkl_root}" PATH
        "Intel MKL installation root"
        INFERRT_DEPENDENCY_MKL_ROOT "${_inferrt_mkl_origin}")
endif()

if(DEFINED MKL_ROOT)
    set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib")
endif()

unset(_inferrt_mkl_root)
unset(_inferrt_mkl_origin)
unset(_faiss_resolved_root)
unset(_faiss_resolved_origin)

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
if(NOT Faiss_LIBS OR NOT Faiss_INCLUDE_DIRS)
    set(Faiss_FOUND FALSE)
    message(FATAL_ERROR
            "Faiss is required by InferRT::features but was not found. "
            "Install Faiss and add its prefix to CMAKE_PREFIX_PATH, or set Faiss_HOME.")
else()
    set(Faiss_FOUND TRUE)
endif()
