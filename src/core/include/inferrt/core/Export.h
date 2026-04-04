#pragma once

#if defined(_WIN32)
#    if defined(INFERRT_CORE_BUILD_SHARED_LIBS)
#        define INFERRT_CORE_API __declspec(dllexport)
#    else
#        define INFERRT_CORE_API __declspec(dllimport)
#    endif
#else
#    define INFERRT_CORE_API
#endif
