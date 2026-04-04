#pragma once

#if defined(_WIN32)
#    if defined(INFERRT_CVCUDA_BUILD_SHARED_LIBS)
#        define INFERRT_CVCUDA_API __declspec(dllexport)
#    else
#        define INFERRT_CVCUDA_API __declspec(dllimport)
#    endif
#else
#    define INFERRT_CVCUDA_API
#endif
