#pragma once

// 根据编译选项决定是否暴露文件、行号、代码调用
#if IRT_EXPOSE_CODE
#    define IRT_SOURCE_FILE_NAME      __FILE__
#    define IRT_SOURCE_FILE_LINENO    __LINE__
#    define IRT_OPTIONAL_STRINGIFY(X) #X
#else
#    define IRT_SOURCE_FILE_NAME      ""
#    define IRT_SOURCE_FILE_LINENO    0
#    define IRT_OPTIONAL_STRINGIFY(X) ""
#endif