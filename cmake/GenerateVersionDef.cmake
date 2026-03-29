if(NOT DEFINED INPUT)
    message(FATAL_ERROR "INPUT not set")
endif()
if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "OUTPUT not set")
endif()

# 兼容某些生成器/构建工具链在传递 -DINPUT/-DOUTPUT 时把引号包含进变量值的情况
# （Windows/MSBuild 下可能导致路径含有 '"'，从而使 file()/configure_file() 失败）
if(INPUT MATCHES "^\"(.*)\"$")
    set(INPUT "${CMAKE_MATCH_1}")
endif()
if(OUTPUT MATCHES "^\"(.*)\"$")
    set(OUTPUT "${CMAKE_MATCH_1}")
endif()

# 确保输出目录存在（configure_file 不会自动创建中间目录）
get_filename_component(_out_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")

# 在 build 阶段生成构建时间：每次执行该脚本都会得到新的时间值
string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M:%S %z")

# 以 @ONLY 方式用变量替换模板中的 @...@ 占位符（例如 @BUILD_TIME@）
configure_file("${INPUT}" "${OUTPUT}" @ONLY ESCAPE_QUOTES)
