set(CMAKE_FOLDER 3rdparty)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(INFERRT_BUILD_TESTS)
    add_subdirectory(googletest)
endif()