set(OpenCV_HOME "/home/pc/github/opencv/install" CACHE PATH  "OpenCV Home" FORCE)
set(OpenCV_DIR "${OpenCV_HOME}/lib/cmake/opencv4" CACHE PATH "OpenCV DIR"  FORCE) # dir contain .cmake
set(OpenCV_LIBRARY_DIR ${OpenCV_DIR})
set(OpenCV_BIN_DIR "${OpenCV_HOME}/bin")
find_package(OpenCV REQUIRED) 
