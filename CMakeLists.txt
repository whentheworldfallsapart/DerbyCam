cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(OpenCVReplay VERSION 0.1.0)

find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
add_executable(OpenCVReplay camera.cpp)
TARGET_LINK_LIBRARIES (OpenCVReplay ${OpenCV_LIBS})