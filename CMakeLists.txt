#cmake最小需要版本
cmake_minimum_required(VERSION 2.8.3)

#项目名字
project(RunwayDetection)

# 指定 c++ 标准
set(CMAKE_CXX_STANDARD 11)

# 设置源码根路径，即 cpp 目录
set(SRC_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)


#include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
#set(OpenCV_DIR /home/wbzhang/OPENCV/opencv-3.2.0/build)
#set(OpenCV_DIR /home/wbzhang/OPENCV/opencv-3.1.0/build)
set(OpenCV_FOUND 1)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#包含原程序,即把给定目录下的源程序复制给变量DIR_SRC
aux_source_directory(DIR_SRC ./)

#生成程序
add_executable(runwaydetection src/runwaydetection.cpp include/utility.h)
add_executable(fastlineDetector src/FastLineDetector.cpp)
add_executable(houghlineDetector src/HoughLineDetector.cpp)
add_executable(clusterLines src/clusterLines.cpp include/utility.h)


# 编译成可执行文件，如果编译动态库可以用：
target_link_libraries(runwaydetection ${OpenCV_LIBS})
target_link_libraries(fastlineDetector ${OpenCV_LIBS})
target_link_libraries(houghlineDetector ${OpenCV_LIBS})
target_link_libraries(clusterLines ${OpenCV_LIBS})
#add_executable(audio_tool ${SOURCE_FILES})