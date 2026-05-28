cmake_minimum_required(VERSION 3.10)
project(RCM_PROJECT CXX)

#OMP检测
find_package(OpenMP)
if (NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found! Please install OpenMP (libgomp/libomp) or use a compiler with OpenMP support.")
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

#指定源文件和目标文件夹
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(OBJ_DIR ${PROJECT_SOURCE_DIR}/obj)

#设置源文件列表
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/rcm.cpp
    ${SRC_DIR}/queue.cpp
    ${SRC_DIR}/mmio.cpp
)

# ============ 编译选项函数 ============
function(set_target_build_flags target)
  if(CMAKE_BUILD_TYPE STREQUAL "release")
    target_compile_options(${target} PRIVATE -O3 -Wno-deprecated-copy -Wno-sign-compare)
  elseif(CMAKE_BUILD_TYPE STREQUAL "debug")
    target_compile_options(${target} PRIVATE -g -Wno-deprecated-copy -Wno-sign-compare)
  endif()
endfunction()

#生成库文件
add_library(rcm STATIC ${SOURCES})

#头文件
target_include_directories(rcm PRIVATE ${PROJECT_SOURCE_DIR}/include)

#OMP自动检测和链接
target_link_libraries(rcm PUBLIC OpenMP::OpenMP_CXX)
set_target_build_flags(rcm)

#生成可执行文件
add_executable(rcm_exec ${SOURCES})
target_include_directories(rcm_exec PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(rcm_exec rcm)
set_target_build_flags(rcm_exec)

