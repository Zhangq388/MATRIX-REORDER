cmake_minimum_required(VERSION 3.12)
project(rabbitorder CXX)

#OMP检测
find_package(OpenMP)
if (NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found! Please install OpenMP (libgomp/libomp) or use a compiler with OpenMP support.")
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 自定义编译警告标志
set(WARNING_FLAGS 
    -Wall 
    -Wextra 
    -Wcast-align 
    -Wcast-qual 
    -Wconversion 
    -Wfloat-equal 
    -Wformat=2 
    -Winit-self 
    -Wmissing-declarations 
    -Wmissing-include-dirs 
    -Wpointer-arith 
    -Wredundant-decls 
    -Wswitch-default 
    -Wuninitialized 
    -Wwrite-strings 
    -Wno-sign-conversion 
    -Wno-unused-function 
    -Wno-missing-declarations
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
add_library(rabbitorder STATIC ${PROJECT_SOURCE_DIR}/src/rabbit_order.cpp)

#头文件
target_include_directories(rabbitorder PRIVATE ${PROJECT_SOURCE_DIR}/include)

#OMP自动检测和链接
target_link_libraries(rabbitorder PUBLIC OpenMP::OpenMP_CXX numa)
set_target_build_flags(rabbitorder)

# 添加可执行文件
add_executable(rabbitorder_exec ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_include_directories(rabbitorder_exec PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(rabbitorder_exec rabbitorder)
set_target_build_flags(rabbitorder_exec)

