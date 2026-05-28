cmake_minimum_required(VERSION 3.18)
project(relabel LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#CUDA
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# ============ 全局编译宏 ============
add_compile_definitions(
  VTIME=0
  VBSIZE=256
  VBWIDTH=65536
)

# ============ 编译选项函数 ============
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89 --extended-lambda -Wno-deprecated-declarations -Xcompiler -fPIC -Xcompiler -Wno-deprecated-copy -Xcompiler -Wno-sign-compare")

function(set_target_build_flags target)
  if(CMAKE_BUILD_TYPE STREQUAL "release") 
    target_compile_options(${target} PRIVATE -O3)
  elseif(CMAKE_BUILD_TYPE STREQUAL "debug")
    target_compile_options(${target} PRIVATE -g -G)
  endif()                                                                                                                                                                                                                 
endfunction()

# ============ 依赖项 ============
# parsb
add_subdirectory(parsb)

# RCM
add_subdirectory(rcm)

# rabbit_order
add_subdirectory(rabbitorder)

# class
add_library(class ${CMAKE_CURRENT_SOURCE_DIR}/src/class.cu)
set_target_properties(class PROPERTIES 
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 89
)

target_include_directories(class PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${CMAKE_CURRENT_SOURCE_DIR}/parsb/include 
  ${CMAKE_CURRENT_SOURCE_DIR}/rcm/include 
  ${CMAKE_CURRENT_SOURCE_DIR}/rabbitorder/include 
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(class PUBLIC 
  parsb
  rcm
  rabbitorder
  OpenMP::OpenMP_CXX
)
set_target_build_flags(class)

# ============ 主文件 ============
# main
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main PUBLIC 
                      class 
                      parsb
                      rcm 
                      rabbitorder 
                      cusparse 
                      cublas)
set_target_build_flags(main)
