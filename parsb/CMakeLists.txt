# Parallel SlashBurn CMake
cmake_minimum_required(VERSION 3.16)
project(par_slashburn LANGUAGES CXX)
set(CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenMP REQUIRED)

# ============ 第三方依赖 ============
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(fmt)

# abseil
FetchContent_Declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20250127.1
)
FetchContent_MakeAvailable(abseil)

# spray
FetchContent_Declare(
  spray
  GIT_REPOSITORY https://github.com/atrostan/spray.git
  GIT_TAG master)
FetchContent_MakeAvailable(spray)
add_library(spray INTERFACE)
target_include_directories(spray INTERFACE ${spray_SOURCE_DIR}/include)

# ips4o
set(IPS4O_USE_OPENMP "true")
set(ONEDPL_PAR_BACKEND "openmp")
set(IPS4O_DISABLE_PARALLEL OFF)
FetchContent_Declare(
  ips4o
  GIT_REPOSITORY https://github.com/SaschaWitt/ips4o.git
  GIT_TAG master
)
FetchContent_MakeAvailable(ips4o)
add_library(ips4o INTERFACE)
target_include_directories(ips4o INTERFACE ${ips4o_SOURCE_DIR}/ips4o)

# ============ 编译选项函数 ============
function(set_target_build_flags target)
  if(CMAKE_BUILD_TYPE STREQUAL "release")
    target_compile_options(${target} PRIVATE -O3 -Wno-deprecated-copy -Wno-sign-compare)
  elseif(CMAKE_BUILD_TYPE STREQUAL "debug")
    target_compile_options(${target} PRIVATE -g -Wno-deprecated-copy -Wno-sign-compare)
  endif()
endfunction()

# ============ 生成库文件 ============
add_library(parsb STATIC src/SlashBurn.cpp)

# include 目录
target_include_directories(parsb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接依赖
target_link_libraries(parsb
  fmt::fmt
  absl::flat_hash_set
  absl::flat_hash_map
  spray
  ips4o
  OpenMP::OpenMP_CXX
  atomic
)

set_target_build_flags(parsb)

# 编译宏定义
# 设置默认值，仅在未定义时才设置
if(NOT DEFINED VTIME)
  set(VTIME 0)
endif()

if(NOT DEFINED VBSIZE)
  set(VBSIZE 512)
endif()

if(NOT DEFINED VBWIDTH)
  set(VBWIDTH 32768)
endif()

target_compile_definitions(parsb PUBLIC
  _REENTRANT=1
  BSIZE=${VBSIZE}
  BWIDTH=${VBWIDTH}
  TIME=${VTIME}
)



# ============ 生成可执行文件 ============
add_executable(parsb_exec src/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(parsb_exec parsb)
set_target_build_flags(parsb_exec)