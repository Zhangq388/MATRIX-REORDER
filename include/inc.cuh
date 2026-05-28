#ifndef __INC__
#define __INC__
#include <iostream>
#include <fstream>
#include <sstream>

#include <cfloat>
#include <math.h>
#include <bitset>
#include <iomanip>
#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <type_traits>


#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/adjacent_difference.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include "/usr/local/cuda/include/cub/cub.cuh"
#include "/usr/local/cuda/include/cub/device/device_histogram.cuh"
#include "/usr/local/cuda/include/cub/device/device_histogram.cuh"
#include "/usr/local/cuda/include/cub/device/device_partition.cuh"
#include "/usr/local/cuda/include/cub/device/device_reduce.cuh"
#include "/usr/local/cuda/include/cub/device/device_segmented_reduce.cuh"
#include "/usr/local/cuda/include/cub/device/device_segmented_radix_sort.cuh"
#include "/usr/local/cuda/include/cub/device/device_spmv.cuh"
#include "/usr/local/cuda/include/cub/device/device_select.cuh"

#include "/usr/local/cuda/include/cub/block/block_adjacent_difference.cuh"
#include "/usr/local/cuda/include/cub/block/block_discontinuity.cuh"
#include "/usr/local/cuda/include/cub/block/block_exchange.cuh"
#include "/usr/local/cuda/include/cub/block/block_radix_rank.cuh"
#include "/usr/local/cuda/include/cub/block/block_radix_sort.cuh"
#include "/usr/local/cuda/include/cub/block/block_scan.cuh"
#include "/usr/local/cuda/include/cub/block/block_shuffle.cuh"
#include "/usr/local/cuda/include/cub/block/block_reduce.cuh"

#include "/usr/local/cuda/include/cub/warp/warp_reduce.cuh"
#include "/usr/local/cuda/include/cub/warp/warp_scan.cuh"

extern std::fstream    FSTR;

#define FREE(ptr)        \
if(ptr != nullptr)       \
{                        \
   free(ptr);            \
   ptr = nullptr;        \
}

#define CUDAFREE(ptr)    \
if(ptr != nullptr)       \
{                        \
   cudaFree(ptr);        \
   ptr = nullptr;        \
}

#define SHOW(ptr, start, size)              \
{                                           \
   for(int loc=0; loc<size; ++loc)          \
   {                                        \
      std::cout << " " << ptr[start + loc]; \
   }                                        \
   std::cout << std::endl;                  \
}

#define CUDASHOW(ptr, type, size)  \
{                              \
   thrust::host_vector<type>  VEC(size, 0.0); \
   thrust::copy(thrust::device_pointer_cast(ptr), thrust::device_pointer_cast(ptr) + size, VEC.begin()); \
   for(auto& vec : VEC)        \
   {                           \
      std::cout << " " << vec; \
   }                           \
   std::cout << std::endl;     \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess)                                                 \
    {                                                                          \
        printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS)                                     \
    {                                                                          \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status); \
    }                                                                          \
}

#define TRYCATCH(func)                                                         \
try                                                                            \
{                                                                              \
   func;                                                                       \
}                                                                              \
catch(const std::exception& e)                                                 \
{                                                                              \
   std::cerr << "Exception caught: " << e.what() << std::endl;                 \
}                                                                              \
catch(...)                                                                     \
{                                                                              \
   std::cerr << "Unknown exception caught!" << std::endl;                      \
}
#endif






