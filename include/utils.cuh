#ifndef __UTILS__
#define __UTILS__
#include "inc.cuh"
//check permutation
template<bool isDevice>
void check_perm(const int* perm, const int nrow)
{
    if (perm == nullptr || nrow <= 0)
    {
        std::cout << "[check_perm] invalid input: perm=" << perm << ", nrow=" << nrow << std::endl;
        return;
    }

    int* d_perm   = nullptr;
    int* d_result = nullptr;
    cudaMalloc((void**)&d_perm,   sizeof(int) * nrow);
    cudaMalloc((void**)&d_result, sizeof(int) * nrow);

    if constexpr (isDevice)
    {
        cudaMemcpy(d_perm, perm, sizeof(int) * nrow, cudaMemcpyDeviceToDevice);
    }
    else
    {
        cudaMemcpy(d_perm, perm, sizeof(int) * nrow, cudaMemcpyHostToDevice);
    }

    thrust::device_ptr<int> ptr_perm(d_perm);
    thrust::device_ptr<int> ptr_result(d_result);

    // 排序后检查
    thrust::sort(ptr_perm, ptr_perm + nrow);
    thrust::adjacent_difference(ptr_perm, ptr_perm + nrow, ptr_result);

    // 相邻差为0 -> 有重复（从 index 1 开始）
    auto it_dup = thrust::find(ptr_result + 1, ptr_result + nrow, 0);
    bool has_dup = (it_dup != ptr_result + nrow);

    // 检查范围 [0, nrow-1]
    auto mm = thrust::minmax_element(ptr_perm, ptr_perm + nrow);
    int h_min = *mm.first;
    int h_max = *mm.second;
    bool range_ok = (h_min == 0 && h_max == nrow - 1);

    if (has_dup || !range_ok)
    {
        std::cout << "[check_perm] INVALID";
        if (has_dup)   std::cout << " | duplicate at sorted idx " << (it_dup - ptr_result);
        if (!range_ok) std::cout << " | range=[" << h_min << "," << h_max << "], expected [0," << (nrow - 1) << "]";
        std::cout << std::endl;
    }

    CUDAFREE(d_perm);
    CUDAFREE(d_result);
}

template<typename ValueType> __forceinline__ __device__
ValueType warp_reduce(ValueType val)
{
    const unsigned int mask = 0xFFFFFFFF; 
    #pragma unroll
    for(unsigned int delta = 16; delta > 0; delta >>= 1)
    {
        val += __shfl_xor_sync(mask, val, delta);
    }
    return val;
}

template<typename ValueType> __forceinline__ __device__
void warp_scan(const int lid, const int len, ValueType& sum)
{
    ValueType val = 0;
    const unsigned int mask = (len == 32)? 0xFFFFFFFF : (0X1 << len) - 1; 
    #pragma unroll
    for(unsigned int i=0x1;i<len;i<<=1)
    {  
        val = __shfl_up_sync(mask, sum, i); //先取再加，稳当些
        sum += (lid >= i)? val : 0;
    }
}

template<typename ValueType> __global__ 
void warmup_kernel(ValueType *d_scan)
{
    const int lid = threadIdx.x & 31;

    int sum = 1;
    warp_scan<ValueType>(lid, 32, sum);

    if(lid == 31)
    {
        d_scan[lid] = sum;
    }
}

template<typename ValueType> 
void format_warmup()
{
    ValueType *d_scan;
    cudaMalloc((void **)&d_scan, 32 * sizeof(ValueType));

    int block = 128;
    int grid  = 4000;

    for (int i = 0; i < 50; i++)
    {
        warmup_kernel<<< grid, block >>>(d_scan);
    }

    cudaFree(d_scan);
}

template<typename ValueType>
void Fun_Check(const ValueType* X, const ValueType* Y, const int nrow)
{
    ValueType err(0.0);

    for(int i=0;i<nrow;++i)
    {
        err += (std::abs(X[i]-Y[i]) > std::abs(X[i]) * 0.0001)? std::abs(X[i]-Y[i]) : 0.0;
        if(std::abs(X[i]-Y[i]) > std::abs(X[i]) * 0.0001)
        {
            std::cout << "X[" << i << "] = " << X[i] << ", Y[" << i << "] = " << Y[i] << "\n";
        }
    }

    std::cout << ", total error = " << err << std::endl;
    //std::cout << "&" << err << std::endl;
}
#endif