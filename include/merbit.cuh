#ifndef __TILE__
#define __TILE__
#include "utils.cuh"
template<unsigned int SIGMA> 
using DescTypeT = typename std::conditional<(SIGMA < 15), unsigned int, unsigned long long>::type;

template<unsigned int BLOCK, unsigned int SIGMA> 
struct TILE
{
    using DescType = DescTypeT<SIGMA>;
    unsigned int*            tile_x;
    unsigned int*            tile_y;
    DescType*                lane_desc;
    int                      tile_num;
    int                      lane_num;
    TILE(){};
    TILE(const int* __restrict__ offset, const int nrow, const int nnz); //全图
    void show();
    void desc();  //统计flag=0-32的tile的数量
    ~TILE();
};

//merge search根据diag寻找coord_x和coord_y
template<unsigned int SIGMA> __device__
void merge_search(const int* offset, 
                  const int  nrow, 
                  const int  nnz, 
                  const int  diag, 
                  int&       coord_x, 
                  int&       coord_y)
{
    //search range
    unsigned int y_min = max(diag - nnz, 0);  //y_min和y_max的设定可以缩小搜索空间
    unsigned int y_max = min(diag, nrow);
    
    //2D binary search
    while(y_min < y_max)
    {
        int mid = (y_min + y_max) >> 1;

        if(offset[mid + 1] <= diag - mid - 1)
        {
            y_min = mid + 1;
        }
        else
        {
            y_max = mid;
        }
    }
    
    coord_x = diag - y_min;
    coord_y = min(y_min, nrow);
}

template<unsigned int BLOCK, unsigned int SIGMA> __global__ 
void Tile_Kernel(const int* __restrict__        offset, 
                 const int                      nrow, 
                 const int                      nnz, 
                 const int                      tile_num, 
                 const int                      lane_num, 
                 unsigned int* __restrict__     tile_x, 
                 unsigned int* __restrict__     tile_y, 
                 DescTypeT<SIGMA>* __restrict__ lane_desc)
{
    using DescType = DescTypeT<SIGMA>;
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int lid = tid % 32;
    const int tile_id = gid / 32;
    
    if(gid < lane_num)
    {
        //基础参数
        const int warp_len = ((tile_id + 1) * 32 < lane_num)? 32 : (lane_num - tile_id * 32); //不够一个完整的warp
        const int lane_nnz = ((gid + 1) * SIGMA < nnz + nrow)? SIGMA : (nnz + nrow - gid * SIGMA);
        const unsigned int mask = (warp_len == 32)? 0xFFFFFFFF : (0X1 << warp_len) - 1;
        DescType one = (sizeof(DescType) == 4)? 0x1 : 1ull;
        const int delta1 = (sizeof(DescType) == 4)? 18 : 22;
        const int delta2 = (sizeof(DescType) == 4)? 9  : 11;

        //计算每个lane的coord_x及coord_y
        int diag = gid * SIGMA;
        int coord_x, coord_y;
        merge_search<SIGMA>(offset, nrow, nnz, diag, coord_x, coord_y);
        __syncwarp();

        unsigned int coord_y_start = coord_y;
        unsigned int warp_coord_x_start = __shfl_sync(mask, coord_x, 0);
        unsigned int warp_coord_y_start = __shfl_sync(mask, coord_y, 0);

        //生成lane_desc: 
        //bit_flag(18-31) | delta_y(9-17) | delta_x(0-8)
        //bit_flag(22-63) | delta_y(11-21) | delta_x(0-10)
        DescType delta_x = coord_x - warp_coord_x_start; //delta_x
        DescType delta_y = coord_y - warp_coord_y_start; //delta_y
        DescType bit_flag = (sizeof(DescType) == 4)? 0x0 : 0ull; //bit_flag
        
        for(int idx=0; idx<lane_nnz; ++idx)
        {
            if(coord_x < offset[coord_y + 1])
            {
                ++coord_x;
            }
            else
            {
                bit_flag |= (one << idx);
                ++coord_y;
            }
        }
        DescType tmp_desc = (bit_flag << delta1) | (delta_y << delta2) | delta_x;
        lane_desc[gid] = tmp_desc;
        
        //生成tile_flag
        bool flag = (coord_y_start == coord_y)? true : false;
        if(__all_sync(mask, flag))
        {
            warp_coord_y_start = 0x80000000 | warp_coord_y_start;
        }

        //生成tile_x及tile_y
        if(!lid)
        {
            tile_x[tile_id] = warp_coord_x_start;
            tile_y[tile_id] = warp_coord_y_start;
        }
    }
    else if(gid == lane_num)
    {
        tile_x[tile_num] = nnz;
        tile_y[tile_num] = nrow;
    }
};

template<unsigned int BLOCK, unsigned int SIGMA>
TILE<BLOCK, SIGMA>::TILE(const int* __restrict__ offset, 
                         const int               nrow, 
                         const int               nnz)
{
    using DescType = DescTypeT<SIGMA>;

    //变量定义并申请空间
    lane_num = (nrow + nnz + SIGMA - 1) / SIGMA;
    tile_num = (lane_num + 31) / 32;
    
    cudaMalloc((void**)&tile_x, sizeof(unsigned int) * (tile_num + 1));
    cudaMalloc((void**)&tile_y, sizeof(unsigned int) * (tile_num + 1));
    cudaMalloc((void**)&lane_desc, sizeof(DescType) * lane_num);
    cudaMemset(tile_x, 0, sizeof(unsigned int) * (tile_num + 1));
    cudaMemset(tile_y, 0, sizeof(unsigned int) * (tile_num + 1));
    cudaMemset(lane_desc, 0, sizeof(DescType) * lane_num);

    int grid = (lane_num + BLOCK) / BLOCK;
    Tile_Kernel<BLOCK, SIGMA><<<grid, BLOCK>>>(offset, nrow, nnz, tile_num, lane_num, tile_x, tile_y, lane_desc);
    cudaDeviceSynchronize();
};

template<unsigned int BLOCK, unsigned int SIGMA>
void TILE<BLOCK, SIGMA>::show()
{
    using DescType = DescTypeT<SIGMA>;
    
    std::cout <<", tile_num = " << tile_num << ", lane_num = " << lane_num << "\n";

    std::cout << "tile_x: " << "\n";
    CUDASHOW(tile_x, unsigned int, tile_num + 1);

    std::cout << "tile_y: " << "\n";
    unsigned int h_tile_y[tile_num + 1];
    cudaMemcpy(h_tile_y, tile_y, sizeof(unsigned int) * (tile_num + 1), cudaMemcpyDeviceToHost);
    for(int i=0; i<tile_num+1; ++i)
    {
        int val = (h_tile_y[i] << 1) >> 1;
        std::cout << val << " ";
    }
    //CUDASHOW(tile_y, unsigned int, tile_num + 1);
    std::cout << "\n";

    std::cout << "lane_desc: " << "\n";
    const int delta1 = (sizeof(DescType) == 4)? 23 : 53;
    const int delta2 = (sizeof(DescType) == 4)? 14 : 42;
    const int delta3 = (sizeof(DescType) == 4)? 18 : 22;
    DescType h_lane_desc[lane_num];
    cudaMemcpy(h_lane_desc, lane_desc, sizeof(DescType) * lane_num, cudaMemcpyDeviceToHost);
    for(int i=0;i<lane_num;++i)
    {
        DescType val = h_lane_desc[i];
        DescType tmp = (sizeof(DescType) == 4)? 0x0 : 0ull;

        //delta_x
        tmp = (val << delta1) >> delta1;
        std::cout << tmp << "    ";

        //delta_y
        tmp = (val << delta2) >> delta1;
        std::cout << tmp << "    ";

        //bit_flag
        for(int j=0; j<SIGMA; ++j)
        {
            tmp = 0x1 & (val >> (delta3 + j));
            std::cout << tmp << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << std::endl;
};

template<unsigned int BLOCK, unsigned int SIGMA> __global__ 
void desc_kernel(const unsigned int* __restrict__ tile_x, 
                 const unsigned int* __restrict__ tile_y,
                 const int                        tile_num, 
                 int*                             tile_fast_num,
                 int*                             tile_empty_num)
{
    const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(tile_id < tile_num)
    {
        //计算快速处理快的数量
        bool tile_fast = tile_y[tile_id] >> 31;
        if(tile_fast) 
        {
            atomicAdd(tile_fast_num, 1);
        }
        
        //计算空块的数量
        if(tile_x[tile_id] == tile_x[tile_id + 1])
        {
            atomicAdd(tile_empty_num, 1);
        }
    }
}

template<unsigned int BLOCK, unsigned int SIGMA>
void TILE<BLOCK, SIGMA>::desc()
{
    int* d_tile_fast_num;
    int* d_tile_empty_num;
    cudaMalloc((void**)&d_tile_fast_num, sizeof(int));
    cudaMalloc((void**)&d_tile_empty_num, sizeof(int));
    cudaMemset(d_tile_fast_num, 0, sizeof(int));
    cudaMemset(d_tile_empty_num, 0, sizeof(int));

    const int grid = (tile_num + BLOCK - 1) / BLOCK;
    desc_kernel<BLOCK, SIGMA><<<grid, BLOCK>>>(tile_x, tile_y, tile_num, d_tile_fast_num, d_tile_empty_num);

    int tile_fast_num, tile_empty_num;
    cudaMemcpy(&tile_fast_num, d_tile_fast_num, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tile_empty_num, d_tile_empty_num, sizeof(int), cudaMemcpyDeviceToHost);
    CUDAFREE(d_tile_fast_num);
    CUDAFREE(d_tile_empty_num);

    std::cout << "&" << tile_num << "&" << tile_fast_num << "&" << tile_empty_num << "\n";
}

template<unsigned int BLOCK, unsigned int SIGMA>
TILE<BLOCK, SIGMA>::~TILE()
{
    CUDAFREE(tile_x);
    CUDAFREE(tile_y);
    CUDAFREE(lane_desc);
}

//fast_track
template<typename ValueType> __device__ 
void fast_track(const int* __restrict__       row_ind, 
                const ValueType* __restrict__ values, 
                const int                     warp_x_nnz, 
                const int                     lid, 
                const ValueType* __restrict__ d_x, 
                ValueType* __restrict__       s_y)
{
    ValueType val = 0;
    ValueType sum = 0;
    
    for(int idx = lid; idx < warp_x_nnz; idx += 32)
    {
        val += values[idx] * d_x[row_ind[idx]];
    }

    sum = warp_reduce<ValueType>(val);
    __syncwarp();
    
    if(!lid)
    {
        atomicAdd(&s_y[0], sum);
    }
}

//一个warp处理一段，用__syncwarps同步
template<typename ValueType> __device__
void load_smem(const int* __restrict__       row_ind, 
               const ValueType* __restrict__ values, 
               const int                     warp_x_nnz, 
               const int                     lid, 
               const ValueType* __restrict__ d_x, 
               ValueType* __restrict__       s_data)
{
    for(int idx=lid; idx<warp_x_nnz; idx+=32)
    {
        s_data[idx] = values[idx] * __ldg(&d_x[row_ind[idx]]);
    }
}

//warp_segmented_sum
template<typename ValueType> __forceinline__ __device__
void warp_segmented_sum(const int       lid, 
                        const bool      flag, 
                        const ValueType sum, 
                        const int       idx_y, 
                        const int       len, 
                        ValueType*      d_y)
{
    ValueType local_sum = sum;
    ValueType tmp_sum = 0;
    bool local_flag = (lid < len)? flag : true;
    bool tmp_flag = local_flag;
    unsigned int i = 0x1;
    const unsigned int mask = (len == 32)? 0xFFFFFFFF : (0X1 << len) - 1; 
    
    while(__any_sync(mask, !local_flag))
    {
        tmp_sum = __shfl_up_sync(mask, local_sum, i);
        tmp_flag = __shfl_up_sync(mask, local_flag, i);
        local_sum += (!local_flag && lid >= i)? tmp_sum : (ValueType)(0);
        local_flag = local_flag? local_flag : tmp_flag;
        i <<= 1;
    }
    
    int loc = (lid == 0)? len - 1 : lid - 1;
    int idx = __shfl_sync(mask, idx_y, loc);
    tmp_sum = __shfl_sync(mask, local_sum, loc);
    if(flag)
    {
        atomicAdd(&d_y[idx], tmp_sum);
    }
}

//normal_track
template<typename ValueType, unsigned int SIGMA> __device__
void normal_track(const ValueType* __restrict__ s_data, 
                  const DescTypeT<SIGMA>        lane_desc, 
                  const int                     warp_len, 
                  const int                     lane_nnz, 
                  const int                     lid, 
                  ValueType* __restrict__       s_y)
{
    using DescType = DescTypeT<SIGMA>;
    const int delta1 = (sizeof(DescType) == 4)? 23 : 53;
    const int delta2 = (sizeof(DescType) == 4)? 14 : 42;
    const int delta3 = (sizeof(DescType) == 4)? 18 : 22;
    int ind_x = (lane_desc << delta1) >> delta1;
    int ind_y = (lane_desc << delta2) >> delta1;
    DescType bit_flag = lane_desc >> delta3;
    bool first_flag = true;
    bool lane_flag = (!lid)? true : false;
    ValueType sum = 0.0;
    
    for(int i=0; i<lane_nnz; ++i)
    {
        bool flag = 0x1 & (bit_flag >> i);
        sum += (1 - flag) * s_data[ind_x];
        
        if(flag)
        {
            if(first_flag)
            {
                atomicAdd(&s_y[ind_y], sum);
                first_flag = false;
            }
            else
            {
                s_y[ind_y] = sum;
            }
            sum = 0.0;
        }
        
        ind_x += 1 - flag;
        ind_y += flag;
        lane_flag |= flag;
    }

    warp_segmented_sum<ValueType>(lid, lane_flag, sum, ind_y, warp_len, s_y);
}

template<typename ValueType> __device__
void load_mem(const ValueType* __restrict__ s_y, 
              const int                     block_y_nnz, 
              const int                     tid, 
              const int                     block_len, 
              ValueType* __restrict__       d_y,
              ValueType* __restrict__       d_z)
{
    for(int idx=tid; idx<block_y_nnz; idx+=block_len)
    {
        if(idx == 0 || idx == block_y_nnz - 1)
        {
            atomicAdd(&d_y[idx], s_y[idx]);
            d_z[idx] = 0;
        }
        else
        {
            d_y[idx] = s_y[idx];
        }
    }
}

template<typename ValueType, unsigned int BLOCK, unsigned int SIGMA> __global__
void merbit_kernel(const int* __restrict__                offset, 
                   const int* __restrict__                row_ind, 
                   const ValueType* __restrict__          values, 
                   const int                              nrow, 
                   const int                              nnz, 
                   const unsigned int* __restrict__       tile_x, 
                   const unsigned int* __restrict__       tile_y, 
                   const DescTypeT<SIGMA>* __restrict__   lane_desc, 
                   const int                              tile_num, 
                   const int                              lane_num, 
                   const ValueType* __restrict__          d_x, 
                   ValueType* __restrict__                d_y,
                   ValueType* __restrict__                d_z)
{
    using DescType = DescTypeT<SIGMA>;

    //变量赋值
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int lid = gid % 32;
    const int tile_id = gid / 32;
    
    //获取block信息
    int block_tile_start = blockIdx.x * BLOCK / 32;
    int block_tile_end = ((blockIdx.x + 1) * BLOCK / 32 < tile_num + 1)? (blockIdx.x + 1) * BLOCK / 32 : tile_num;
    int block_coord_x_start = tile_x[block_tile_start];
    int block_coord_x_end = tile_x[block_tile_end];
    
    int block_x_nnz = block_coord_x_end - block_coord_x_start;
    int block_coord_y_start = (tile_y[block_tile_start] << 1) >> 1;
    int block_coord_y_end = (tile_y[block_tile_end] << 1) >> 1;
    int block_y_nnz = (block_coord_y_end == nrow)? block_coord_y_end - block_coord_y_start : block_coord_y_end - block_coord_y_start + 1; //最后一条线是虚拟的，不能加1
    
    //申请共享内存
    __shared__ ValueType s_shared[BLOCK * SIGMA + 1];
    for(int idx=tid; idx<=BLOCK * SIGMA; idx+=BLOCK)
    {
        s_shared[idx] = 0;
    }
    __syncthreads();
    ValueType* s_data = s_shared;
    ValueType* s_y = s_shared + block_x_nnz;
    
    if(tile_id<tile_num)
    {
        //获取warp信息
        int warp_coord_x_start = tile_x[tile_id];
        int warp_coord_x_end = tile_x[tile_id + 1];
        
        //只跑有用的
        if(warp_coord_x_start < warp_coord_x_end)
        {
            unsigned int local_tile_y = tile_y[tile_id];
            bool fast = local_tile_y >> 31;
            int warp_coord_y_start = (local_tile_y << 1) >> 1;
            int warp_block_coord_y_start = warp_coord_y_start - block_coord_y_start;
            int warp_x_nnz = warp_coord_x_end - warp_coord_x_start;
            
            if(fast)
            {
                fast_track<ValueType>(&row_ind[warp_coord_x_start], &values[warp_coord_x_start], warp_x_nnz, lid, d_x, &s_y[warp_block_coord_y_start]);
            }
            else
            {
                //加载至共享内存
                int warp_block_coord_x_start = warp_coord_x_start - block_coord_x_start;
                load_smem<ValueType>(&row_ind[warp_coord_x_start], &values[warp_coord_x_start], warp_x_nnz, lid, d_x, &s_data[warp_block_coord_x_start]);
                __syncwarp();

                //开始处理
                if(gid < lane_num)
                {   
                    int warp_len = ((tile_id + 1) * 32 < lane_num)? 32 : (lane_num - tile_id * 32);
                    int lane_nnz = ((gid + 1) * SIGMA < nnz + nrow)? SIGMA : (nnz + nrow - gid * SIGMA);
                    DescType local_lane_desc = lane_desc[gid];
                    normal_track<ValueType, SIGMA>(&s_data[warp_block_coord_x_start], local_lane_desc, warp_len, lane_nnz, lid, &s_y[warp_block_coord_y_start]);
                }
            }
        }
    }
    __syncthreads();

    //将数据从共享内存加载到主存
    load_mem<ValueType>(s_y, block_y_nnz, tid, BLOCK, &d_y[block_coord_y_start], &d_z[block_coord_y_start]);
}
#endif