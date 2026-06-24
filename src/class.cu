#include "../include/class.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//基图
template<typename ValueType>
BASEGRAPH<ValueType>::BASEGRAPH(const ENV& _env, const int _gid)
{
    //STEP 1：变量赋值
    std::fstream  file;
    std::string   line = "";
    int           id  = 0;
    int           i   = 0;
    bool          pattern = true;
    bool          zero_base = true;
    std::string   filename = "";
    
    file.open(_env.config, std::fstream::ios_base::in);
    do
    {
        std::getline(file, line);
        std::stringstream(line) >> id >> filename;
    } while (id != _gid);
    file.close();
    
    
    //STEP 2：读取数据文件位置
    int Row(0), Col(0), Nnz(0);
    ValueType Val(0.0);

    file.open(filename, std::fstream::ios_base::in);
    do 
    {
        std::getline(file, line);
        if(i == 0)
        {
            pattern = (line.find("symmetric") != std::string::npos)? true : false;
            zero_base = (line.find("basezero") != std::string::npos)? true : false;
        }
        ++i;
    } while (line[0] == '%');
    std::stringstream(line) >> Row >> Col >> Nnz;
 
    graph_id = _gid;
    nrow = Row;
    nnz  = pattern? 2 * Nnz : Nnz;
    
    row_ind = (int*)calloc(nnz, sizeof(int));
    col_ind = (int*)calloc(nnz, sizeof(int));
    values = (ValueType*)calloc(nnz, sizeof(ValueType));

    i=0;
    std::stringstream v_str;
    while (std::getline(file, line)) 
    {
        v_str.clear();
        v_str.str(line);
        v_str >> Row >> Col >> Val;
        
        row_ind[i] = (zero_base)? Row : Row - 1;
        col_ind[i] = (zero_base)? Col : Col - 1;
        values[i]  = 1.0;

        if(pattern)
        {
            row_ind[Nnz + i] = (zero_base)? Col : Col - 1;
            col_ind[Nnz + i] = (zero_base)? Row : Row - 1;
            values[Nnz + i]  = 1.0;
        }

        ++i;
    }
    file.close();
}

template<typename ValueType>
BASEGRAPH<ValueType>::~BASEGRAPH()
{
    FREE(row_ind);
    FREE(col_ind);
    FREE(values);
}

//图
template<typename ValueType> 
GRAPH<ValueType>::GRAPH(const ENV& _env, const int _gid)
{
    //STEP 1：变量赋值
    std::fstream  file;
    std::string   line = "";
    int           id  = 0;
    int           i   = 0;
    bool          pattern = true;
    bool          zero_base = true;
    std::string   filename = "";
    
    file.open(_env.config, std::fstream::ios_base::in);
    do
    {
        std::getline(file, line);
        std::stringstream(line) >> id >> filename;
    } while (id != _gid);
    file.close();
    
    
    //STEP 2：读取数据文件位置
    int Row(0), Col(0), Nnz(0);
    ValueType Val(0.0);

    file.open(filename, std::fstream::ios_base::in);
    do 
    {
        std::getline(file, line);
        if(i == 0)
        {
            pattern = (line.find("symmetric") != std::string::npos)? true : false;
            zero_base = (line.find("basezero") != std::string::npos)? true : false;
        }
        ++i;
    } while (line[0] == '%');
    std::stringstream(line) >> Row >> Col >> Nnz;
 
    graph_id = _gid;
    nrow = Row;
    nnz  = pattern? 2 * Nnz : Nnz;
    h_row_ind = (int*)malloc(sizeof(int) * nnz);
    h_col_ind = (int*)malloc(sizeof(int) * nnz);
    h_values = (ValueType*)malloc(sizeof(ValueType) * nnz);

    i=0;
    std::stringstream v_str;
    while (std::getline(file, line)) 
    {
        v_str.clear();
        v_str.str(line);
        v_str >> Row >> Col >> Val;
        
        h_row_ind[i] = (zero_base)? Row : Row - 1;
        h_col_ind[i] = (zero_base)? Col : Col - 1;
        h_values[i]  = 1.0;

        if(pattern)
        {
            h_row_ind[Nnz + i] = (zero_base)? Col : Col - 1;
            h_col_ind[Nnz + i] = (zero_base)? Row : Row - 1;
            h_values[Nnz + i]  = 1.0;
        }

        ++i;
    }
    file.close();
    
    //读入显存
    cudaMalloc((void**)&row_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&col_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&values, sizeof(ValueType) * nnz);
    cudaMalloc((void**)&perm, sizeof(int) * nrow);
    cudaMalloc((void**)&y, sizeof(ValueType) * nrow);
    cudaMemcpy(row_ind, h_row_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, h_col_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(values,  h_values, sizeof(ValueType) * nnz, cudaMemcpyHostToDevice);
    
    //生成perm
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::sequence(ptr_perm, ptr_perm + nrow);
    
    //STEP 3：计算spmv的真实值
    //生成x与y
    ValueType* x;
    cudaMalloc((void**)&x, sizeof(ValueType) * nrow);
    thrust::device_ptr<ValueType> ptr_x(x);
    thrust::device_ptr<ValueType> ptr_y(y);
    thrust::fill(ptr_x, ptr_x + nrow, 1.0);
    thrust::fill(ptr_y, ptr_y + nrow, 0.0);

    //生成offset
    sort(false);
    
    int* offset;
    CHECK_CUDA(cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1)));
    CHECK_CUSPARSE(cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO));

    //生成稀疏库对象
    int64_t    Rows = nrow;
    int64_t    Nnzs  = nnz;
    ValueType  alpha = 1.0;
    ValueType  beta  = 0.0;
    size_t     bufferSize = 0;
    void*      d_buffer = nullptr;

    //生成矩阵
    cusparseSpMatDescr_t      SpMat;
    CHECK_CUSPARSE(cusparseCreateCsr(&SpMat, Rows, Rows, Nnzs, offset, row_ind, values, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, CudaDataType<ValueType>::value));
    
    //生成向量X
    cusparseDnVecDescr_t      Dn_VecX;
    CHECK_CUSPARSE(cusparseCreateDnVec(&Dn_VecX, Rows, x, CudaDataType<ValueType>::value));
    
    //生成向量Y
    cusparseDnVecDescr_t      Dn_VecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&Dn_VecY, Rows, y, CudaDataType<ValueType>::value));
    
    //buffersize
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(_env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, &bufferSize));
    CHECK_CUDA(cudaMalloc((void**)&d_buffer, bufferSize));

    //计算
    CHECK_CUSPARSE(cusparseSpMV(_env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, d_buffer));
    
    //STEP 3：清理内存
    FREE(h_row_ind);
    FREE(h_col_ind);
    FREE(h_values);
    CUDAFREE(x);
    CUDAFREE(offset);
    CUDAFREE(d_buffer);
}

template<typename ValueType> GRAPH<ValueType>::GRAPH(const int _graph_id, const int _nrow, const int _nnz, const unsigned int* _row_ind, const unsigned int* _col_ind)
{
    graph_id = _graph_id;
    nrow = _nrow;
    nnz = _nnz;
    
    //读入显存
    cudaMalloc((void**)&row_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&col_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&values,  sizeof(ValueType) * nnz);
    cudaMemcpy(row_ind, _row_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, _col_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
}


/*
*受控扰动
*/
void local_swap(const int                _nrow,
                const float              _rho,
                const int                _near,
                const int                _window,
                const unsigned long long _seed,
                int*                     permutation)
{
    //step 1：整体shuffle，取前k个
    std::mt19937_64   rng(_seed);
    int n_swaps = (int)std::llround(_rho * (double)_nrow);
    std::vector<int> i_list(_nrow);
    std::iota(i_list.begin(), i_list.end(), 0);
    std::shuffle(i_list.begin(), i_list.end(), rng);
    i_list.resize(n_swaps);

    //step 2：顺序转换生成permutation
    for (int i : i_list) 
    {
        // 左区间 [i-window_max, i-dist_min]
        int l1 = std::max(0, i - _window);
        int r1 = std::min(_nrow - 1, i - _near);

        // 右区间 [i+dist_min, i+window_max]
        int l2 = std::max(0, i + _near);
        int r2 = std::min(_nrow - 1, i + _window);

        int len1 = (l1 <= r1) ? (r1 - l1 + 1) : 0;
        int len2 = (l2 <= r2) ? (r2 - l2 + 1) : 0;

        if (len1 + len2 == 0) continue;

        int j = -1;

        // 先抽签决定左/右（按区间长度加权）
        bool choose_left = false;
        if (len1 > 0 && len2 > 0)
        {
            std::bernoulli_distribution pick_left(double(len1) / double(len1 + len2));
            choose_left = pick_left(rng);
        }
        else if (len1 > 0)
        {
            choose_left = true;
        }
        else
        {
            choose_left = false;
        }

        if (choose_left)
        {
            std::uniform_int_distribution<int> uid(l1, r1);
            j = uid(rng);
        }
        else
        {
            std::uniform_int_distribution<int> uid(l2, r2);
            j = uid(rng);
        }

        if (j != i) 
        {
            std::swap(permutation[i], permutation[j]);
        }
    }
}

void block_shuffle(const int                _nrow,
                   const float              _rho,
                   const int                _block_size,
                   const unsigned long long _seed,
                   int*                     permutation)
{
    int B = std::max(2, _block_size);
    int n_blocks = (_nrow + B - 1) / B;
    int n_shuffle_blocks = (int)std::llround(_rho * n_blocks);
    std::mt19937_64 rng(_seed);

    // 选哪些块做块内shuffle
    std::vector<int> blocks(n_blocks);
    std::iota(blocks.begin(), blocks.end(), 0);
    std::shuffle(blocks.begin(), blocks.end(), rng);
    blocks.resize(n_shuffle_blocks);

    for (int b : blocks)
    {
        int l = b * B;
        int r = std::min(_nrow, l + B);   // [l, r)
        if (r - l <= 1) continue;
        std::shuffle(permutation + l, permutation + r, rng);
    }
}


template<typename ValueType> GRAPH<ValueType>::GRAPH(const ENV& _env, const BASEGRAPH<ValueType>& _basegraph, const float _rho, const int _near, const int _window)
{
    //step 1：申请空间并复制内容
    graph_id = _basegraph.graph_id;
    nrow = _basegraph.nrow;
    nnz = _basegraph.nnz;
    
    //申请空间
    h_row_ind = (int*)malloc(sizeof(int) * nnz);
    h_col_ind = (int*)malloc(sizeof(int) * nnz);
    h_values  = (ValueType*)malloc(sizeof(ValueType) * nnz);
    cudaMemcpy(h_row_ind, _basegraph.row_ind, sizeof(int) * nnz, cudaMemcpyHostToHost);
    cudaMemcpy(h_col_ind, _basegraph.col_ind, sizeof(int) * nnz, cudaMemcpyHostToHost);
    cudaMemcpy(h_values,  _basegraph.values,  sizeof(ValueType) * nnz, cudaMemcpyHostToHost);
    
    //step 2：生成perm
    int* h_permutation = (int*)malloc(sizeof(int) * nrow);
    std::iota(h_permutation, h_permutation + nrow, 0);
    local_swap(nrow, _rho, _near, _window, 20260224ULL + graph_id, h_permutation);
    //block_shuffle(nrow, _rho, _window, 20260224ULL, h_permutation);
    check_perm<false>(h_permutation, nrow);
    
    //step 3：更新数据集
    for (int idx = 0; idx < nnz; ++idx) 
    {
        h_row_ind[idx] = h_permutation[h_row_ind[idx]];
        h_col_ind[idx] = h_permutation[h_col_ind[idx]];
    }
    
    //step 4：复制到显存
    cudaMalloc((void**)&row_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&col_ind, sizeof(int) * nnz);
    cudaMalloc((void**)&values,  sizeof(ValueType) * nnz);
    cudaMalloc((void**)&perm, sizeof(int) * nrow);
    cudaMalloc((void**)&y, sizeof(ValueType) * nrow);
    cudaMemcpy(row_ind, h_row_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, h_col_ind, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(values,  h_values,  sizeof(ValueType) * nnz, cudaMemcpyHostToDevice);
    
    //生成perm
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::sequence(ptr_perm, ptr_perm + nrow);
    
    //STEP 3：计算spmv的真实值
    //生成x与y
    ValueType* x;
    cudaMalloc((void**)&x, sizeof(ValueType) * nrow);
    thrust::device_ptr<ValueType> ptr_x(x);
    thrust::device_ptr<ValueType> ptr_y(y);
    thrust::fill(ptr_x, ptr_x + nrow, 1.0);
    thrust::fill(ptr_y, ptr_y + nrow, 0.0);

    //生成offset
    sort(false);
    
    int* offset;
    CHECK_CUDA(cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1)));
    CHECK_CUSPARSE(cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO));

    //生成稀疏库对象
    int64_t    Rows = nrow;
    int64_t    Nnzs  = nnz;
    ValueType  alpha = 1.0;
    ValueType  beta  = 0.0;
    size_t     bufferSize = 0;
    void*      d_buffer = nullptr;

    //生成矩阵
    cusparseSpMatDescr_t      SpMat;
    CHECK_CUSPARSE(cusparseCreateCsr(&SpMat, Rows, Rows, Nnzs, offset, row_ind, values, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, CudaDataType<ValueType>::value));
    
    //生成向量X
    cusparseDnVecDescr_t      Dn_VecX;
    CHECK_CUSPARSE(cusparseCreateDnVec(&Dn_VecX, Rows, x, CudaDataType<ValueType>::value));
    
    //生成向量Y
    cusparseDnVecDescr_t      Dn_VecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&Dn_VecY, Rows, y, CudaDataType<ValueType>::value));
    
    //buffersize
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(_env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, &bufferSize));
    CHECK_CUDA(cudaMalloc((void**)&d_buffer, bufferSize));

    //计算
    CHECK_CUSPARSE(cusparseSpMV(_env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, d_buffer));
    
    //STEP 3：清理内存
    FREE(h_permutation);
    CUDAFREE(x);
    CUDAFREE(offset);
    CUDAFREE(d_buffer);
}

template<typename ValueType>
void GRAPH<ValueType>::sort(const bool dir)
{
    using tupleT = thrust::tuple<int, int, ValueType>;
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(row_ind), thrust::device_pointer_cast(col_ind), thrust::device_pointer_cast(values)));
    auto zip_end = zip_begin + nnz;
    
    if(dir)
    {
        thrust::sort(zip_begin, zip_end, 
                    []__device__(const tupleT& _lhs, const tupleT& _rhs) 
                    {
                        if (thrust::get<0>(_lhs) == thrust::get<0>(_rhs)) 
                        {
                            return thrust::get<1>(_lhs) < thrust::get<1>(_rhs); 
                        } 
                        return thrust::get<0>(_lhs) < thrust::get<0>(_rhs);
                    }
                    );
    }
    else
    {
        thrust::sort(zip_begin, zip_end, 
                    []__device__(const tupleT& _lhs, const tupleT& _rhs) 
                    {
                        if (thrust::get<1>(_lhs) == thrust::get<1>(_rhs)) 
                        {
                            return thrust::get<0>(_lhs) < thrust::get<0>(_rhs); 
                        } 
                        return thrust::get<1>(_lhs) < thrust::get<1>(_rhs);
                    }
                    );
    }
    cudaDeviceSynchronize();
}


//dangling
__global__ void dangling_kernel(const int* offset, const int nrow, int* num_dang)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid<nrow)
    {
        int num = (offset[gid + 1] == offset[gid])? 1 : 0;
        atomicAdd(num_dang, num);
    }
}

//desc，统计图的出度及入度分布
__global__ void desc_kernel1(const int* indices, const int nnz, int* deg)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if(n < nnz)
    {
        atomicAdd(&(deg[indices[n]]), 1);
    }
}

__global__ void desc_kernel2(const int* d_hist, const int maxdeg, int* reducennz)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    if(n < maxdeg)
    {
        reducennz[n] = n * d_hist[n];
    }
}

__global__ void desc_kernel3(const int* d_reducerow, const int nrow, const int* d_reducennz, const int nnz, const int maxdeg, float* d_ratio1, float* d_ratio2)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    //反过来
    if(n < maxdeg)
    {
        d_ratio1[n] = d_reducerow[n] / (double)nrow;
        d_ratio2[n] = d_reducennz[n] / (double)nnz;
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::desc(const std::string& file)
{
    //step 1：统计顶点入度出度
    int*     d_deg_in;
    int*     d_deg_out;
    cudaMalloc((void**)&d_deg_in, sizeof(int) * nrow);
    cudaMalloc((void**)&d_deg_out, sizeof(int) * nrow);
    cudaMemset(d_deg_in, 0, sizeof(int) * nrow);
    cudaMemset(d_deg_out, 0, sizeof(int) * nrow);
    thrust::device_ptr<int>  ptr_deg_in(d_deg_in);
    thrust::device_ptr<int>  ptr_deg_out(d_deg_out);

    const int block = 256;
    int grid = (nnz + block - 1) / block;

    desc_kernel1<<<grid, block>>>(col_ind, nnz, d_deg_in);
    int max_deg_in  = (*thrust::max_element(ptr_deg_in, ptr_deg_in + nrow)) + 1;

    desc_kernel1<<<grid, block>>>(row_ind, nnz, d_deg_out);
    int max_deg_out = (*thrust::max_element(ptr_deg_out, ptr_deg_out + nrow)) + 1;

    //step 2：生成入度出度直方图
    int*     d_hist_in;
    int*     d_hist_out;
    cudaMalloc((void**)&d_hist_in, sizeof(int) * max_deg_in);
    cudaMalloc((void**)&d_hist_out, sizeof(int) * max_deg_out);
    thrust::device_ptr<int>    ptr_hist_in(d_hist_in);
    thrust::device_ptr<int>    ptr_hist_out(d_hist_out);

    void*    d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_deg_in, d_hist_in, max_deg_in + 1, 0, max_deg_in, nrow);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_deg_in, d_hist_in, max_deg_in + 1, 0, max_deg_in, nrow);
    CUDAFREE(d_temp_storage);

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_deg_out, d_hist_out, max_deg_out + 1, 0, max_deg_out, nrow);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_deg_out, d_hist_out, max_deg_out + 1, 0, max_deg_out, nrow);
    CUDAFREE(d_temp_storage);

    //step 3：统计行数
    int*     d_reducerow_in;
    int*     d_reducerow_out;
    cudaMalloc((void**)&d_reducerow_in, sizeof(int) * max_deg_in);
    cudaMalloc((void**)&d_reducerow_out, sizeof(int) * max_deg_out);
    cudaMemset(d_reducerow_in, 0, sizeof(int) * max_deg_in);
    cudaMemset(d_reducerow_out, 0, sizeof(int) * max_deg_out);
    thrust::device_ptr<int>    ptr_reducerow_in(d_reducerow_in);
    thrust::device_ptr<int>    ptr_reducerow_out(d_reducerow_out);
    thrust::inclusive_scan(ptr_hist_in, ptr_hist_in + max_deg_in, ptr_reducerow_in);
    thrust::inclusive_scan(ptr_hist_out, ptr_hist_out + max_deg_out, ptr_reducerow_out);

    //step 4：统计非零元
    int*     d_reducennz_in;
    int*     d_reducennz_out;
    cudaMalloc((void**)&d_reducennz_in, sizeof(int) * max_deg_in);
    cudaMalloc((void**)&d_reducennz_out, sizeof(int) * max_deg_out);
    cudaMemset(d_reducennz_in, 0, sizeof(int) * max_deg_in);
    cudaMemset(d_reducennz_out, 0, sizeof(int) * max_deg_out);

    grid = (max_deg_in + block - 1) / block;
    desc_kernel2<<<grid, block>>>(d_hist_in, max_deg_in, d_reducennz_in);

    grid = (max_deg_out + block - 1) / block;
    desc_kernel2<<<grid, block>>>(d_hist_out, max_deg_out, d_reducennz_out);

    thrust::device_ptr<int>    ptr_reducennz_in(d_reducennz_in);
    thrust::device_ptr<int>    ptr_reducennz_out(d_reducennz_out);
    thrust::inclusive_scan(ptr_reducennz_in, ptr_reducennz_in + max_deg_in, ptr_reducennz_in);
    thrust::inclusive_scan(ptr_reducennz_out, ptr_reducennz_out + max_deg_out, ptr_reducennz_out);

    //step 5：计算比例
    int*   h_hist_in   = (int*)calloc(max_deg_in, sizeof(int));
    int*   h_hist_out  = (int*)calloc(max_deg_out, sizeof(int));
    float* h_ratio_row_in = (float*)calloc(max_deg_in, sizeof(float));
    float* h_ratio_row_out = (float*)calloc(max_deg_out, sizeof(float));
    float* h_ratio_nnz_in = (float*)calloc(max_deg_in, sizeof(float));
    float* h_ratio_nnz_out = (float*)calloc(max_deg_out, sizeof(float));
    float* d_ratio_row_in;
    float* d_ratio_row_out;
    float* d_ratio_nnz_in;
    float* d_ratio_nnz_out;
    cudaMalloc((void**)&d_ratio_row_in, sizeof(int) * max_deg_in);
    cudaMalloc((void**)&d_ratio_row_out, sizeof(int) * max_deg_out);
    cudaMalloc((void**)&d_ratio_nnz_in, sizeof(int) * max_deg_in);
    cudaMalloc((void**)&d_ratio_nnz_out, sizeof(int) * max_deg_out);
    
    grid = (max_deg_in + block - 1) / block;
    desc_kernel3<<<grid, block>>>(d_reducerow_in, nrow, d_reducennz_in, nnz, max_deg_in, d_ratio_row_in, d_ratio_nnz_in);
    
    grid = (max_deg_out + block - 1) / block;
    desc_kernel3<<<grid, block>>>(d_reducerow_out, nrow, d_reducennz_out, nnz, max_deg_out, d_ratio_row_out, d_ratio_nnz_out);

    cudaMemcpy(h_hist_in, d_hist_in, sizeof(int) * max_deg_in, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist_out, d_hist_out, sizeof(int) * max_deg_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ratio_row_in, d_ratio_row_in, sizeof(float) * max_deg_in, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ratio_row_out, d_ratio_row_out, sizeof(float) * max_deg_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ratio_nnz_in, d_ratio_nnz_in, sizeof(float) * max_deg_in, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ratio_nnz_out, d_ratio_nnz_out, sizeof(float) * max_deg_out, cudaMemcpyDeviceToHost);

    //step 6：写文件
    std::fstream    fstr;
    fstr.open(file, std::ios_base::app);
    int num = 0;
    for(int i=0; i<2048; ++i)
    {
        if(h_hist_in[i]>0 || h_hist_out[i]>0)
        {
            fstr << graph_id << "&" << num << "&" << h_ratio_row_in[i] << "&" << h_ratio_nnz_in[i] << "&" << h_ratio_row_out[i] << "&" << h_ratio_nnz_out[i] << "\n";
        }
        ++num;
    }
    fstr.close();
    
    //step 7：回收空间
    CUDAFREE(d_deg_in);
    CUDAFREE(d_deg_out);
    CUDAFREE(d_hist_in);
    CUDAFREE(d_hist_out);
    CUDAFREE(d_reducerow_in);
    CUDAFREE(d_reducerow_out);
    CUDAFREE(d_reducennz_in);
    CUDAFREE(d_reducennz_out);
    CUDAFREE(d_ratio_row_in);
    CUDAFREE(d_ratio_row_out);
    CUDAFREE(d_ratio_nnz_in);
    CUDAFREE(d_ratio_nnz_out);
    FREE(h_hist_in);
    FREE(h_hist_out);
    FREE(h_ratio_row_in);
    FREE(h_ratio_row_out);
    FREE(h_ratio_nnz_in);
    FREE(h_ratio_nnz_out);
}

//show
template<typename ValueType> 
void GRAPH<ValueType>::show()
{
    //打印统计信息
    std::cout << "GRAPH = " << graph_id << ", nrow = " << nrow << ", nnz = " << nnz << "\n";
    
    //定义图
    thrust::host_vector<ValueType> vec_Graph(nrow * nrow, 0.0);
    thrust::host_vector<int>       vec_row_ind(nnz, 0);
    thrust::host_vector<int>       vec_col_ind(nnz, 0);
    thrust::host_vector<ValueType> vec_values(nnz, 0.0);
    thrust::copy(thrust::device_pointer_cast(row_ind), thrust::device_pointer_cast(row_ind) + nnz, vec_row_ind.begin());
    thrust::copy(thrust::device_pointer_cast(col_ind), thrust::device_pointer_cast(col_ind) + nnz, vec_col_ind.begin());
    thrust::copy(thrust::device_pointer_cast(values),  thrust::device_pointer_cast(values) + nnz,  vec_values.begin());

    for(int i=0;i<nnz;++i)
    {
        vec_Graph[vec_col_ind[i] * nrow + vec_row_ind[i]] = vec_values[i];
    }
    
    //打印数据
    for(int i=0;i<nrow;++i)
    {
        for(int j=0;j<nrow;++j)
        {
            std::cout << " " << std::left << std::setw(9) << vec_Graph[i * nrow + j];
        }
        std::cout << "\n";
    }
    
    std::cout << "perm: \n";
    CUDASHOW(perm, int, nrow);

    std::cout << "y_real: \n";
    CUDASHOW(y, ValueType, nrow);
    
    std::cout << std::endl;
}

//visual matrix
__global__ void visual_kernel(const int* row_ind,
                              const int* col_ind,
                              const int  nnz,
                              const int  block_size,
                              const int  image_size, 
                              int*       d_image)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < nnz)
    {
        atomicAdd(&d_image[col_ind[gid] / block_size * image_size + row_ind[gid] / block_size], 1);
    }
}

void density2rgba(float density, unsigned char& r, unsigned char& g, unsigned char& b, unsigned char& a)
{
    if (density <= 0.0f) 
    {
        r=g=b=0; a=0; 
    }
    else if (density > 0.0f && density <= 0.1f) 
    {
        // 蓝(0,0,255) -> 绿(0,255,0)
        float t = density / 0.1f;
        r = 0;
        g = (unsigned char)(255 * t);
        b = (unsigned char)(255 * (1.0f-t));
        a = 255;
    }
    else if (density > 0.1f && density <= 0.5f) 
    {
        // 绿(0,255,0) -> 黄(255,255,0)
        float t = (density-0.1f) / 0.1f;
        r = (unsigned char)(255 * t);
        g = 255;
        b = 0;
        a = 255;
    }
    else 
    {
        // 黄(255,255,0) -> 红(255,0,0)
        float t = (density-0.5f) / 0.5f;
        r = 255;
        g = (unsigned char)(255 * (1.0f-t));
        b = 0;
        a = 255;
    }
}

template<typename ValueType> 
int GRAPH<ValueType>::get_width()
{
    float bound = std::round(std::log2(std::sqrt((float)nrow / (nnz / nrow))));

    if (bound < 10) 
    {
        return 1024;
    }
    else if (bound >= 10 && bound < 12) 
    {
        return 16384;
    }
    else
    {
        return 262144;
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::visual(std::string _filename)
{
    //变量定义
    int image_size = 1024; 
    int block_size = (nrow + image_size - 1) / image_size; 
    
    int* d_image;
    cudaMalloc((void**)&d_image, sizeof(int) * image_size * image_size);
    cudaMemset(d_image, 0, image_size * image_size);
    const int block = 256;
    const int grid = (nnz + block - 1) / block;
    visual_kernel<<<grid, block>>>(row_ind, col_ind, nnz, block_size, image_size, d_image);

    int* h_image = (int*)malloc(sizeof(int) * image_size * image_size);
    cudaMemcpy(h_image, d_image, sizeof(int) * image_size * image_size, cudaMemcpyDeviceToHost);
    
    std::vector<unsigned char> image(image_size * image_size * 4);
    for(int idx=0; idx<image_size * image_size; ++idx)
    {
        float density = std::clamp(h_image[idx] / (float)(block_size), 0.0f, 1.0f);
        density2rgba(density, image[4*idx + 0], image[4*idx + 1], image[4*idx + 2], image[4*idx + 3]);
    }
    
    stbi_write_png(_filename.c_str(), image_size, image_size, 4, image.data(), image_size * 4);

    CUDAFREE(d_image);
    FREE(h_image);
}


/*
*compute gscore
*/
__global__ void gscore_kernel(const int* offset1,
                              const int* indice1,
                              const int* offset2,
                              const int* indice2,
                              const int  nrow,
                              const int  width,
                              int*       score)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid < nrow)
    {
        for(int i=max(0, gid - width); i<gid; ++i)
        {
            for(int j=offset2[i]; j<offset2[i+1]; ++j)
            {
                for(int k=offset1[indice2[j]]; k<offset1[indice2[j]+1]; ++k)
                {
                    if(indice1[k] == gid)
                    {
                        atomicAdd(&score[gid], 1);
                    }
                }
            }
        }
    }
}

template<typename ValueType> 
int GRAPH<ValueType>::gscore(const ENV& _env, 
                             const int  _width)
{
    //step 1：生成图数据
    int*           d_offset1;
    int*           d_indice1;
    int*           d_offset2;
    int*           d_indice2;
    CHECK_CUDA(cudaMalloc((void**)&d_offset1, sizeof(int) * (nrow + 1)));
    CHECK_CUDA(cudaMalloc((void**)&d_indice1, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc((void**)&d_offset2, sizeof(int) * (nrow + 1)));
    CHECK_CUDA(cudaMalloc((void**)&d_indice2, sizeof(int) * nnz));

    sort(true);
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, d_offset1, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, d_offset2, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    //step 2：计算每一个顶点的score
    const int block = 256;
    const int grid = (nrow + block - 1) /  block;
    int* d_score;
    cudaMalloc((void**)&d_score, sizeof(int) * nrow);
    cudaMemset(d_score, 0, sizeof(int) * nrow);
    thrust::device_ptr<int>    ptr_score(d_score);
    CUDASHOW(d_score, int, 16);
    gscore_kernel<<<grid, block>>>(d_offset1, d_indice1, d_offset2, d_indice2, nrow, _width, d_score);
    cudaDeviceSynchronize();
    CUDASHOW(d_score, int, 16);
    int score = thrust::reduce(ptr_score, ptr_score + nrow);
    
    //step 3：回收空间
    CUDAFREE(d_score);
    CUDAFREE(d_offset1);
    CUDAFREE(d_indice1);
    CUDAFREE(d_offset2);
    CUDAFREE(d_indice2);

    return score;
}






//大尺度reuse-distance
struct Fenwick 
{
    int              n;
    std::vector<int> bit;
    explicit Fenwick(int n_) : n(n_), bit(n_ + 1, 0) {};
    inline void add(int idx, int delta) 
    {
        for (int i = idx + 1; i <= n; i += i & -i) bit[i] += delta;
    };
    inline long long sum(int idx) const 
    {
        long long s = 0;
        for (int i = idx + 1; i > 0; i -= i & -i) s += bit[i];
        return s;
    };
    inline long long range_sum(int l, int r) const 
    {
        if (r < l) return 0;
        if (l <= 0) return sum(r);
        return sum(r) - sum(l - 1);
    }
};

inline int bin_id(const double rd, const std::vector<int>& bin_edges)
{
    return static_cast<int>(std::upper_bound(bin_edges.begin(), bin_edges.end(), rd) - bin_edges.begin());
}

void compute_sector_rd_histogram(const int*               col_ind,
                                 const int                nrow,
                                 const int                nnz,
                                 const int                sector_size,
                                 const std::vector<int>&  bin_edges,
                                 std::vector<double>&     bin_prob_out)
{
    const int num_sectors = (nrow + sector_size - 1) / sector_size;
    std::vector<int> last_pos(num_sectors, -1);
    Fenwick bit(nnz);

    const int num_bins = static_cast<int>(bin_edges.size()) + 1;
    std::vector<long long> bin_counts(num_bins, 0);
    long long valid_reuse = 0;

    for (int k = 0; k < nnz; ++k)
    {
        const int sector = col_ind[k] / sector_size;
        const int old = last_pos[sector];

        if (old < 0)
        {
            bit.add(k, 1);
            last_pos[sector] = k;
            continue;
        }

        bit.add(old, -1);
        const long long rd = bit.range_sum(old + 1, k - 1);
        ++bin_counts[bin_id(static_cast<double>(rd), bin_edges)];
        ++valid_reuse;

        bit.add(k, 1);
        last_pos[sector] = k;
    }

    bin_prob_out.assign(num_bins, 0.0);
    if (valid_reuse > 0)
    {
        for (int i = 0; i < num_bins; ++i)
        {
            bin_prob_out[i] = static_cast<double>(bin_counts[i]) / static_cast<double>(valid_reuse);
        }
    }
}






/*
* Sampled approximate RD by one CUDA block per sampled position.
* This version uses HyperLogLog (HLL) registers in shared memory, so it can
* estimate large reuse distances such as RD > 589824 without a huge hash table.
*
* status code:
*   0: previous same sector found, rd_est is estimated RD
*   1: cold access, no previous same sector in the scanned prefix
*   2: estimated RD exceeded rd_cap, stopped early
*   3: max_scan limit reached before finding the previous same sector
*/
__device__ __forceinline__ unsigned int hllrd_hash_u32(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int hllrd_rank_from_hash(unsigned int h, const int p)
{
    // Use the low p bits as register id. The remaining 32-p high bits estimate rank.
    unsigned int w = h >> p;
    const int max_rank = 32 - p + 1;
    if (w == 0U) return max_rank;
    int r = __clz(w) - p + 1;
    if (r < 1) r = 1;
    if (r > max_rank) r = max_rank;
    return r;
}

__device__ __forceinline__ void hllrd_update(int* regs, const int hll_m, const int hll_p, const int key)
{
    unsigned int h = hllrd_hash_u32((unsigned int)key);
    int rid = (int)(h & (unsigned int)(hll_m - 1));
    int rank = hllrd_rank_from_hash(h, hll_p);
    atomicMax(&regs[rid], rank);
}

__device__ float hllrd_estimate(const int* regs, const int hll_m)
{
    float sum = 0.0f;
    int zeros = 0;

    for (int i = 0; i < hll_m; ++i)
    {
        int r = regs[i];
        if (r == 0) ++zeros;
        sum += exp2f(-(float)r);
    }

    const float m = (float)hll_m;
    float alpha;
    if (hll_m == 16) alpha = 0.673f;
    else if (hll_m == 32) alpha = 0.697f;
    else if (hll_m == 64) alpha = 0.709f;
    else alpha = 0.7213f / (1.0f + 1.079f / m);

    float estimate = alpha * m * m / sum;

    // Small range correction. For the large-RD threshold this often does not matter,
    // but it improves estimates for small RD values.
    if (estimate <= 2.5f * m && zeros > 0)
    {
        estimate = m * logf(m / (float)zeros);
    }
    return estimate;
}

__global__ void sampled_hll_rd_block_kernel(const int* __restrict__ indice,
                                            const int               nnz,
                                            const int               samples,
                                            const int               sector_size,
                                            const unsigned int      seed,
                                            const int               rd_cap,
                                            const int               max_scan,
                                            const int               hll_p,
                                            const int               hll_m,
                                            float*                  d_rd_est,
                                            int*                    d_status,
                                            int*                    d_pos)
{
    extern __shared__ int regs[];

    const int sid = blockIdx.x;
    if (sid >= samples || nnz <= 1) return;

    for (int i = threadIdx.x; i < hll_m; i += blockDim.x)
    {
        regs[i] = 0;
    }
    __syncthreads();

    __shared__ int status;
    __shared__ int sample_pos;
    __shared__ int scan_lo;
    __shared__ int chunk_found;
    __shared__ float rd_est;
    __shared__ int chunks_done;

    if (threadIdx.x == 0)
    {
        unsigned int r = hllrd_hash_u32(seed ^ (unsigned int)(sid * 0x9e3779b9U + 17U));
        sample_pos = 1 + (int)(r % (unsigned int)(nnz - 1)); // [1, nnz-1]
        scan_lo = (max_scan > 0) ? max(0, sample_pos - max_scan) : 0;
        status = -99;
        rd_est = 0.0f;
        chunks_done = 0;
        d_pos[sid] = sample_pos;
    }
    __syncthreads();

    const int target = indice[sample_pos] / sector_size;
    const int check_interval_chunks = 16; // 16 * 256 = 4096 scanned positions between early-stop checks

    for (int chunk_hi = sample_pos - 1; chunk_hi >= scan_lo && status == -99; chunk_hi -= blockDim.x)
    {
        if (threadIdx.x == 0) chunk_found = -1;
        __syncthreads();

        const int idx = chunk_hi - threadIdx.x;
        const bool valid = (idx >= scan_lo);

        int sec = -1;
        if (valid)
        {
            sec = indice[idx] / sector_size;
            if (sec == target)
            {
                // nearest previous occurrence in this chunk is the largest idx
                atomicMax(&chunk_found, idx);
            }
        }
        __syncthreads();

        const int found_idx = chunk_found;
        const bool found_this_chunk = (found_idx >= 0);

        // Insert only intervening sectors. If the target is found in this chunk,
        // positions <= found_idx are before the previous occurrence and should not count.
        if (valid && sec != target && (!found_this_chunk || idx > found_idx))
        {
            hllrd_update(regs, hll_m, hll_p, sec);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            ++chunks_done;

            if (found_this_chunk)
            {
                rd_est = hllrd_estimate(regs, hll_m);
                status = 0;
            }
            else if (rd_cap > 0 && (chunks_done % check_interval_chunks == 0))
            {
                float tmp_est = hllrd_estimate(regs, hll_m);
                // Slight guard band to reduce premature stopping from HLL over-estimation.
                if (tmp_est > (float)rd_cap * 1.03f)
                {
                    rd_est = tmp_est;
                    status = 2;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if (status == -99)
        {
            rd_est = hllrd_estimate(regs, hll_m);
            if (scan_lo == 0)
            {
                status = 1; // cold access
            }
            else
            {
                status = 3; // max_scan limit reached
            }
        }
        d_rd_est[sid] = rd_est;
        d_status[sid] = status;
    }
}

void sampled_hll_rd_block_gpu(const ENV&               _env,
                              const int                graph_id,
                              const int*               row_ind,
                              const int                nnz,
                              const int                samples,
                              const int                sector_size,
                              const int                rd_cap,
                              const int                hll_p_in,
                              const int                max_scan,
                              const int                stream_type,
                              const std::vector<int>&  bin_edges,
                              std::vector<double>&     bin_prob_out)
{
    const int num_bins = static_cast<int>(bin_edges.size()) + 1;
    bin_prob_out.assign(num_bins, 0.0);

    if (samples <= 0 || nnz <= 1)
    {
        return;
    }
    if (sector_size <= 0)
    {
        throw std::invalid_argument("sector_size must be positive");
    }
    if (bin_edges.empty())
    {
        throw std::invalid_argument("bin_edges must not be empty");
    }
    if (rd_cap > 0 && rd_cap < bin_edges.back())
    {
        throw std::invalid_argument("rd_cap must be at least the last RD bin edge; otherwise status=2 cannot be assigned to the final bin safely");
    }

    int hll_p = hll_p_in;
    if (hll_p <= 0) hll_p = 12;
    hll_p = std::max(4, std::min(hll_p, 14));

    const int hll_m = 1 << hll_p;
    const size_t smem_bytes = sizeof(int) * static_cast<size_t>(hll_m);

    // Use the device recorded by ENV instead of querying the current device.
    const int dev = _env.device_id;
    int optin_smem = 0;
    int default_smem = 0;
    cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&default_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev);

    const int max_smem = (optin_smem > 0) ? optin_smem : default_smem;
    if (static_cast<int>(smem_bytes) > max_smem)
    {
        throw std::runtime_error("sampled_hll_rd_block_gpu: requested dynamic shared memory exceeds the device limit; reduce hll_p");
    }

    if (static_cast<int>(smem_bytes) > default_smem)
    {
        cudaFuncSetAttribute(sampled_hll_rd_block_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    }

    float* d_rd_est = nullptr;
    int* d_status = nullptr;
    int* d_pos = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_rd_est), sizeof(float) * samples);
    cudaMalloc(reinterpret_cast<void**>(&d_status), sizeof(int) * samples);
    cudaMalloc(reinterpret_cast<void**>(&d_pos), sizeof(int) * samples);

    std::vector<float> h_rd_est(samples);
    std::vector<int> h_status(samples);

    constexpr int block_threads = 256;
    const int grid = samples;
    const unsigned int seed = 246813579U + static_cast<unsigned int>(graph_id) * 1009U + static_cast<unsigned int>(stream_type) * 9176U;

    // row_ind is the access stream after sort(false).
    sampled_hll_rd_block_kernel<<<grid, block_threads, smem_bytes, _env.stream>>>(row_ind, nnz, samples, sector_size, seed, rd_cap, max_scan, hll_p, hll_m, d_rd_est, d_status, d_pos);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h_rd_est.data(), d_rd_est, sizeof(float) * static_cast<size_t>(samples), cudaMemcpyDeviceToHost, _env.stream));
    CHECK_CUDA(cudaMemcpyAsync(h_status.data(), d_status, sizeof(int) * static_cast<size_t>(samples), cudaMemcpyDeviceToHost, _env.stream));
    CHECK_CUDA(cudaStreamSynchronize(_env.stream));

    std::vector<long long> bin_counts(num_bins, 0);
    int found = 0;
    int cold = 0;
    int large = 0;
    int unresolved = 0;

    for (int i = 0; i < samples; ++i)
    {
        switch (h_status[i])
        {
            case 0:
            {
                ++found;
                // Do not round the HLL estimate before applying bin boundaries.
                const int id = bin_id(static_cast<double>(h_rd_est[i]), bin_edges);
                ++bin_counts[id];
                break;
            }
            case 1:
                ++cold;
                break;
            case 2:
                ++large;
                ++bin_counts.back();
                break;
            case 3:
                ++unresolved;
                break;
            default:
                ++unresolved;
                break;
        }
    }

    // Match compute_sector_rd_histogram(): normalize only over accesses that
    // have a previous occurrence. status=2 is also a reused access.
    const int valid_reuse = found + large;
    if (valid_reuse > 0)
    {
        for (int i = 0; i < num_bins; ++i)
        {
            bin_prob_out[i] = static_cast<double>(bin_counts[i]) / static_cast<double>(valid_reuse);
        }
    }

    CUDAFREE(d_rd_est);
    CUDAFREE(d_status);
    CUDAFREE(d_pos);
}

template<typename ValueType>
void GRAPH<ValueType>::locality_predict(const ENV& _env, float& speedup)
{
    constexpr int sector_size = 32;
    const std::vector<int> bin_edges = {1, 112, 512, 18432, 36864, 73728, 147456, 294912, 589824};
    std::array<double, rd_merbit_model::kFeatureCount> rd_feature{};
    float  tim1(0.0), tim2(0.0);

    // Recommended comparison settings. max_scan=0 is required to avoid an
    // unresolved-tail bias relative to the exact histogram.
    const int samples = 30960;
    const int rd_cap = bin_edges.back();
    const int hll_p = 12;
    const int max_scan = 0;
    const int stream_type = 0;

    Timer timer;

    // Build the access stream once. It is common to exact and sampled RD, so
    // report it separately rather than charging it only to exact RD.
    timer.Start();
    if (false)
    {
        // Exact RD: D2H copy + CPU histogram.
        std::vector<double> exact_prob(bin_edges.size() + 1, 0.0);
        {
            std::vector<int> h_stream(nnz);
            cudaMemcpy(h_stream.data(), row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
            compute_sector_rd_histogram(h_stream.data(), nrow, nnz, sector_size, bin_edges, exact_prob);
        }

        for (size_t i = 0; i < rd_feature.size(); ++i)
        {
            rd_feature[i] = exact_prob[i];
        }
    }
    

    if (true)
    {
        // Sampled RD: GPU sampling + D2H result copy + host bin aggregation.
        std::vector<double> sampled_prob(bin_edges.size() + 1, 0.0);
        sampled_hll_rd_block_gpu(_env, graph_id, row_ind, nnz, samples, sector_size, rd_cap, hll_p, max_scan, stream_type, bin_edges, sampled_prob);
        
        for (size_t i = 0; i < rd_feature.size(); ++i)
        {
            rd_feature[i] = sampled_prob[i];
        }
    }
    timer.Stop();
    tim1 = timer.Millisecs();

    timer.Start();
    const double predicted_speedup = rd_merbit_model::PredictSpeedup(rd_feature);
    if (!std::isfinite(predicted_speedup) || predicted_speedup <= 0.0)
    {
        throw std::runtime_error("Predictor returned an invalid speedup");
    }
    speedup = static_cast<float>(predicted_speedup);
    timer.Stop();
    tim2 = timer.Millisecs();

    FSTR << "&" << tim1 << "&" << tim2 << "&" << speedup;
}

/*
*spmv测试
*/
template<typename ValueType> 
void GRAPH<ValueType>::spmv(const ENV& _env, std::string _method)
{
    //变量定义
    Timer   timer;
    float   tim1(0), tim2(0);

    //生成offset
    sort(false);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    
    //生成d_x,d_y
    ValueType* d_x;
    ValueType* d_y;
    ValueType* d_z;
    cudaMalloc((void**)&d_x, sizeof(ValueType) * nrow);
    cudaMalloc((void**)&d_y, sizeof(ValueType) * nrow);
    cudaMalloc((void**)&d_z, sizeof(ValueType) * nrow);
    thrust::device_ptr<ValueType>  ptr_x(d_x);
    thrust::device_ptr<ValueType>  ptr_y(d_y);
    thrust::device_ptr<ValueType>  ptr_z(d_z);
    thrust::fill(ptr_x, ptr_x + nrow, 1.0);
    thrust::fill(ptr_y, ptr_y + nrow, 0.0);
    thrust::fill(ptr_z, ptr_z + nrow, 0.0);
    
    //开始运行
    if(_method == "csr")
    {
        //STEP 2：生成稀疏库对象
        int64_t    Rows = nrow;
        int64_t    Nnz  = nnz;
        ValueType  alpha = 1.0;
        ValueType  beta  = 0.0;
        size_t     bufferSize = 0;
        void*      d_buffer = nullptr;

        //生成矩阵
        timer.Start();
        cusparseSpMatDescr_t      SpMat;
        cusparseCreateCsr(&SpMat, Rows, Rows, Nnz, offset, row_ind, values, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexType_t::CUSPARSE_INDEX_32I, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO, CudaDataType<ValueType>::value);

        //生成向量X
        cusparseDnVecDescr_t      Dn_VecX;
        cusparseCreateDnVec(&Dn_VecX, Rows, d_x, CudaDataType<ValueType>::value);
        
        //生成向量Y
        cusparseDnVecDescr_t      Dn_VecY;
        cusparseCreateDnVec(&Dn_VecY, Rows, d_y, CudaDataType<ValueType>::value);
        
        //buffersize
        cusparseSpMV_bufferSize(_env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
        cudaMalloc((void**)&d_buffer, bufferSize);
        cudaDeviceSynchronize();
        timer.Stop();
        tim1 = timer.Millisecs();

        //计算
        timer.Start();
        for(int rnd = 0; rnd<400;++rnd)
        {
            cusparseSpMV(_env.handle_sparse, cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, SpMat, Dn_VecX, &beta, Dn_VecY, CudaDataType<ValueType>::value, cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1, d_buffer);
        }
        cudaDeviceSynchronize();
        timer.Stop();
        tim2 = timer.Millisecs();

        FSTR << "&" << tim1 << "&" << tim2 / 400 << std::endl;

        //回收内存
        CUDAFREE(d_buffer);
        cusparseDestroySpMat(SpMat);
        cusparseDestroyDnVec(Dn_VecX);
        cusparseDestroyDnVec(Dn_VecY);
    }
    else if (_method == "merbit")
    {
        const unsigned int BLOCK = 256;
        const unsigned int SIGMA = 14;

        //step 2：生成tile
        timer.Start();
        TILE<BLOCK, SIGMA> tile(offset, nrow, nnz);
        //tile.desc();
        timer.Stop();
        tim1 = timer.Millisecs();
        
        //step 3：计算
        const int grid = (tile.lane_num + BLOCK - 1) / BLOCK;
        timer.Start();
        for(int rnd=0; rnd<400; ++rnd)
        {
            merbit_kernel<ValueType, BLOCK, SIGMA><<<grid, BLOCK>>>(offset, row_ind, values, nrow, nnz, tile.tile_x, tile.tile_y, tile.lane_desc, tile.tile_num, tile.lane_num, d_x, d_y, d_z);
            std::swap(d_y, d_z);
        }
        cudaMemcpy(d_y, d_z, sizeof(ValueType) * nrow, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        timer.Stop();
        tim2 = timer.Millisecs();
        
        FSTR << "&" << tim1 << "&" << tim2 / 400 << std::endl;
    }
    
    //STEP 5：check 结果
    //根据perm恢复结果
    check_perm<true>(perm, nrow);
    thrust::fill(ptr_z, ptr_z + nrow, 0.0);
    thrust::device_ptr<int>        ptr_perm(perm);
    thrust::gather(ptr_perm, ptr_perm + nrow, ptr_y, ptr_z);
    ValueType* h_y = (ValueType*)malloc(sizeof(ValueType) * nrow);
    ValueType* h_z = (ValueType*)malloc(sizeof(ValueType) * nrow);
    cudaMemcpy(h_y, y, sizeof(ValueType) * nrow, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, sizeof(ValueType) * nrow, cudaMemcpyDeviceToHost);
    Fun_Check<ValueType>(h_y, h_z, nrow);

    //回收空间
    FREE(h_y);
    FREE(h_z);
    CUDAFREE(offset);
    CUDAFREE(d_x);
    CUDAFREE(d_y);
    CUDAFREE(d_z);
}

//重排技术
/*
*随机重排
*/
template<typename ValueType> 
void GRAPH<ValueType>::relabel_random()
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);

    timer.Start();
    //step 1：创建一个随机数生成器
    thrust::default_random_engine rng;
    rng.seed(static_cast<unsigned int>(graph_id));

    //step 2：使用 thrust::shuffle 来重排向量元素
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::shuffle(ptr_perm, ptr_perm + nrow, rng);
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 3：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&random&&" << tim1 << "&" << tim2;
}

/*
 * 按出度大小重排
*/
template<typename ValueType> 
void GRAPH<ValueType>::relabel_degree()
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);

    timer.Start();
    //step 1：统计顶点出度
    int*     d_deg;
    cudaMalloc((void**)&d_deg, sizeof(int) * nrow);   
    cudaMemset(d_deg, 0, sizeof(int) * nrow);
    const int block = 256;
    int grid = (nnz + block - 1) / block;
    desc_kernel1<<<grid, block>>>(row_ind, nnz, d_deg);
    thrust::device_ptr<int>  ptr_deg(d_deg);

    //step 2：生成用于排序的辅助数组
    int* d_sequence;
    cudaMalloc((void**)&d_sequence, sizeof(int) * nrow);
    thrust::device_ptr<int> ptr_sequence(d_sequence);
    thrust::sequence(ptr_sequence, ptr_sequence + nrow);

    //step 3：按照非零元素数量降序排序列索引
    thrust::sort_by_key(ptr_deg, ptr_deg + nrow, ptr_sequence, thrust::greater<int>());
    //std::cout << "d_indices: ";
    //thrust::copy(ptr_indices, ptr_indices + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 4：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_sequence, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 5：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&degree&&" << tim1 << "&" << tim2;

    //step 6：回收内存
    CUDAFREE(d_deg);
    CUDAFREE(d_sequence);
}

/*
*RCM排序
*/
template<typename ValueType> 
void GRAPH<ValueType>::relabel_rcm()
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);
    
    //step 1：复制到内存
    int* h_row_ind = (int*)malloc(sizeof(int) * nnz);
    int* h_col_ind = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_row_ind, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_ind, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    unsigned int* h_sequence = (unsigned int*)malloc(sizeof(unsigned int) * nrow);
    unsigned int* d_sequence;
    cudaMalloc((void**)&d_sequence, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_sequence(d_sequence);
    
    timer.Start();
    //step 2：RCM
    R_Cuthill_Mckee(nrow, nnz, h_row_ind, h_col_ind, h_sequence);
    cudaMemcpy(d_sequence, h_sequence, sizeof(unsigned int) * nrow, cudaMemcpyHostToDevice);

    //step 3：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_sequence, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 4：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&rcm&&" << tim1 << "&" << tim2;

    //step 5：回收内存
    FREE(h_row_ind);
    FREE(h_col_ind);
    FREE(h_sequence);
    CUDAFREE(d_sequence);
}

/*
*rabbitorder
*/
template<typename ValueType> 
void GRAPH<ValueType>::rabbitorder()
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);
    
    //step 1：复制到内存
    int* h_row_ind = (int*)malloc(sizeof(int) * nnz);
    int* h_col_ind = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_row_ind, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_ind, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    unsigned int* h_sequence = (unsigned int*)malloc(sizeof(unsigned int) * nrow);
    unsigned int* d_sequence;
    cudaMalloc((void**)&d_sequence, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_sequence(d_sequence);
    
    timer.Start();
    //step 2：rabbit order
    rabbit_order(h_row_ind, h_col_ind, nrow, nnz, h_sequence);
    cudaMemcpy(d_sequence, h_sequence, sizeof(unsigned int) * nrow, cudaMemcpyHostToDevice);

    //step 3：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_sequence, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 4：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&rabbitorder&&" << tim1 << "&" << tim2;

    //step 5：回收内存
    FREE(h_row_ind);
    FREE(h_col_ind);
    FREE(h_sequence);
    CUDAFREE(d_sequence);
}

/*
*DBG
*/
struct DegId
{
    int   deg = 0;
    int   id = 0;
    __host__ __device__ DegId() = default;
    __host__ __device__ DegId(const int _deg, const int _id): deg(_deg), id(_id) {};
    friend std::ostream& operator<<(std::ostream& os, const DegId& degid)
    {
        os << degid.deg << " " << degid.id << "\n";
        return os;
    }
};

struct DegIdCompare 
{
    __host__ __device__ bool operator()(const DegId& _lhs, const DegId& _rhs) const 
    {
        return (_lhs.deg > _rhs.deg) || (_lhs.deg == _rhs.deg && _lhs.id < _rhs.id);
    }
};

__global__ void dbg_kernel(const int*   offset,
                           const int    nrow,
                           DegId*       degid)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < nrow)
    {
        degid[gid] = DegId(ceilf(log2(offset[gid + 1] - offset[gid] + 1.0)), gid);
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::dbg(const ENV& _env)
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);

    timer.Start();
    //step 1：生成offset
    sort(true); 
    int* offset; 
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);

    //step 2：生成degid
    DegId* degid;
    cudaMalloc((void**)&degid, sizeof(DegId) * nrow);
    thrust::device_ptr<DegId>  ptr_degid(degid);
    const int block = 256;
    const int grid = (nrow + block - 1) / block;
    dbg_kernel<<<grid, block>>>(offset, nrow, degid);

    //std::vector<DegId> host_degid(nrow);
    //cudaMemcpy(host_degid.data(), thrust::raw_pointer_cast(ptr_degid), sizeof(DegId) * nrow, cudaMemcpyDeviceToHost);
    //for (int i = 0; i < nrow; ++i) 
    //{
    //    std::cout << host_degid[i];  // 已重载 << 操作符
    //}
    

    //step 3：生成用于排序的辅助数组
    int* d_sequence;
    cudaMalloc((void**)&d_sequence, sizeof(int) * nrow);
    thrust::device_ptr<int> ptr_sequence(d_sequence);
    thrust::sequence(ptr_sequence, ptr_sequence + nrow);

    //step 3：按照非零元素数量降序排序列索引
    thrust::sort_by_key(ptr_degid, ptr_degid + nrow, ptr_sequence, DegIdCompare());
    cudaDeviceSynchronize();
    //std::cout << "d_indices: ";
    //thrust::copy(ptr_indices, ptr_indices + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 4：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_sequence, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 5：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&dbg&&" << tim1 << "&" << tim2;

    //step 6：回收内存
    CUDAFREE(offset);
    CUDAFREE(degid);
    CUDAFREE(d_sequence);
}

/*
*slashburn
*/
template<typename ValueType> 
void GRAPH<ValueType>::relabel_slashburn(const ENV& _env)
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);

    //step 1：变量赋值
    std::fstream  file;
    std::string   line = "";
    int           id  = 0;
    int           i   = 0;
    std::string   filename = "";
    
    file.open(_env.config, std::fstream::ios_base::in);
    do
    {
        std::getline(file, line);
        std::stringstream(line) >> id >> filename;
    } while (id != graph_id);
    file.close();

    //step 2：生成perm
    CLApp     cli(filename, 0.005, 32);
    Builder   b(cli);
    Graph g = b.MakeGraph();
    int n_neighbour_rounds = 2;
    Bitmap bmap(g.num_nodes());
    timer.Start();
    SlashBurn sb = SlashBurn(g, n_neighbour_rounds, cli.percent(), bmap, cli.num_threads());
    timer.Stop();
    tim1 = timer.Millisecs();

    //step 3：将perm写入显存
    int* h_perm = (int*)malloc(sizeof(int) * g.num_nodes());
    for(auto& val : sb.perm)
    {
        h_perm[i] = val;
        ++i;
    }
    cudaMemcpy(perm, h_perm, sizeof(int) * nrow, cudaMemcpyHostToDevice);
    thrust::device_ptr<int> ptr_perm(perm);
    
    //step 4：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    timer.Stop();
    cudaDeviceSynchronize();
    tim2 = timer.Millisecs();
    FSTR << "&" << tim1 << "&" << tim2;

    //step 4：回收空间
    FREE(h_perm);
}

/*
*GOrder
*/
template<typename ValueType> 
void GRAPH<ValueType>::gorder1(const ENV&  _env, 
                               const int   _width)
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);
    
    //step 1：将数据由显存复制到内存
    sort(true);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    int* h_offset1 = (int*)malloc(sizeof(int) * (nrow + 1));
    int* h_indice1 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset1, offset, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    int* h_offset2 = (int*)malloc(sizeof(int) * (nrow + 1));
    int* h_indice2 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset2, offset, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    
    timer.Start();
    //step 2：变量定义
    using ptrNode = std::unique_ptr<Node>;
    ptrNode* NODE = (ptrNode*)calloc(nrow, sizeof(ptrNode));
    Node*                                     node;  //节点
    Node*                                     tmp1;  //节点
    Node*                                     tmp2;  //节点
    Integer_Heap                              PQUE(nnz);  //整数堆
    int                                       ID(0);
    int                                       L = std::sqrt(nrow) / 4;
    //WINDOW<Node*, PtrNodeLess>                window(_width);

    //step 3：生成度数向量，此处只是在起始点选择入度最大的点，如果改点位于一个单独的强连通块，则扩展到其它连通块的时候不是按照入度最大的规则执行的
    thrust::device_vector<int>     vec_deg(nrow + 1);
    thrust::device_ptr<int>        ptr_offset(offset);
    thrust::adjacent_difference(ptr_offset, ptr_offset + nrow, vec_deg.begin());
    vec_deg.erase(vec_deg.begin());
    ID = thrust::max_element(vec_deg.begin(), vec_deg.end()) - vec_deg.begin();
    
    //step 4：创建节点并生成堆
    for(int i=0; i<nrow; ++i)
    {
        NODE[i] = std::make_unique<Node>(i);
        node = NODE[i].get();
        PQUE.push(node);
    }
    node = NODE[ID].get();
    PQUE.modify(node, 1);

    //step 5：开始循环
    int* h_permutation = (int*)calloc(nrow, sizeof(int));
    ID = 0;
    while(PQUE.CNT>0)
    {
        //出队列
        PQUE.pop(node);
        node->STS = false;
        h_permutation[ID] = node->ID;

        //升值
        if(h_offset1[node->ID + 1] - h_offset1[node->ID] < L)
        {
            for(int j=h_offset1[node->ID]; j<h_offset1[node->ID + 1]; ++j) 
            {
                tmp1 = NODE[h_indice1[j]].get();
                if(tmp1->STS)
                {
                    PQUE.modify(tmp1, 1);
                }
            }
        }
        
        for(int i=h_offset2[node->ID]; i<h_offset2[node->ID + 1]; ++i)
        {
            tmp1 = NODE[h_indice2[i]].get();
            if(tmp1->STS)
            {
                PQUE.modify(tmp1, 1);
            }
            if(h_offset1[tmp1->ID + 1] - h_offset1[tmp1->ID] < L)
            {
                for(int j=h_offset1[tmp1->ID]; j<h_offset1[tmp1->ID + 1]; ++j)
                {
                    tmp2 = NODE[h_indice1[j]].get();
                    if(tmp2->STS)
                    {
                        PQUE.modify(tmp2, 1);
                    }
                }
            }
        }
        
        //std::cout << node->ID << ", " << node->key1 << ", " << node->key2 << " | ";
        //window.replace(node);
        //std::cout << node->ID << ", " << node->key1 << ", " << node->key2 << std::endl;

        //for (auto& val : window.data)
        //{
        //    //val->key1 -= 1;
        //    val->key2 = 0;
        //}

        //降职
        if(ID >= _width)
        {
            node = NODE[h_permutation[ID - _width]].get();
            
            if(h_offset1[node->ID + 1] - h_offset1[node->ID] < L)
            {
                for(int j=h_offset1[node->ID]; j<h_offset1[node->ID + 1]; ++j) 
                {
                    tmp1 = NODE[h_indice1[j]].get();
                    if(tmp1->STS)
                    {
                        PQUE.modify(tmp1, -1);
                    }
                }
            }
            
            for(int i=h_offset2[node->ID]; i<h_offset2[node->ID + 1]; ++i)
            {
                tmp1 = NODE[h_indice2[i]].get();
                if(tmp1->STS)
                {
                    PQUE.modify(tmp1, -1);
                }
                
                if(h_offset1[tmp1->ID + 1] - h_offset1[tmp1->ID] < L)
                {
                    for(int j=h_offset1[tmp1->ID]; j<h_offset1[tmp1->ID + 1]; ++j)
                    {
                        tmp2 = NODE[h_indice1[j]].get();
                        if(tmp2->STS)
                        {
                            PQUE.modify(tmp2, -1);
                        }
                    }
                }
            }
        }

        ++ID;
    }

    //step 6：生成用于排序的辅助数组
    int* d_permutation;
    cudaMalloc((void**)&d_permutation, sizeof(int) * nrow);
    cudaMemcpy(d_permutation, h_permutation, sizeof(int) * nrow, cudaMemcpyHostToDevice);
    thrust::device_ptr<int> ptr_permutation(d_permutation);
    //std::cout << "d_permutation: ";
    //thrust::copy(ptr_permutation, ptr_permutation + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 7：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_permutation, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 8：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&gorder1&&" << tim1 << "&" << tim2;
    
    //step 9：回收内存
    FREE(NODE);
    FREE(h_offset1);
    FREE(h_offset2);
    FREE(h_indice1);
    FREE(h_indice2);
    FREE(h_permutation);
    CUDAFREE(offset);
    CUDAFREE(d_permutation);
}


//GOrder_GPU
__global__ void gorder_gpu_kernel1(const int*    offset, 
                                   const int     nrow, 
                                   unsigned int* d_key_in, 
                                   unsigned int* d_val_in)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid < nrow)
    {
        unsigned int val = offset[gid + 1] - offset[gid];
        d_key_in[gid] = gid;
        d_val_in[gid] = val | (0x1u << 31);
    }
}

__global__ void gorder_gpu_kernel2(const unsigned int* d_key_out, 
                                   const int           nrow, 
                                   const int           len, 
                                   const int           batch_id,
                                   unsigned int*       d_val_in, 
                                   unsigned int*       d_permutation, 
                                   unsigned long long* d_values)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid < nrow)
    {
        d_val_in[d_key_out[gid]] &= 0x80000000u;  //按位与运算，第31位为1的保留，为0的就是0了
        if(gid < len)
        {   
            d_val_in[d_key_out[gid]] = 0x0u;
            d_permutation[gid] = d_key_out[gid];
            d_values[gid] = (static_cast<unsigned long long>(batch_id) << 32) | static_cast<unsigned long long>(d_key_out[gid]);
        }
    }
}

//cuda graph不支持动态并行
//一个block处理一个点u，一个warp处理一个点v，一个thread处理一个w
__global__ void gorder_gpu_kernel3(const unsigned int* d_permutation, 
                                   const int           omega_start, 
                                   const int*          offset1, 
                                   const int*          indice1, 
                                   const int*          offset2, 
                                   const int*          indice2, 
                                   unsigned int*       d_val_in)
{
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x % 32;

    int xid = d_permutation[omega_start + blockIdx.x];
    int start_y = offset2[xid];
    int end_y = offset2[xid + 1];
    for(int idy=start_y+wid; idy<end_y; idy+=blockDim.x/32)
    {
        int yid = indice2[idy];
        int start_z = offset1[yid];
        int end_z = offset1[yid + 1];
        for(int idz=start_z+lid; idz<end_z; idz+=32)
        {
            int zid = indice1[idz];
            bool sts = d_val_in[zid] >> 31;
            
            if(sts)
            {
                atomicAdd(&d_val_in[zid], 1);
            }
        }
    }
}


template<typename ValueType> 
void GRAPH<ValueType>::gorder2(const ENV&  _env, 
                               const int   _vsize, 
                               const int   _width)
{
    Timer    timer;
    float tim1(0.0), tim2(0.0);

    //step 1：生成图数据
    int*           d_offset1;
    int*           d_indice1;
    int*           d_offset2;
    int*           d_indice2;
    cudaMalloc((void**)&d_offset1, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice1, sizeof(int) * nnz);
    cudaMalloc((void**)&d_offset2, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice2, sizeof(int) * nnz);

    sort(true);
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, d_offset1, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, d_offset2, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);
    
    timer.Start();
    //step 2：生成idval
    int                 block = 256;
    int                 grid = (nrow + block - 1) / block;
    int                 maxiter = (nrow + _vsize - 1) / _vsize;
    unsigned int*       d_key_in;
    unsigned int*       d_val_in;
    unsigned int*       d_key_out;
    unsigned int*       d_val_out;
    unsigned long long* d_values_in;
    unsigned long long* d_values_out; //需要这两者是因为要考虑原始排序的信息
    unsigned int*       d_perm_in;
    unsigned int*       d_perm_out;
    cudaMalloc((void**)&d_key_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_key_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_values_in, sizeof(unsigned long long) * nrow);
    cudaMalloc((void**)&d_values_out, sizeof(unsigned long long) * nrow);
    cudaMalloc((void**)&d_perm_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_perm_out, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_perm_out(d_perm_out);
    
    gorder_gpu_kernel1<<<grid, block>>>(d_offset2, nrow, d_key_in, d_val_in);
    cudaMemcpy(d_key_out, d_key_in, sizeof(unsigned int) * nrow, cudaMemcpyDeviceToDevice);
    
    void*      d_temp_storage = nullptr;
    size_t     d_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow, 0, 32);
    cudaMalloc((void**)&d_temp_storage, d_temp_storage_bytes);
    //cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow, 0, 32);

    //step 3：开始执行
    for(int batch_id=0; batch_id<maxiter; ++batch_id)
    {
        int start = batch_id * _vsize;
        int len = (start + _vsize < nrow)? _vsize : nrow - start;
        int omega_start = (batch_id < _width)? 0 : (batch_id - _width + 1) * _vsize;
        int omega_len = (batch_id < _width)? (batch_id + 1) * _vsize : _width * _vsize;
        omega_len = (omega_start + omega_len < nrow)? omega_len : nrow - omega_start;

        //step 3.1：取前len个点
        gorder_gpu_kernel2<<<grid, block>>>(d_key_out, nrow, len, batch_id, d_val_in, &d_perm_in[start], &d_values_in[start]);

        //step 3.2：重新计算指标
        gorder_gpu_kernel3<<<omega_len, block>>>(d_perm_in, omega_start, d_offset1, d_indice1, d_offset2, d_indice2, d_val_in);

        //step 3.3：排序
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow, 0, 32);
    }
    
    //重新排序，保留原始排序信息
    //声明两对缓冲区
    cub::DoubleBuffer<unsigned int>         key_buf(d_perm_in, d_perm_out);
    cub::DoubleBuffer<unsigned long long>   val_buf(d_values_in, d_values_out);
    d_temp_storage_bytes = 0;
    CUDAFREE(d_temp_storage);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, d_temp_storage_bytes, val_buf, key_buf, nrow);
    cudaMalloc((void**)&d_temp_storage, d_temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, d_temp_storage_bytes, val_buf, key_buf, nrow);


    //step 4：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_perm_out, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "d_permutation: ";
    //CUDASHOW(d_permutation, unsigned int, nrow);

    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 5：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&gorder2&" << _vsize << "&" << tim1 << "&" << tim2;

    //step 6：回收内存
    CUDAFREE(d_key_in);
    CUDAFREE(d_key_out);
    CUDAFREE(d_val_in);
    CUDAFREE(d_val_out);
    CUDAFREE(d_values_in);
    CUDAFREE(d_values_out);
    CUDAFREE(d_perm_in);
    CUDAFREE(d_perm_out);
    CUDAFREE(d_temp_storage);
    CUDAFREE(d_offset1);
    CUDAFREE(d_offset2);
    CUDAFREE(d_indice1);
    CUDAFREE(d_indice2);
}



//GPU-CPU HYBRID GRAPH ORDER
//graph order on cpus, only generating h_ermutation
void order_on_cpu(const std::vector<std::unique_ptr<Node>>* NODE,  
                  const int*                                h_offset1, 
                  const int*                                h_indice1, 
                  const int*                                h_offset2, 
                  const int*                                h_indice2, 
                  unsigned int*                             h_permutation,        
                  int                                       batch_id,
                  int                                       len,  
                  int                                       width,
                  int                                       degree)
{
    //step 1：变量定义
    Node*            node;  //节点
    Node*            tmp1;  //节点
    Node*            tmp2;  //节点
    Integer_Heap     PQUE(width * len * 1000);  //整数堆
    int              ID(h_permutation[0]);
    int              deg = h_offset1[ID + 1] - h_offset1[ID];

    for (int i=0; i<len; ++i)
    {
        if (h_offset1[h_permutation[i] + 1] - h_offset1[h_permutation[i]] > deg)
        {
            deg = h_offset1[h_permutation[i] + 1] - h_offset1[h_permutation[i]];
            ID = h_permutation[i];
        }
    }

    //step 2：创建节点并生成堆
    for(int i=0; i<len; ++i)
    {
        node = (*NODE)[h_permutation[i]].get();
        node->batch_id = batch_id;
        PQUE.push(node);
    }
    node = (*NODE)[ID].get();
    PQUE.modify(node, 1);

    //step 3：开始循环
    ID = 0;
    while(PQUE.CNT>0)
    {
        //出队列
        PQUE.pop(node);
        node->STS = false;
        h_permutation[ID] = node->ID;

        //升值
        if(h_offset2[node->ID + 1] - h_offset2[node->ID] < degree)
        {
            for(int j=h_offset2[node->ID]; j<h_offset2[node->ID + 1]; ++j)
            {
                tmp1 = (*NODE)[h_indice2[j]].get();
                if(h_offset1[tmp1->ID + 1] - h_offset1[tmp1->ID] < degree)
                {
                    for(int j=h_offset1[tmp1->ID]; j<h_offset1[tmp1->ID + 1]; ++j)
                    {
                        tmp2 = (*NODE)[h_indice1[j]].get();
                        if(tmp2->batch_id == batch_id && tmp2->STS)
                        {
                            PQUE.modify(tmp2, 1);
                        }
                    }
                }
            }
        }

        //降职
        if(ID >= width)
        {
            node = (*NODE)[h_permutation[ID - width]].get();
            if(h_offset2[node->ID + 1] - h_offset2[node->ID] < degree)
            {
                for(int j=h_offset2[node->ID]; j<h_offset2[node->ID + 1]; ++j)
                {
                    tmp1 = (*NODE)[h_indice2[j]].get();

                    if(h_offset1[tmp1->ID + 1] - h_offset1[tmp1->ID] < degree)
                    {
                        for(int j=h_offset1[tmp1->ID]; j<h_offset1[tmp1->ID + 1]; ++j)
                        {
                            tmp2 = (*NODE)[h_indice1[j]].get();
                            if(tmp2->batch_id == batch_id && tmp2->STS)
                            {
                                PQUE.modify(tmp2, -1);
                            }
                        }
                    }
                }
            }
        }
    
        ++ID;
    }
}

struct HOSTPARAM
{
    ThreadPool*                            threadpool;
    std::vector<std::future<void>>*        futures;
    std::vector<std::unique_ptr<Node>>*    NODE;
    int*                                   h_offset1;
    int*                                   h_indice1;
    int*                                   h_offset2;
    int*                                   h_indice2;
    unsigned int*                          h_permutation;
    int                                    batch_id;
    int                                    len;
    int                                    width;
    int                                    degree;                               
    HOSTPARAM(ThreadPool* _threadpool, std::vector<std::future<void>>* _futures, std::vector<std::unique_ptr<Node>>* _NODE, int* _h_offset1, int* _h_indice1, int* _h_offset2, int* _h_indice2, unsigned int* _h_permutation, int _batch_id, int _len, int _width, int _degree): 
    threadpool(_threadpool), futures(_futures), NODE(_NODE), h_offset1(_h_offset1), h_indice1(_h_indice1), h_offset2(_h_offset2), h_indice2(_h_indice2), h_permutation(_h_permutation)
    { 
        batch_id = _batch_id;
        len = _len;
        width = _width;
        degree = _degree;
    };
};

void hostFun(void* _hostparam)
{
    auto param = static_cast<HOSTPARAM*>(_hostparam);
    param->futures->emplace_back(param->threadpool->enqueue(order_on_cpu, 
                                                            param->NODE,
                                                            param->h_offset1,
                                                            param->h_indice1,
                                                            param->h_offset2,
                                                            param->h_indice2,
                                                            param->h_permutation,
                                                            param->batch_id, 
                                                            param->len,
                                                            param->width,
                                                            param->degree));
}

//GOrder GPU2
__global__ void gorder_gpu_cpu_kernel1(const int*    offset, 
                                       const int     nrow, 
                                       unsigned int* d_key_in, 
                                       unsigned int* d_val_in)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid < nrow)
    {
        unsigned int val = offset[gid + 1] - offset[gid];
        d_key_in[gid] = gid;
        d_val_in[gid] = val | (0x1u << 31);
    }
}

__global__ void gorder_gpu_cpu_kernel2(const unsigned int* d_key_out, 
                                       const int           nrow, 
                                       const int           len, 
                                       unsigned int*       d_val_in, 
                                       unsigned int*       d_permutation, 
                                       const int           start)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid < nrow)
    {
        d_val_in[d_key_out[gid]] &= 0x80000000u;  //一个线程只能处理一个点
        if(gid < len)
        {   
            d_val_in[d_key_out[gid]] = 0x0u;
            d_permutation[start + gid] = d_key_out[gid];
        }
    }
}

//cuda graph不支持动态并行
//一个block处理一个点u，一个warp处理一个点v，一个thread处理一个w
__global__ void gorder_gpu_cpu_kernel3(const unsigned int* d_permutation, 
                                       const int           omega_start, 
                                       const int*          offset1, 
                                       const int*          indice1, 
                                       const int*          offset2, 
                                       const int*          indice2, 
                                       unsigned int*       d_val_in)
{
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x % 32;

    int xid = d_permutation[omega_start + blockIdx.x]; //每个x分配一个block
    int start_y = offset2[xid];
    int end_y = offset2[xid + 1];
    for(int idy=start_y+wid; idy<end_y; idy+=blockDim.x/32)
    {
        int yid = indice2[idy];
        int start_z = offset1[yid];
        int end_z = offset1[yid + 1];
        for(int idz=start_z+lid; idz<end_z; idz+=32)
        {
            int zid = indice1[idz];
            bool sts = d_val_in[zid] >> 31;
            
            if(sts)
            {
                atomicAdd(&d_val_in[zid], 1);
            }
        }
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::gorder3(const ENV&  _env, 
                                      const int   _vsize, 
                                      const int   _width1, 
                                      const int   _width2)
{
    //变量赋值
    Timer  timer;
    float  tim1(0.0), tim2(0.0);

    //step 1：生成图数据
    //显存
    int*           d_offset1; //出度
    int*           d_indice1;
    int*           d_offset2; //入度
    int*           d_indice2;
    cudaMalloc((void**)&d_offset1, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice1, sizeof(int) * nnz);
    cudaMalloc((void**)&d_offset2, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice2, sizeof(int) * nnz);

    sort(true);
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, d_offset1, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, d_offset2, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    //复制到内存
    int*            h_offset1 = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice1 = (int*)malloc(sizeof(int) * nnz);
    int*            h_offset2 = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice2 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset1, d_offset1, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice1, d_indice1, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offset2, d_offset2, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice2, d_indice2, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    timer.Start();
    //step 2：GPU部分
    //idVal重新设计，借助int64_t 类型，用bit位，结构为 sts(1位)|val(31位)|Id(32)位
    //排序可以用radixsort
    int                 block = 256;
    int                 grid = (nrow + block - 1) / block;
    unsigned int*       d_key_in;
    unsigned int*       d_val_in;
    unsigned int*       d_key_out;
    unsigned int*       d_val_out;
    unsigned int*       d_permutation;
    cudaMalloc((void**)&d_key_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_key_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_permutation, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_permutation(d_permutation);
    
    //初始化按出度最大
    gorder_gpu_cpu_kernel1<<<grid, block>>>(d_offset1, nrow, d_key_in, d_val_in);
    cudaMemcpy(d_key_out, d_key_in, sizeof(unsigned int) * nrow, cudaMemcpyDeviceToDevice);
    
    void*      d_temp_storage = nullptr;
    size_t     d_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow, 0, 32, _env.stream);
    cudaMalloc((void**)&d_temp_storage, d_temp_storage_bytes);

    //step 3：CPU排序部分
    //统一生成NODE
    std::vector<std::unique_ptr<Node>>    NODE(nrow);
    #pragma omp parallel for
    for(int i=0; i<nrow; ++i)
    {
        NODE[i] = std::make_unique<Node>(i);
    }
    unsigned int* h_permutation = (unsigned int*)malloc(sizeof(unsigned int) * nrow);
    
    //定义线程池
    int maxiter = (nrow + _vsize - 1) / _vsize;
    ThreadPool                         threadpool(72);
    std::vector<std::future<void>>     futures;
    std::vector<HOSTPARAM>             host_params;
    futures.reserve(maxiter);
    host_params.reserve(maxiter);
    
    //step 4：生成图
    cudaGraph_t  graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t kernelNode2 = nullptr, memcpyNode = nullptr, kernelNode3 = nullptr, hostNode = nullptr;
    for(int batch_id=0; batch_id<maxiter; ++batch_id)
    {
        int start = batch_id * _vsize;
        int len = (start + _vsize < nrow)? _vsize : nrow - start;
        int omega_start = (batch_id < _width1)? 0 : (batch_id - _width1 + 1) * _vsize;
        int omega_len = (batch_id < _width1)? (batch_id + 1) * _vsize : _width1 * _vsize;
        omega_len = (omega_start + omega_len < nrow)? omega_len : nrow - omega_start;

        //step 4.1：添加kernelNode2, 取前len个点
        cudaKernelNodeParams   paramKernel2 = {0};
        void* args[] = {(void*)&(d_key_out), (void*)&nrow, (void*)&len, (void*)&d_val_in, (void*)&d_permutation, (void*)&start};
        paramKernel2.func = (void*)gorder_gpu_cpu_kernel2;
        paramKernel2.gridDim = dim3(grid);
        paramKernel2.blockDim = dim3(block);
        paramKernel2.kernelParams = args;
        cudaGraphAddKernelNode(&kernelNode2, graph, &kernelNode3, (batch_id==0)? 0 : 1, &paramKernel2);

        //step 4.2：添加数据传输节点
        cudaGraphAddMemcpyNode1D(&memcpyNode, graph, &kernelNode2, 1, &h_permutation[start], &d_permutation[start], sizeof(unsigned int) * len, cudaMemcpyDeviceToHost);

        //step 4.3：添加kernelNode3计算指标并排序
        cudaGraph_t sub_graph;
        cudaStreamBeginCapture(_env.stream, cudaStreamCaptureModeGlobal);
        gorder_gpu_cpu_kernel3<<<omega_len, block, 0, _env.stream>>>(d_permutation, omega_start, d_offset1, d_indice1, d_offset2, d_indice2, d_val_in);
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow, 0, 32, _env.stream); //排序                                  //取前len个
        cudaStreamEndCapture(_env.stream, &sub_graph);
        cudaGraphAddChildGraphNode(&kernelNode3, graph, &kernelNode2, 1, sub_graph);
        cudaGraphDestroy(sub_graph);

        //step 4.4：添加主机函数节点
        cudaHostNodeParams     paramHost = {0};
        paramHost.fn = hostFun;
        host_params.emplace_back(&threadpool, &futures, &NODE, h_offset1, h_indice1, h_offset2, h_indice2, &h_permutation[start], batch_id, len, _width2, 1000);
        paramHost.userData = (void*)(&host_params.back());
        cudaGraphAddHostNode(&hostNode, graph, &memcpyNode, 1, &paramHost);
    }

    //step 5：实例化并执行主图
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(graph_exec, _env.stream);
    cudaStreamSynchronize(_env.stream);
    cudaDeviceSynchronize();

    for(auto& fu : futures) 
    {
        fu.get();
    }
    cudaMemcpy(d_permutation, h_permutation, sizeof(unsigned int) * nrow, cudaMemcpyHostToDevice);
    cudaGraphExecDestroy(graph_exec); 
    cudaGraphDestroy(graph);

    //step 6：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_permutation, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "d_permutation: ";
    //CUDASHOW(d_permutation, unsigned int, nrow);
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 7：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&gorder3&" << _vsize << "&" << tim1 << "&" << tim2;

    //step 8：回收内存
    CUDAFREE(d_key_in);
    CUDAFREE(d_key_out);
    CUDAFREE(d_val_in);
    CUDAFREE(d_val_out);
    CUDAFREE(d_permutation);
    CUDAFREE(d_temp_storage);
    CUDAFREE(d_offset1);
    CUDAFREE(d_offset2);
    CUDAFREE(d_indice1);
    CUDAFREE(d_indice2);
    FREE(h_offset1);
    FREE(h_indice1);
    FREE(h_offset2);
    FREE(h_indice2);
    FREE(h_permutation);
}


//gorder_cpu_parallel
template<typename ValueType> 
void GRAPH<ValueType>::gorder4(const ENV& _env, const int _vsize, const int _width)
{
    //变量赋值
    Timer  timer;
    float  tim1(0.0), tim2(0.0);

    //step 1：生成图数据
    //显存
    int*           d_offset1;
    int*           d_indice1;
    int*           d_offset2;
    int*           d_indice2;
    unsigned int*  d_permutation;
    cudaMalloc((void**)&d_offset1, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice1, sizeof(int) * nnz);
    cudaMalloc((void**)&d_offset2, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice2, sizeof(int) * nnz);
    cudaMalloc((void**)&d_permutation, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_permutation(d_permutation);

    sort(true);
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, d_offset1, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, d_offset2, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    //复制到内存
    int*            h_offset1  = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice1 = (int*)malloc(sizeof(int) * nnz);
    int*            h_offset2  = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice2 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset1, d_offset1, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice1, d_indice1, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offset2, d_offset2, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice2, d_indice2, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    //生成perm
    unsigned int*   h_permutation = (unsigned int*)malloc(sizeof(unsigned int) * nrow);
    std::iota(h_permutation, h_permutation + nrow, 0);

    timer.Start();
    //step 2：排序
    //统一生成NODE
    std::vector<std::unique_ptr<Node>>    NODE(nrow);
    #pragma omp parallel for
    for(int i=0; i<nrow; ++i)
    {
        NODE[i] = std::make_unique<Node>(i);
    }
    
    //定义线程池
    int maxiter = (nrow + _vsize - 1) / _vsize;
    ThreadPool                         threadpool(128);
    std::vector<std::future<void>>     futures;
    futures.reserve(maxiter);
    int                                L = std::sqrt(nrow) / 4;
    
    //step 4：将参数插入线程池任务队列
    for(int batch_id=0; batch_id<maxiter; ++batch_id)
    {
        int start = batch_id * _vsize;
        int len = (start + _vsize < nrow)? _vsize : nrow - start;
        futures.emplace_back(threadpool.enqueue([&NODE, h_offset1, h_indice1, h_offset2, h_indice2, h_permutation, start, batch_id, len, _width, L]()
        {
            order_on_cpu(&NODE, h_offset1, h_indice1, h_offset2, h_indice2, &h_permutation[start], batch_id, len, _width, L);
        }));
    }

    for(auto& fu : futures) 
    {
        fu.get();
    }
    
    //复制到显存
    cudaMemcpy(d_permutation, h_permutation, sizeof(unsigned int) * nrow, cudaMemcpyHostToDevice);

    //step 6：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_permutation, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "d_permutation: ";
    //CUDASHOW(d_permutation, unsigned int, nrow);
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 7：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&gorder4&" << _vsize << "&" << tim1 << "&" << tim2;

    //step 8：回收内存
    CUDAFREE(d_permutation);
    CUDAFREE(d_offset1);
    CUDAFREE(d_offset2);
    CUDAFREE(d_indice1);
    CUDAFREE(d_indice2);
    FREE(h_offset1);
    FREE(h_indice1);
    FREE(h_offset2);
    FREE(h_indice2);
    FREE(h_permutation);
}

/*
*gorder_gpu_cpu_seq
*/
template<typename ValueType> 
void GRAPH<ValueType>::gorder5(const ENV&  _env, 
                               const int   _vsize, 
                               const int   _width1, 
                               const int   _width2)
{
    //变量赋值
    Timer  timer;
    float  tim1(0.0), tim2(0.0);

    //step 1：生成图数据
    //显存
    int*           d_offset1; //出度
    int*           d_indice1;
    int*           d_offset2; //入度
    int*           d_indice2;
    cudaMalloc((void**)&d_offset1, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice1, sizeof(int) * nnz);
    cudaMalloc((void**)&d_offset2, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice2, sizeof(int) * nnz);

    sort(true);
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, d_offset1, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, d_offset2, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    //复制到内存
    int*            h_offset1 = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice1 = (int*)malloc(sizeof(int) * nnz);
    int*            h_offset2 = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice2 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset1, d_offset1, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice1, d_indice1, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offset2, d_offset2, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice2, d_indice2, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    timer.Start();
    //公共变量
    unsigned int* h_permutation = (unsigned int*)malloc(sizeof(unsigned int) * nrow);
    int maxiter = (nrow + _vsize - 1) / _vsize;

    //step 2：GPU部分
    //idVal重新设计，借助int64_t 类型，用bit位，结构为 sts(1位)|val(31位)|Id(32)位
    //排序可以用radixsort
    int                 block = 256;
    int                 grid = (nrow + block - 1) / block;
    unsigned int*       d_key_in;
    unsigned int*       d_val_in;
    unsigned int*       d_key_out;
    unsigned int*       d_val_out;
    unsigned int*       d_permutation;
    cudaMalloc((void**)&d_key_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_key_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_permutation, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_permutation(d_permutation);
    
    //初始化按出度最大
    gorder_gpu_cpu_kernel1<<<grid, block>>>(d_offset1, nrow, d_key_in, d_val_in);
    cudaMemcpy(d_key_out, d_key_in, sizeof(unsigned int) * nrow, cudaMemcpyDeviceToDevice);
    
    void*      d_temp_storage = nullptr;
    size_t     d_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow);
    cudaMalloc((void**)&d_temp_storage, d_temp_storage_bytes);
    
    for(int batch_id=0; batch_id<maxiter; ++batch_id)
    {
        int start = batch_id * _vsize;
        int len = (start + _vsize < nrow)? _vsize : nrow - start;
        int omega_start = (batch_id < _width1)? 0 : (batch_id - _width1 + 1) * _vsize;
        int omega_len = (batch_id < _width1)? (batch_id + 1) * _vsize : _width1 * _vsize;
        omega_len = (omega_start + omega_len < nrow)? omega_len : nrow - omega_start;
        
        //step 2.1：排序
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow);

        //step 2.2：取前len个点
        gorder_gpu_cpu_kernel2<<<grid, block>>>(d_key_out, nrow, len, d_val_in, d_permutation, start);

        //step 2.3：重新计算指标
        gorder_gpu_cpu_kernel3<<<omega_len, block>>>(d_permutation, omega_start, d_offset1, d_indice1, d_offset2, d_indice2, d_val_in); 
    }

    //step 3：复制到CPU
    cudaMemcpy(h_permutation, d_permutation, sizeof(unsigned int) * nrow, cudaMemcpyDeviceToHost);
    
    //step 4：CPU排序部分
    //统一生成NODE
    std::vector<std::unique_ptr<Node>>    NODE(nrow);
    #pragma omp parallel for
    for(int i=0; i<nrow; ++i)
    {
        NODE[i] = std::make_unique<Node>(i);
    }
    
    //定义线程池
    ThreadPool                         threadpool(128);
    std::vector<std::future<void>>     futures;
    futures.reserve(maxiter);
    int                                L = std::sqrt(nrow) / 4;
    
    //step 4：生成图
    for(int batch_id=0; batch_id<maxiter; ++batch_id)
    {
        int start = batch_id * _vsize;
        int len = (start + _vsize < nrow)? _vsize : nrow - start;
        futures.emplace_back(threadpool.enqueue([&NODE, h_offset1, h_indice1, h_offset2, h_indice2, h_permutation, start, batch_id, len, _width2, L]()
        {                                        
            order_on_cpu(&NODE, h_offset1, h_indice1, h_offset2, h_indice2, &h_permutation[start], batch_id, len, _width2, L);
        }));
    }
    
    for(auto& fu : futures) 
    {
        fu.get();
    }
    
    cudaMemcpy(d_permutation, h_permutation, sizeof(unsigned int) * nrow, cudaMemcpyHostToDevice);
    
    //step 6：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_permutation, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "d_permutation: ";
    //CUDASHOW(d_permutation, unsigned int, nrow);
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 7：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&gorder5&" << _vsize << "&" << tim1 << "&" << tim2;

    //step 8：回收内存
    CUDAFREE(d_key_in);
    CUDAFREE(d_key_out);
    CUDAFREE(d_val_in);
    CUDAFREE(d_val_out);
    CUDAFREE(d_permutation);
    CUDAFREE(d_temp_storage);
    CUDAFREE(d_offset1);
    CUDAFREE(d_offset2);
    CUDAFREE(d_indice1);
    CUDAFREE(d_indice2);
    FREE(h_offset1);
    FREE(h_indice1);
    FREE(h_offset2);
    FREE(h_indice2);
    FREE(h_permutation);
}


/*
*gorder6
*/
__global__ void gorder6_kernel2(const unsigned int* d_key_out, 
                                const int           nrow, 
                                const int           len, 
                                unsigned int*       d_val_in, 
                                unsigned int*       d_permutation)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid < nrow)
    {
        d_val_in[d_key_out[gid]] &= 0x80000000u;  //一个线程只能处理一个点
        if(gid < len)
        {   
            d_val_in[d_key_out[gid]] = 0x0u;
            d_permutation[gid] = d_key_out[gid];
        }
    }
}

//cuda graph不支持动态并行
//一个block处理一个点u，一个warp处理一个点v，一个thread处理一个w
__global__ void gorder6_kernel3(const unsigned int* d_permutation, 
                                const int*          offset1, 
                                const int*          indice1, 
                                const int*          offset2, 
                                const int*          indice2, 
                                unsigned int*       d_val_in)
{
    const int wid = threadIdx.x / 32;
    const int lid = threadIdx.x % 32;

    int xid = d_permutation[blockIdx.x]; //每个x分配一个block
    int start_y = offset2[xid];
    int end_y = offset2[xid + 1];
    if (end_y - start_y < 10000)
    {
        for(int idy=start_y+wid; idy<end_y; idy+=blockDim.x/32)
        {
            int yid = indice2[idy];
            int start_z = offset1[yid];
            int end_z = offset1[yid + 1];
            
            if (end_z - start_z < 10000)
            {
                for(int idz=start_z+lid; idz<end_z; idz+=32)
                {
                    int zid = indice1[idz];
                    bool sts = d_val_in[zid] >> 31;
                    
                    if(sts)
                    {
                        atomicAdd(&d_val_in[zid], 1);
                    }
                }
            }
        }
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::gorder6(const ENV&  _env, 
                               const int   _width)
{
    //变量赋值
    Timer  timer;
    float  tim1(0.0), tim2(0.0);

    //step 1：生成图数据
    //显存
    int*           d_offset1; //出度
    int*           d_indice1;
    int*           d_offset2; //入度
    int*           d_indice2;
    cudaMalloc((void**)&d_offset1, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice1, sizeof(int) * nnz);
    cudaMalloc((void**)&d_offset2, sizeof(int) * (nrow + 1));
    cudaMalloc((void**)&d_indice2, sizeof(int) * nnz);

    sort(true);
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, d_offset1, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, d_offset2, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    cudaMemcpy(d_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);

    //复制到内存
    int*            h_offset1 = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice1 = (int*)malloc(sizeof(int) * nnz);
    int*            h_offset2 = (int*)malloc(sizeof(int) * (nrow + 1));
    int*            h_indice2 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset1, d_offset1, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice1, d_indice1, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offset2, d_offset2, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice2, d_indice2, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    timer.Start();
    //公共变量
    unsigned int* h_permutation = (unsigned int*)malloc(sizeof(unsigned int) * nrow);

    //基数为32，直到_width，引入Start和Leng数组
    int base = 5;
    int batch_size = (nrow + _width - 1) / _width + std::log2(_width / std::pow(2, base));

    int Pstart[batch_size] = {0};
    int Pleng[batch_size] = {std::pow(2, base)};
    int Ostart[batch_size] = {0};
    int Oleng[batch_size] = {std::pow(2, base)};
    for(int batch_id=1; batch_id<batch_size; ++batch_id)
    {
        int start(0), leng(0);
        start = (std::pow(2, batch_id + base - 1) < _width)? std::pow(2, batch_id + base - 1) : (batch_id - std::log2(_width / std::pow(2, base))) * _width;
        leng  = (std::pow(2, batch_id + base - 1) < _width)? std::pow(2, batch_id + base - 1) : _width;
        Pstart[batch_id] = start;
        Pleng[batch_id]  = (start + leng < nrow)? leng : nrow - start;
        
        start = (std::pow(2, batch_id + base - 1) < _width)? 0 : (batch_id - std::log2(_width / std::pow(2, base))) * _width;
        leng  = (std::pow(2, batch_id + base) < _width)? std::pow(2, batch_id + base) : _width;
        Ostart[batch_id] = start;
        Oleng[batch_id] = (start + leng < nrow)? leng : nrow - start;
    }
    //std::cout << std::endl;
    //std::cout << batch_size << std::endl;
    //SHOW(Pstart, 0, 20);
    //SHOW(Pleng, 0, 20);
    //SHOW(Ostart, 0, 20);
    //SHOW(Oleng, 0, 20);
    
    //step 2：GPU部分
    //排序可以用radixsort
    int                 block = 256;
    int                 grid = (nrow + block - 1) / block;
    unsigned int*       d_key_in;
    unsigned int*       d_val_in;
    unsigned int*       d_key_out;
    unsigned int*       d_val_out;
    unsigned int*       d_permutation;
    cudaMalloc((void**)&d_key_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_in, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_key_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_val_out, sizeof(unsigned int) * nrow);
    cudaMalloc((void**)&d_permutation, sizeof(unsigned int) * nrow);
    thrust::device_ptr<unsigned int> ptr_permutation(d_permutation);
    
    //初始化按出度最大
    gorder_gpu_cpu_kernel1<<<grid, block>>>(d_offset2, nrow, d_key_in, d_val_in);
    cudaMemcpy(d_key_out, d_key_in, sizeof(unsigned int) * nrow, cudaMemcpyDeviceToDevice);
    
    void*      d_temp_storage = nullptr;
    size_t     d_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow);
    cudaMalloc((void**)&d_temp_storage, d_temp_storage_bytes);

    for(int batch_id=0; batch_id<batch_size; ++batch_id)
    {
        //step 2.1：排序
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, d_temp_storage_bytes, d_val_in, d_val_out, d_key_in, d_key_out, nrow);

        //step 2.2：取前len个点
        gorder6_kernel2<<<grid, block>>>(d_key_out, nrow, Pleng[batch_id], d_val_in, &d_permutation[Pstart[batch_id]]);

        //step 2.3：重新计算指标
        gorder6_kernel3<<<Oleng[batch_id], block>>>(&d_permutation[Ostart[batch_id]], d_offset1, d_indice1, d_offset2, d_indice2, d_val_in); 
    }

    //step 3：复制到CPU
    cudaMemcpy(h_permutation, d_permutation, sizeof(unsigned int) * nrow, cudaMemcpyDeviceToHost);

    //step 4：CPU排序部分
    //统一生成NODE
    std::vector<std::unique_ptr<Node>>    NODE(nrow);
    #pragma omp parallel for
    for(int i=0; i<nrow; ++i)
    {
        NODE[i] = std::make_unique<Node>(i);
    }
    
    //定义线程池
    ThreadPool                         threadpool(128);
    std::vector<std::future<void>>     futures;
    futures.reserve(batch_size);
    int                                L = std::sqrt(nrow) / 4;
    
    //step 4：生成图
    for(int batch_id=std::log2(_width); batch_id<batch_size; ++batch_id)
    {
        int start = Ostart[batch_id];
        int leng  = Oleng[batch_id];
        futures.emplace_back(threadpool.enqueue([&NODE, h_offset1, h_indice1, h_offset2, h_indice2, h_permutation, start, batch_id, leng, L]()
        {
            order_on_cpu(&NODE, h_offset1, h_indice1, h_offset2, h_indice2, &h_permutation[start], batch_id, leng, 5, L);
        }));
    }
    
    for(auto& fu : futures) 
    {
        fu.get();
    }
    
    cudaMemcpy(d_permutation, h_permutation, sizeof(unsigned int) * nrow, cudaMemcpyHostToDevice);
    timer.Stop();
    tim1 = timer.Millisecs();

    //step 6：创建新的列索引映射
    timer.Start();
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_permutation, ptr_perm);
    cudaDeviceSynchronize();
    
    //std::cout << "d_permutation: ";
    //CUDASHOW(d_permutation, unsigned int, nrow);
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 7：更新COO格式的列索引
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    
    //FSTR << "&gorder6&" << _width << "&" << tim1 << "&" << tim2;
    FSTR << "&" << tim1 << "&" << tim2;
    
    //step 8：回收内存
    CUDAFREE(d_key_in);
    CUDAFREE(d_key_out);
    CUDAFREE(d_val_in);
    CUDAFREE(d_val_out);
    CUDAFREE(d_permutation);
    CUDAFREE(d_temp_storage);
    CUDAFREE(d_offset1);
    CUDAFREE(d_offset2);
    CUDAFREE(d_indice1);
    CUDAFREE(d_indice2);
    FREE(h_offset1);
    FREE(h_indice1);
    FREE(h_offset2);
    FREE(h_indice2);
    FREE(h_permutation);
}

/*
*order：窗口为block * sigma
*/
__global__ void order_kernel(const int* __restrict__ offset,
                             const int               nrow,
                             int*                    key,
                             int*                    val)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < nrow)
    {
        key[gid] = gid;
        val[gid] = offset[gid + 1] - offset[gid];
    }
}

template<typename ValueType> 
void GRAPH<ValueType>::order(const ENV&  _env,
                             const int   _width1,
                             const int   _width2)
{
    Timer   timer;
    float   tim1(0.0), tim2(0.0);
    
    //step 1：将数据由显存复制到内存
    sort(true);
    int* offset;
    cudaMalloc((void**)&offset, sizeof(int) * (nrow + 1));
    cusparseXcoo2csr(_env.handle_sparse, row_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    int* h_offset1 = (int*)malloc(sizeof(int) * (nrow + 1));
    int* h_indice1 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset1, offset, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice1, col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);

    sort(false);
    cusparseXcoo2csr(_env.handle_sparse, col_ind, nnz, nrow, offset, cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
    int* h_offset2 = (int*)malloc(sizeof(int) * (nrow + 1));
    int* h_indice2 = (int*)malloc(sizeof(int) * nnz);
    cudaMemcpy(h_offset2, offset, sizeof(int) * (nrow + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indice2, row_ind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    
    timer.Start();
    //step 2：变量定义
    using ptrNode = std::unique_ptr<Node>;
    ptrNode* NODE = (ptrNode*)calloc(nrow, sizeof(ptrNode));
    Node*                                     node;  //节点
    Node*                                     tmp;   //节点
    Integer_Heap                              PQUE(_width1);  //整数堆，最大block * sigma
    int                                       ID(0);
    int* h_permutation = (int*)calloc(nrow, sizeof(int));
    //int* h_indice = (int*)malloc(sizeof(int) * nnz);   //申请新的indice
    int h_indice[_width1] = {0};   //申请新的indice
    int start(0), curr(0), idx(0);

    //step 3：生成入度向量并排序
    thrust::device_vector<int>     vec_deg(nrow + 1);
    thrust::device_ptr<int>        ptr_offset(offset);
    thrust::adjacent_difference(ptr_offset, ptr_offset + nrow, vec_deg.begin());
    vec_deg.erase(vec_deg.begin());
    //auto ptr = vec_deg.data();

    //thrust::device_vector<int>     vec_idx(nrow);
    //thrust::sequence(vec_idx.begin(), vec_idx.end());
    
    //thrust::sort(vec_idx.begin(), vec_idx.end(), [ptr]__device__(int _a, int _b){ return ptr[_a] > ptr[_b]; });
    
    //std::vector<int> h_vec_idx(nrow);
    //thrust::copy(vec_idx.begin(), vec_idx.end(), h_vec_idx.begin());
    
    ID = thrust::max_element(vec_deg.begin(), vec_deg.end()) - vec_deg.begin();

    //long long inc = 0;
    //long long des = 0;
    
    //step 4：创建节点并生成堆
    for(int i=0; i<nrow; ++i)
    {
        NODE[i] = std::make_unique<Node>(i);
        node = NODE[i].get();
        PQUE.push(node);
    }

    node = NODE[ID].get();
    PQUE.modify(node, 1);

    //step 5：开始循环
    ID = 0;
    //int  cnt_y = 1;
    while(PQUE.CNT>0)
    {
        //出队列
        PQUE.pop(node);
        node->STS = false;
        h_permutation[ID] = node->ID;
        
        int deg = h_offset2[node->ID + 1] - h_offset2[node->ID];
        if (curr - start + deg < _width1)//没超过这个tile
        {
            for(int i=0; i<deg; ++i)
            {
                //将源顶点拉入h_indice
                idx = h_indice2[h_offset2[node->ID] + i];
                h_indice[curr - start] = idx;
                
                //更新对应的目标权值
                if (h_offset1[idx + 1] - h_offset1[idx] < _width2)
                {
                    for (int j=h_offset1[idx]; j<h_offset1[idx + 1]; ++j)
                    {
                        tmp = NODE[h_indice1[j]].get();
                        if(tmp->STS)
                        {
                            PQUE.modify(tmp, 1);
                            //++inc;
                            //if (tmp->ID == 4007) std::cout << "++ node = " << j << ", key1 = " << tmp->key1 << ", key2 = " << tmp->key2 << std::endl;
                        }
                    }
                }
                
                ++curr;
            }
        }
        else //跨tile了
        {
            //原tile贬值
            for(int i=start; i<curr; ++i)
            {
                idx = h_indice[i - start];
                if (h_offset1[idx + 1] - h_offset1[idx] < _width2)
                {
                    for (int j=h_offset1[idx]; j<h_offset1[idx + 1]; ++j)
                    {
                        tmp = NODE[h_indice1[j]].get();
                        if(tmp->STS)
                        {
                            //PQUE.modify(tmp, -1);
                            PQUE.refresh(tmp);
                            //++des;
                            //if (tmp->ID == 4007) std::cout << "-- node = " << j << ", key1 = " << tmp->key1 << ", key2 = " << tmp->key2 << std::endl;
                        }
                    }
                }
            }
            PQUE.PMAX = 0;
            
            //更新start
            start = (curr + deg) / _width1 * _width1;
            //std::cout << "start = " << start << std::endl; 
            
            for (int i=0; i<deg; ++i)
            {
                if(curr >= start)
                {
                    //将源顶点拉入h_indice
                    idx = h_indice2[h_offset2[node->ID] + i];
                    h_indice[curr - start] = idx;
                    
                    //更新对应的目标权值
                    if (h_offset1[idx + 1] - h_offset1[idx] < _width2)
                    {
                        for (int j=h_offset1[idx]; j<h_offset1[idx + 1]; ++j)
                        {
                            tmp = NODE[h_indice1[j]].get();
                            if(tmp->STS)
                            {
                                PQUE.modify(tmp, 1);
                                //++inc;
                                //if (tmp->ID == 4007) std::cout << "+++ node = " << j << ", key1 = " << tmp->key1 << ", key2 = " << tmp->key2 << std::endl;
                            }
                        }

                    }
                }

                ++curr;
            }
        }
        
        ++ID;
        //std::cout << "ID = " << ID << std::endl;
    }
    //std::cout << "inc = " << inc / (float)nrow << ", des = " << des / (float)nrow << std::endl;
    
    //step 6：生成用于排序的辅助数组
    int* d_permutation;
    cudaMalloc((void**)&d_permutation, sizeof(int) * nrow);
    cudaMemcpy(d_permutation, h_permutation, sizeof(int) * nrow, cudaMemcpyHostToDevice);
    thrust::device_ptr<int> ptr_permutation(d_permutation);
    //std::cout << "d_permutation: ";
    //thrust::copy(ptr_permutation, ptr_permutation + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 7：创建新的列索引映射
    thrust::device_ptr<int> ptr_perm(perm);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrow), ptr_permutation, ptr_perm);
    cudaDeviceSynchronize();
    timer.Stop();
    tim1 = timer.Millisecs();
    //std::cout << "ptr_map: ";
    //thrust::copy(ptr_map, ptr_map + nrow, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << std::endl;

    //step 8：更新COO格式的列索引
    timer.Start();
    thrust::device_ptr<int>  ptr_row(row_ind);
    thrust::device_ptr<int>  ptr_col(col_ind);
    thrust::gather(ptr_row, ptr_row + nnz, ptr_perm, ptr_row);
    thrust::gather(ptr_col, ptr_col + nnz, ptr_perm, ptr_col);
    sort(false);
    cudaDeviceSynchronize();
    timer.Stop();
    tim2 = timer.Millisecs();
    FSTR << "&order&" << _width1 << "&" << _width2 << "&" << tim1 << "&" << tim2;
    
    //step 9：回收内存
    FREE(NODE);
    FREE(h_offset1);
    FREE(h_offset2);
    FREE(h_indice1);
    FREE(h_indice2);
    FREE(h_permutation);
    CUDAFREE(offset);
    CUDAFREE(d_permutation);
}

template<typename ValueType> GRAPH<ValueType>::~GRAPH()
{
    FREE(h_row_ind); 
    FREE(h_col_ind); 
    FREE(h_values);
    CUDAFREE(row_ind); 
    CUDAFREE(col_ind); 
    CUDAFREE(values);
    CUDAFREE(perm);
    CUDAFREE(y);
}

template struct BASEGRAPH<float>;
template struct BASEGRAPH<double>;

template struct GRAPH<float>;
template struct GRAPH<double>;