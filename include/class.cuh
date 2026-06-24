#ifndef __CLASS__
#define __CLASS__
#include "utils.cuh"
#include "timer.cuh"
#include "ThreadPool.h"
#include "integer_heap.cuh"
#include "model.cuh"
#include "SlashBurn.h"
#include "rcm.h"
#include "rabbit_order.h"
#include "merbit.cuh"
//环境
struct ENV
{   
    const int            device_id = 0;
    const std::string    config; //配置文件
    const std::string    log;    //结果文件
    cudaStream_t         stream;
    cusparseHandle_t     handle_sparse;
    cublasHandle_t       handle_cublas;
    ENV(const int id, const std::string& config, const std::string log): device_id(id), config(config), log(log)
    {
        cudaSetDevice(1);
        cudaStreamCreate(&stream);
        cusparseCreate(&handle_sparse);
        cublasCreate_v2(&handle_cublas);
        cusparseSetStream(handle_sparse, stream);
        cublasSetStream_v2(handle_cublas, stream);
    };
    ~ENV()
    {
        cudaStreamDestroy(stream);
        cusparseDestroy(handle_sparse);
        cublasDestroy_v2(handle_cublas);
    };
};

template<typename ValueType> struct CudaDataType;
template<> struct CudaDataType<half>{ static const cudaDataType value = cudaDataType::CUDA_R_16F;};
template<> struct CudaDataType<float>{ static const cudaDataType value = cudaDataType::CUDA_R_32F;};
template<> struct CudaDataType<double>{ static const cudaDataType value = cudaDataType::CUDA_R_64F;};

//基图
template<typename ValueType>
struct BASEGRAPH
{
    int             graph_id = -1;
    int             nrow = 0;
    int             nnz = 0;
    int*            row_ind = nullptr;
    int*            col_ind = nullptr;
    ValueType*      values = nullptr;
    BASEGRAPH() {}; //默认构造函数
    BASEGRAPH(const ENV& _env, const int _gid); //构造函数1
    ~BASEGRAPH(); //析构函数
};

//图
template<typename ValueType> 
struct GRAPH
{
    int             graph_id = -1;
    int             nrow = 0;
    int             nnz = 0;
    int*            h_row_ind = nullptr;
    int*            h_col_ind = nullptr;
    ValueType*      h_values  = nullptr;
    int*            row_ind = nullptr;
    int*            col_ind = nullptr;
    ValueType*      values = nullptr;
    int*            perm = nullptr;
    ValueType*      y = nullptr;
    GRAPH() {}; //默认构造函数
    GRAPH(const ENV& _env, const int _gid); //构造函数1：从指定的文件读入数据
    GRAPH(const int _graph_id, const int _nrow, const int _nnz, const unsigned int* _row_ind, const unsigned int* _col_ind); //构造函数2：根据传入的向量生成graph
    GRAPH(const ENV& _env, const BASEGRAPH<ValueType>& _basegraph, const float _rho, const int _near, const int _window); //构造函数3：根据basegraph做可控变换
    void sort(const bool dir);
    void desc(const std::string& file);
    int get_width();
    void show(); //打印矩阵
    void visual(std::string _filename);
    int  gscore(const ENV& _env, const int _width);
    void locality_predict(const ENV& _env, float& speedup);
    void spmv(const ENV& _env, std::string _method); 
    void relabel_random();
    void relabel_degree();
    void dbg(const ENV& _env); //DBG
    void relabel_rcm(); 
    void rabbitorder();
    void relabel_slashburn(const ENV& _env);
    void gorder1(const ENV& _env, const int _width);  //GOrder  origin
    void gorder2(const ENV& _env, const int _vsize, const int _width); //GOrder based on GPU
    void gorder3(const ENV& _env, const int _vsize, const int _width1, const int _width2); //GPU_CPU_Hybrid
    void gorder4(const ENV& _env, const int _vsize, const int _width); //CPU并行
    void gorder5(const ENV& _env, const int _vsize, const int _width1, const int _width2); //GPU_CPU_Hybrid
    void gorder6(const ENV& _env, const int _width); //初始化32
    void order(const ENV& _env, const int _width1, const int _width2);
    ~GRAPH(); //析构函数
};

//Sliding Window
template <typename T, typename Less = std::less<T>>
struct WINDOW
{
    unsigned int    maxsize = 0;
    std::vector<T>  data;
    Less            less;
    WINDOW(){};
    explicit WINDOW(unsigned int _maxsize, Less _cmp = Less{}): maxsize(_maxsize), less(std::move(_cmp))
    {
        data.reserve(maxsize);
    };
    void replace(T& _in)
    {
        if (data.size() < maxsize)
        {
            data.push_back(_in);
        }
        else if (!data.empty())
        {
            auto it = std::min_element(data.begin(), data.end(), less);
            std::swap(*it, _in);
        }
    };
    ~WINDOW(){}; 
};

struct PARA
{
    float      rho;
    int        near;
    int        window;
};
#endif