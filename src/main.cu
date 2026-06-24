#include "../include/fun.cuh"
std::fstream      FSTR;
int main(int argc, char* argv[])
{  
    //变量赋值
    const int device_id = 1;
    const std::string config = "/home/ta/zhangq388/RELABEL/config/config2.txt";
    const std::string log = "/home/ta/zhangq388/RELABEL/log.txt";
    FSTR.open(log, std::ios_base::app);
    int gid = std::stoi(argv[1]);
    std::string type = argv[2];
    Timer   timer;
    float   tim1(0), tim2(0);
    float   speedup(0.0);
    
    //环境
    ENV env(device_id, config, log);
    
    //预热
    format_warmup<int>();

    using ValueType = float;
    
    //读取图
    BASEGRAPH<ValueType> basegraph(env, gid);
    
    //PARA  para[9] = {{0.10, 64, 256}, {0.10, 256, 1024}, {0.20, 1024, 4096}, 
    //                 {0.30, 4096, 16384}, {0.30, 16384, 65536}, {0.40, 65536, 262144}, 
    //                 {0.50, 262144, 1048576}, {0.70, 1048576, 4194304}, {0.90, 0, 10000000}};
    //for (int i=0; i<9; ++i)
    //{
        //GRAPH<ValueType> graph(env, basegraph, para[i].rho, para[i].near, para[i].window);
        GRAPH<ValueType> graph(env, basegraph, 0, 0, 0);

        //graph.desc("/home/ta/zhangq388/RELABEL/desc1.txt");
        //graph.show();
        //graph.visual("/home/ta/zhangq388/RELABEL/matrix.jpg");
        //graph.jaccard(env, 1, 2);
        
        //std::cout << "graph_id = " << graph.graph_id << ", rho = " << para[i].rho << ", near = " << para[i].near << ", window = " << para[i].window << ", type = " << type;
        //FSTR << graph.graph_id << "_" << para[i].rho << "_" << para[i].near << "_" << para[i].window;
        std::cout << "graph_id = " << graph.graph_id << ", type = " << type;
        FSTR << graph.graph_id;
        
        if (type == "origin")
        {
            FSTR << "&origin&&&";
            timer.Start();
            graph.locality_predict(env, speedup);
            timer.Stop();
            tim1 = timer.Millisecs();
            std::cout << ", tim1 = " << tim1;
        }
        else if (type == "random") //random
        {
            graph.relabel_random();
            graph.locality_predict(env, speedup);
        }
        else if (type == "degree") //degree
        {
            graph.relabel_degree();
            graph.locality_predict(env, speedup);
            //graph.show();
        }
        else if (type == "dbg") //dbg
        {
            graph.dbg(env);
            graph.locality_predict(env, speedup);
            //graph.show();
        }
        else if (type == "rcm")
        {
            graph.relabel_rcm();
            graph.locality_predict(env, speedup);
            //graph.show();
            //graph.visual("/home/ta/zhangq388/RELABEL/matrix_rcm.jpg");
        }
        else if (type == "rabbitorder")
        {
            graph.rabbitorder();
            graph.locality_predict(env, speedup);
            //graph.show();
            //graph.visual("./matrix1.jpg");
        }
        else if (type == "gorder1")
        {
            graph.gorder1(env, 5);
            graph.locality_predict(env, speedup);
            //graph.show();
            //graph.visual("./matrix1.jpg");
        }
        else if (type == "gorder2")
        {   
            //graph.gorder2(env, std::stoi(argv[3]), 1);
            graph.gorder2(env, graph.get_width(), 1);
            graph.locality_predict(env, speedup);
            //graph.show();
            //graph.visual("./matrix2.jpg");
        }
        else if (type == "gorder3")
        {
            TRYCATCH(graph.gorder3(env, std::stoi(argv[3]), 1, 5));
            //graph.show();
            //graph.visual("./matrix3.jpg");
        }
        else if (type == "gorder4")
        {
            //graph.gorder4(env, std::stoi(argv[3]), 1);
            graph.gorder4(env, graph.get_width(), 5);
            graph.locality_predict(env, speedup);
            //graph.show();
            //graph.visual("./matrix3.jpg");
        }
        else if (type == "gorder5")
        {
            //graph.gorder5(env, std::stoi(argv[3]), 1, 5);
            graph.gorder5(env, graph.get_width(), 1, 5);
            graph.locality_predict(env, speedup);
            //graph.show();
            //graph.visual("./matrix3.jpg");
        }
        else if (type == "gorder6")
        {
            //graph.gorder6(env, std::stoi(argv[3]));
            graph.locality_predict(env, speedup);
            graph.gorder6(env, graph.get_width());
            //graph.show();
            //graph.visual("./matrix3.jpg");
        }
        else if (type == "order")
        {
            graph.order(env, std::stoi(argv[3]), std::stoi(argv[4]));
            //graph.order(env, 1024, 512);
        }
        
        //graph.relabel_slashburn(env); //不用这种方法对比：(1)该算法目标为压缩；(2)代码构造图时nrow=max(el)，对不上；
        //graph.show();

        //spmv测试
        graph.spmv(env, "merbit");
    //}
    
    FSTR.close();

    return 0;
}