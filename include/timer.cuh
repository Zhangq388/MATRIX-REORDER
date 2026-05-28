//时间
//计时要用cpu计时
#ifndef __TIMER__
#define __TIMER__
#include <chrono>
class Timer
{
    public:
      Timer() {}
      void Start() 
      {
        elapsed_time = start_time = std::chrono::high_resolution_clock::now();
      }
      void Stop() 
      {
        elapsed_time = std::chrono::high_resolution_clock::now();
      }
      double Seconds() const 
      {
        return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_time - start_time).count();
      }
      double Millisecs() const 
      {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(elapsed_time - start_time).count();
      }
      double Microsecs() const 
      {
        return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(elapsed_time - start_time).count();
      }
    private:
      std::chrono::high_resolution_clock::time_point start_time, elapsed_time;
};
#endif