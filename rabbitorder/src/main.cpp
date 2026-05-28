#include "../include/rabbit_order.h"
int main(int argc, char* argv[]) 
{
  using boost::adaptors::transformed;

  std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;

  int row_ind[11] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int col_ind[11] = {2, 2, 4, 5, 5, 2, 8, 9, 10, 8, 1};
  unsigned int permutation[12];

  std::cout << "Generating a permutation...\n";
  const double tstart = now_sec();
  rabbit_order(row_ind, col_ind, 12, 11, permutation);

  for(int i=0; i<12; ++i)
  {
    std::cout << permutation[i] << "\n";
  }

  std::cout << "Runtime for permutation generation [sec]: " << now_sec() - tstart << std::endl;

  return 0;
}