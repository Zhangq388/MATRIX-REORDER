#include "SlashBurn.h"
void print_sep() 
{
  for (int i = 0; i < 74; ++i) 
  {
    std::cout << "-";
  }
  std::cout << "\n";
}

void info(uint32_t n_threads, uint32_t k, float p) 
{
  print_sep();
  fmt::print("Parallel SlashBurn Parameters:\n");
  print_sep();
  fmt::print("| Number of Threads: {:^14}| p: {:^14}| k: {:^14} |\n", n_threads, p, k);
  print_sep();
}

int main(int argc, char *argv[]) 
{
  CLApp cli(argc, argv, "slashburn");
  if (!cli.ParseArgs())
    return -1;
  Builder   b(cli);
  Graph g = b.MakeGraph();
  int n_neighbour_rounds = 2;
  float percent = cli.percent();
  Bitmap bmap(g.num_nodes());
  uint32_t num_nodes = g.num_nodes();
  uint num_threads = cli.num_threads();
  uint32_t k = num_nodes * percent;
  fmt::print("{} vertices\n", num_nodes);
  fmt::print("{} undirected edges\n", g.num_edges());
  info(num_threads, k, percent);

  SlashBurn sb = SlashBurn(g, n_neighbour_rounds, percent, bmap, num_threads);
  sb.write_permutation(cli.out_filename());
   
  fmt::print("{}ms to complete {} iterations.\n", sb.time, sb.n_iters);
  return 0;
}
