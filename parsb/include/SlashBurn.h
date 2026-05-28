#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "benchmark.h"
#include "bitmap.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "graph.h"
#include "ips4o.hpp"
#include "pvector.h"
#include "spray.hpp"
#include <omp.h>

typedef absl::flat_hash_map<uint32_t, absl::flat_hash_set<uint32_t>> Spokes;

typedef uint32_t NID;//node id
typedef std::chrono::duration<long long, std::milli> time_unit;

template<typename contentType> using BlockReductionBWIDTH = spray::AlignedBlockReduction<contentType, BWIDTH>;
#pragma omp declare reduction(+ : BlockReductionBWIDTH<uint32_t> : BlockReductionBWIDTH<uint32_t>::ompReduce(&omp_out, &omp_in)) initializer(BlockReductionBWIDTH<uint32_t>::ompInit(&omp_priv, &omp_orig))

class SlashBurn 
{
public:
  Graph&                g;
  int                   n_neighbour_rounds;
  uint32_t              k;
  float                 prec;
  bool                  verif = false;  //verify sparse array reductions
  uint32_t              num_nodes;
  pvector<uint32_t>     degrees;
  pvector<uint32_t>     vids;
  pvector<uint32_t>     top_k;
  pvector<uint32_t>     perm;
  pvector<uint32_t>     comp;
  uint64_t              time;
  uint32_t              hub_start;
  uint32_t              spokes_end;
  uint32_t              gcc_id;
  uint32_t              gcc_size;
  uint32_t              n_components;
  uint32_t              n_iters;
  std::vector<uint64_t> sort_times;
  std::vector<uint64_t> aff_times;
  std::vector<uint64_t> cc_sizes_times;
  std::vector<uint64_t> spoke_times;
  std::vector<uint64_t> decr_times;
  uint64_t              decr_time = 0;

  typedef struct 
  {
    uint32_t size;
    uint32_t id;
  } cc_size_t;

  uint32_t   block_reduction_totalsize;
  uint32_t*  cc_sizes;
  cc_size_t* cc_size_structs;
  uint32_t*  decrs;

  pvector<uint32_t> active_vertex_set;
  uint32_t          active_vertex_set_size;
  uint              num_threads;
  Bitmap&           bmap;

  SlashBurn(Graph &g, int n_neighs, float p, Bitmap& bitmap, uint nt);
  void init();
  void populate_degrees();
  void sort_by_degrees();
  void remove_k_hubs();
  void compute_decrs(pvector<uint32_t> &vertices_to_decrement);
  bool verif_decrs(pvector<uint32_t> &vertices_to_decrement);
  bool vertex_inactive(uint32_t vid);
  void par_decr();
  void Afforest();
  void Link(NID u, NID v);
  void Compress();
  NID SampleFrequentElement(uint32_t num_samples);
  void get_active_vertex_set();
  void compute_cc_sizes();
  uint32_t verif_cc_sizes();
  cc_size_t get_max_cc_reduction();
  void compute_spoke_offsets(bool final_iter);
  void compute_offsets_for_split_spokes(std::vector<Spokes> &t_spokes, absl::flat_hash_map<uint32_t, std::vector<uint32_t>> &tprivate_offsets);
  void print_progress();
  void clear();
  void write_permutation(std::string path);
  void translate_edge_list(std::string path);
  void map_reduce_cc_sizes();
  void incr_write_perm();
  void translate_coo(uint32_t* row_ind, uint32_t* col_ind); 

  /**
   * @brief Flatten a vector of vector
   * 
   * @tparam T 
   * @param orig 
   * @return std::vector<T> 
   */
  template<typename T>
  std::vector<T> flatten(const std::vector<std::vector<T>>& orig) 
  {
    std::vector<T> ret;
    for (const auto& v: orig)
    {
      ret.insert(ret.end(), v.begin(), v.end());
    }
    return ret;
  }
};