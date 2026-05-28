#include "SlashBurn.h"

/**
 * @brief Construct a new SlashBurn object
 * 
 * @param g a gapbs graph
 * @param n_neighs number of neighbour rounds, a parameter of Afforest
 * @param p 0.005 by default, repeat SB algorithm until the size of GCC is k = 0.005n 
 * @param bitmap bit i is set if vertex i has been deleted
 * @param nt number of threads
 */
SlashBurn::SlashBurn(Graph& g, int n_neighs, float p, Bitmap& bitmap, uint nt) : g(g), bmap(bitmap) 
{
  prec = p;
  n_neighbour_rounds = n_neighs;
  num_nodes = g.num_nodes();
  num_threads = nt;
  gcc_size = num_nodes;
  k = num_nodes * prec;
  k = (k == 0)? 1 : k;

  init();
  auto start = std::chrono::high_resolution_clock::now();

  n_iters = 0;
  while (true) 
  {
    sort_by_degrees();

    // if, after sorting, the highest degree vertex is a singleton, the graph is completely fragmented, so break
    if (degrees[vids[0]] == 0) 
    {
      break;
    }

    remove_k_hubs();
    compute_decrs(top_k);
    if (verif) 
    {
      verif_decrs(top_k);
    }
    par_decr();

    // compute connected components
    Afforest();
    if (active_vertex_set_size == 0) 
    {
      break;
    }

    compute_cc_sizes();
    if (spokes_end == hub_start) 
    {
      break;
    }

    // if, after k-hub removal, there exist a single connected component, continue
    if (n_components == 1) 
    {
      clear();
      n_iters++;
      continue;
    }

    compute_spoke_offsets(false);
    n_iters++;
    clear();

    //#if TIME == 1
    //  // incr_write_perm();
    //  auto iter_end = std::chrono::high_resolution_clock::now();
    //  auto slashburn_time = std::chrono::duration_cast<time_unit>(iter_end - iter_start);
    //  uint64_t iter_time = (uint64_t) slashburn_time.count();
    //  iter_times.push_back({n_iters, iter_time});
    //  n_active_vertices.push_back(active_vertex_set_size);
    //#endif
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto slashburn_time = std::chrono::duration_cast<time_unit>(end - start);
  time = (uint64_t) slashburn_time.count();

  // clean up
  free(cc_sizes);
  free(decrs);
  delete[] cc_size_structs;
}

/**
 * @brief Reset the component size and degree decrement vector
 * 
 */
void SlashBurn::clear() 
{
  #pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; ++i) 
  {
    cc_size_structs[i].size = 0;
    cc_size_structs[i].id = i;
  }

  #pragma omp parallel for
  for (uint32_t i = 0; i < block_reduction_totalsize; ++i) 
  {
    cc_sizes[i] = 0;
    decrs[i] = 0;
  }
}

/**
 * @brief 
 * 
 * @param final_iter true if this is the final iteration of the SlashBurn algo
 */
void SlashBurn::compute_spoke_offsets(bool final_iter) 
{
  // sort all spokes by their size to compute their global offset into perm
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> spokes(num_threads);
  std::vector<Spokes> spokes_vs(num_threads);
  uint32_t n_vertices_in_spokes = 0;

  // compute the number of vertices per spoke and the total number of vertices in all spokes
  #pragma omp parallel num_threads(num_threads) reduction(+ : n_vertices_in_spokes)
  {
    uint tid = omp_get_thread_num();
    #pragma omp for
    for (uint32_t i = 0; i < num_nodes; ++i) 
    {
      uint32_t cid = comp[i];
      if (bmap.get_bit(i)) continue; 
      if (!final_iter && cid == gcc_id) continue; 
      if (cc_sizes[i] > 0) 
      {
        spokes[tid].push_back({i, cc_sizes[i]});
        n_vertices_in_spokes += cc_sizes[i];
      }
    }
  }

  // flatten s
  pvector<uint32_t> spoke_vec(n_vertices_in_spokes);
  std::vector<std::pair<uint32_t, uint32_t>> flat = flatten<std::pair<uint32_t, uint32_t>>(spokes);
  ips4o::parallel::sort(flat.begin(), flat.end(), [](std::pair<uint32_t, uint32_t> c1, std::pair<uint32_t, uint32_t> c2) -> bool {if (c1.second == c2.second) {return c1.first < c2.first;}  return c1.second > c2.second;});

  //compute offsets into spokes vector
  std::vector<uint32_t> prefix_offset(flat.size() + 1);

  // compute prefix sum of spoke sizes
  uint32_t accum = 0;
  if (flat.size() < num_threads) 
  {
    for (uint32_t i = 0; i < flat.size(); ++i) 
    {
      accum += flat[i].second;
      prefix_offset[i + 1] = accum;
    }
  } 
  else 
  {
    #pragma omp parallel for reduction(inscan, + : accum) num_threads(num_threads) shared(flat)
    for (uint32_t i = 0; i < flat.size(); ++i) 
    {
      accum += flat[i].second;
      #pragma omp scan inclusive(accum)
      prefix_offset[i + 1] = accum;
    }
  }

  #pragma omp parallel for
  for (uint32_t i = 0; i < flat.size(); ++i) 
  {
    cc_size_structs[flat[i].first].id = flat[i].first;
    cc_size_structs[flat[i].first].size = prefix_offset[i];
  }

  #pragma omp parallel num_threads(num_threads)
  {
    uint tid = omp_get_thread_num();
    Spokes &cs = spokes_vs[tid];
    #pragma omp for
    for (uint32_t i = 0; i < num_nodes; ++i) 
    {
      uint32_t cid = comp[i];
      if (bmap.get_bit(i)) { continue; }
      if (!final_iter && cid == gcc_id) { continue; }
      if (cs.find(cid) == cs.end()) 
      {
        cs.insert({cid, {i}});
      } 
      else 
      {
        cs[cid].insert(i);
      }
    }
  }
  absl::flat_hash_map<uint32_t, std::vector<uint32_t>> tprivate_offsets;
  // spokes may be split across different threads
  // compute the per-thread offsets for split spokes
  compute_offsets_for_split_spokes(spokes_vs, tprivate_offsets);

  // populate perm array with spokes
  #pragma omp parallel num_threads(num_threads)
  {
    uint tid = omp_get_thread_num();
    Spokes &cs = spokes_vs[tid];
    for (auto &kv: cs) 
    {
      uint32_t comp_id = kv.first;
      absl::flat_hash_set<uint32_t> &spoke_vs = kv.second;
      bool exclusive = false;
      if (cc_sizes[comp_id] == kv.second.size()) 
      {
        // all the vertices of this spoke are owned by this thread
        exclusive = true;
      }

      if (exclusive) 
      {
        uint32_t i = cc_size_structs[comp_id].size;
        for (auto &spoke: spoke_vs) 
        {
          spoke_vec[i] = spoke;
          bmap.set_bit_atomic(spoke);
          i++;
        }
      } 
      else 
      {
        // use the thread private offsets to write to the spoke vector
        uint32_t offset = tprivate_offsets[comp_id][tid];
        for (auto &spoke: spoke_vs) 
        {
          spoke_vec[offset] = spoke;
          bmap.set_bit_atomic(spoke);
          offset++;
        }
      }
    }
  }
  // decrement degrees based on spokes
  compute_decrs(spoke_vec);
  if (verif) verif_decrs(spoke_vec);
  par_decr();

  #pragma omp parallel for
  for (uint32_t i = 0; i < n_vertices_in_spokes; ++i) 
  {
    perm[i + spokes_end - n_vertices_in_spokes] = spoke_vec[i];
  }

  spokes_end -= n_vertices_in_spokes;
  active_vertex_set_size -= n_vertices_in_spokes;
}

/**
 * @brief after computing the spokes in parallel the vertices within each spoke 
 * may be split across threads
 * in this case, compute the per-thread offset ranges into the spoke vector 
 * 
 * @param spokes 
 */
void SlashBurn::compute_offsets_for_split_spokes(std::vector<Spokes>& t_spokes, absl::flat_hash_map<uint32_t, std::vector<uint32_t>> &tprivate_offsets) 
{
  for (uint t = 0; t < num_threads; ++t) 
  {
    Spokes& ss = t_spokes[t];
    for (auto &kv: ss) 
    {
      auto cid = kv.first;
      auto &spoke = kv.second;
      if (spoke.size() != cc_sizes[cid]) 
      {
        tprivate_offsets[cid].resize(num_threads + 1);
      }
    }
  }

  #pragma omp parallel num_threads(num_threads)
  {
    auto tid = omp_get_thread_num();
    Spokes &ss = t_spokes[tid];

    for (auto &kv: tprivate_offsets) 
    {
      auto cid = kv.first;
      if (ss.find(cid) == ss.end()) 
      {
        // not found
      } 
      else 
      {
        // found
        tprivate_offsets[cid][tid + 1] = ss[cid].size();
      }
    }
  }
  // compute prefix sum for thread private offsets
  // todo this is done sequentially - worth to parallelize?
  for (auto &kv: tprivate_offsets) 
  {
    auto cid = kv.first;
    uint32_t accum = cc_size_structs[cid].size;
    for (uint i = 0; i < num_threads + 1; ++i) 
    {
      accum += tprivate_offsets[cid][i];
      tprivate_offsets[cid][i] = accum;
    }
  }
}

/**
 * @brief 
 * Verify that the connected component sizes that were calculated using the 
 * spray reduction are correct
 * Using a single thread, iterate over the comp array, to get the sizes of 
 * each connected component
 * 
 * @return uint32_t the size of the largest connected component
 */
uint32_t SlashBurn::verif_cc_sizes() 
{
  pvector<uint32_t> gt_cc_sizes(num_nodes, 0);
  for (uint32_t i = 0; i < num_nodes; ++i) 
  {
    if (bmap.get_bit(i)) continue;
    gt_cc_sizes[comp[i]]++;
  }

  uint32_t max_size = 0;
  for (uint32_t i = 0; i < num_nodes; ++i) 
  {
    if (bmap.get_bit(i)) continue;
    if (gt_cc_sizes[i] != cc_sizes[i]) 
    {
      fmt::print("Invalid Component Size: {} {} {}\n", i, gt_cc_sizes[i], cc_sizes[i]);
    }
    if (gt_cc_sizes[i] > max_size)
    {
      max_size = gt_cc_sizes[i];
    }
  }
  return max_size;
}

/**
 * @brief Compute the sizes of connected components using map-reduce
 * Split component ID assignment vector between threads
 * Each thread maintains a thread-private hash map and computes number of 
 * vertices in each component
 * Merge thread-private maps atomically into a global component size vector
 * 
 */
void SlashBurn::map_reduce_cc_sizes() 
{
  std::vector<absl::flat_hash_map<uint32_t, uint32_t>> counts(num_threads);

  #pragma omp parallel num_threads(num_threads)
  {
    uint tid = omp_get_thread_num();
    auto &t_counts = counts[tid];
    #pragma omp for
    for (uint64_t i = 0; i < num_nodes; ++i) 
    {
      if (bmap.get_bit(i)) { continue; }
      auto cid = comp[i];
      if (t_counts.find(cid) == t_counts.end()) 
      {
        // not found
        t_counts.insert({cid, 1});
      } 
      else 
      {
        // found
        t_counts[cid]++;
        // tprivate_offsets[cid][tid + 1] = ss[cid].size();
      }
    }
    // atomically update the component sizes array
    for (const auto &kv: t_counts) 
    {
      auto cid = kv.first;
      auto n_vs_in_cid = kv.second;
      #pragma omp atomic update
      cc_sizes[cid] += n_vs_in_cid;
    }
  }
}

/**
 * @brief Use sparse array block reduction to calculate the size of each 
 * connected component in the graph
 * the comp array contains component assignment of all the (active) vertices
 * in the graph
 * in parallel, iterate over the comp array, and accumulate component id 
 * assignments to get the sizes of each vector
 * (optionally, verify the correctness of the reduction by comparing sizes
 * with a ground-truth vector, and the size of the gcc 
 * (giant-connected-component))
 * 
 */
void SlashBurn::compute_cc_sizes() 
{
  // todo block reductions produce incorrect sums
  //   BlockReductionBWIDTH<uint32_t> cc_sizes_red(block_reduction_totalsize, cc_sizes);
  // #pragma omp parallel for schedule(static) reduction(+ : cc_sizes_red) \
  //         num_threads(num_threads) shared(comp, bmap)
  //   for (uint32_t i = 0; i < num_nodes; ++i) {
  //     uint32_t cid = comp[i];
  //     if (!bmap.get_bit(i)) {
  //       cc_sizes_red[cid]++;
  //     }
  //   }
  map_reduce_cc_sizes();

  n_components = 0;
  #pragma omp parallel for reduction(+ : n_components)
  for (uint32_t i = 0; i < num_nodes; ++i) 
  {
    if (bmap.get_bit(i)) continue; 
    if (cc_sizes[i] > 0) 
    {
      n_components++;
      cc_size_structs[i].size = cc_sizes[i];
      cc_size_structs[i].id = i;
    }
  }

  auto mx = get_max_cc_reduction();
  gcc_id = mx.id;
  gcc_size = mx.size;
  //print_progress();
  if (verif) verif_cc_sizes();

  if (gcc_size <= k) 
  {
    compute_spoke_offsets(true);
  }
}

/**
 * @brief Use a user defined reduction to get the size of the largest 
 * connected component
 * 
 * (from https://stackoverflow.com/a/67507849)
 * @return SlashBurn::cc_size_t 
 */
SlashBurn::cc_size_t SlashBurn::get_max_cc_reduction() {
  cc_size_t best = {0, 0};

#pragma omp declare reduction(get_max:cc_size_t : omp_out = omp_in.size > omp_out.size ? omp_in : omp_out) initializer(omp_priv = (omp_orig))
#pragma omp parallel for reduction(get_max : best)
  for (uint32_t i = 0; i < num_nodes; ++i)// for all ants
  {
    if (bmap.get_bit(i)) { continue; }
    if (cc_size_structs[i].size > best.size) {
      best.size = cc_size_structs[i].size;
      best.id = cc_size_structs[i].id;
    }
  }
  return best;
}


/**
 * @brief Place nodes u and v in same component of lower component ID
 * 
 * @param u 
 * @param v 
 */
void SlashBurn::Link(NID u, NID v) 
{
  NID p1 = comp[u];
  NID p2 = comp[v];
  while (p1 != p2) 
  {
    NID high = p1 > p2 ? p1 : p2;
    NID low = p1 + (p2 - high);
    NID p_high = comp[high];
    // Was already 'low' or succeeded in writing 'low'
    if ((p_high == low) ||
        (p_high == high && compare_and_swap(comp[high], high, low)))
      break;
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}

/**
 * @brief Reduce depth of tree for each component to 1 by crawling up parents
 * 
 */
void SlashBurn::Compress() 
{
  #pragma omp parallel for schedule(dynamic, 16384)
  for (NID n = 0; n < g.num_nodes(); n++) 
  {
    if (bmap.get_bit(n)) 
    { 
      continue; 
    }

    while (comp[n] != comp[comp[n]]) 
    {
      comp[n] = comp[comp[n]];
    }
  }
}

/**
 * @brief iterate (in parallel) over the active vertex bitmap
 * each thread will iterate over its assigned section of the bitmap
 * for each active vertex seen, it will store that vertex's id in a local array
 * once all active vertices have been seen, merge them into a global array
 */
void SlashBurn::get_active_vertex_set() 
{

  pvector<pvector<uint32_t>> all_vs(num_threads);
  uint32_t max_results_per_thread = num_nodes / num_threads + 1;
  uint32_t n_active_vertices = 0;
  // once all active vertices have been identified, the start, end points for
  // each thread need to be stored (they may not all be equal)
  pvector<uint32_t> bounds(num_threads);
  #pragma omp parallel num_threads(num_threads)
  {
    int t = omp_get_thread_num();
    pvector<uint32_t> &local_newq = all_vs[t];
    local_newq.resize(max_results_per_thread);
    uint32_t local_newq_count = 0;
    uint32_t start_bm_idx = t * max_results_per_thread;
    uint32_t end_bm_idx = (t + 1) * max_results_per_thread;
    end_bm_idx = (end_bm_idx > num_nodes) ? num_nodes : end_bm_idx;
    for (uint32_t i = start_bm_idx; i < end_bm_idx; ++i) 
    {
      if (!vertex_inactive(i)) 
      {
        local_newq[local_newq_count] = i;
        ++local_newq_count;
      }
    }
    #pragma omp atomic update
    n_active_vertices += local_newq_count;
    bounds[t] = local_newq_count;
  }

  pvector<uint32_t> cumulative_bounds(bounds.size());
  std::exclusive_scan(
          bounds.begin(),
          bounds.end(),
          cumulative_bounds.begin(),
          0,
          [](uint32_t l, uint32_t r) -> uint32_t { return l + r; });
  active_vertex_set.resize(n_active_vertices);
  #pragma omp parallel num_threads(num_threads)
  {
    int t = omp_get_thread_num();
    uint32_t local_mx = bounds[t];
    uint32_t start = cumulative_bounds[t];
    for (uint32_t i = 0; i < local_mx; ++i) {
      active_vertex_set[start] = all_vs[t][i];
      ++start;
    }
  }
}

/**
 * @brief Estimate the largest component ID by sampling component IDs from 
 * the connected component ID vector
 * 
 * @param num_samples 
 * @return NID 
 */
NID SlashBurn::SampleFrequentElement(uint32_t num_samples = 1024) {
  std::unordered_map<NID, int> sample_counts(32);
  using kvp_type = std::unordered_map<NID, int>::value_type;
  // Sample elements from 'comp'
  // need to only sample the component ids of active vertices
  get_active_vertex_set();
  active_vertex_set_size = active_vertex_set.size();
  if (active_vertex_set_size == 0) return -1;
  std::mt19937 gen;
  std::uniform_int_distribution<NID> distribution(0, active_vertex_set.size() - 1);
  for (NID i = 0; i < num_samples; i++) {
    NID n = active_vertex_set[distribution(gen)];
    sample_counts[comp[n]]++;
  }

  // Find most frequent element in samples (estimate of most frequent overall)
  auto most_frequent = std::max_element(
          sample_counts.begin(), sample_counts.end(),
          [](const kvp_type &a, const kvp_type &b) { return a.second < b.second; });
  float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
  return most_frequent->first;
}

/**
 * @brief Afforest Connectivity Algorithm 
 * Adapted from gapbs
 * 
 */
void SlashBurn::Afforest() 
{
  // Initialize each node to a single-node self-pointing tree
  #pragma omp parallel for
  for (NID n = 0; n < g.num_nodes(); n++) 
  {
    comp[n] = n;
  }
  // Process a sparse sampled subgraph first for approximating components.
  // Sample by processing a fixed number of neighbors for each node (see paper)
  for (int r = 0; r < n_neighbour_rounds; ++r) 
  {
    #pragma omp parallel for schedule(dynamic, 16384)
    for (NID u = 0; u < g.num_nodes(); u++) 
    {
      if (bmap.get_bit(u)) continue;
      for (NID v: g.out_neigh(u, r)) 
      {
        if (bmap.get_bit(v)) continue;
        // Link at most one time if neighbor available at offset r
        Link(u, v);
        break;
      }
    }
    Compress();
  }

  // Sample 'comp' to find the most frequent element -- due to prior
  // compression, this value represents the largest intermediate component
  NID c = SampleFrequentElement(1024);
  if (c == -1) return;

  // Final 'link' phase over remaining edges (excluding largest component)
  if (!g.directed()) 
  {
    #pragma omp parallel for schedule(dynamic, 16384)
    for (NID u = 0; u < g.num_nodes(); u++) 
    {
      // skip vertices already assigned
      if (bmap.get_bit(u)) continue; 
      // Skip processing nodes in the largest component
      if (comp[u] == c) continue; 
      // Skip over part of neighborhood (determined by neighbor_rounds)
      for (NID v: g.out_neigh(u, n_neighbour_rounds)) 
      {
        if (bmap.get_bit(v)) continue; 
        Link(u, v);
      }
    }
  } 
  else 
  {
    #pragma omp parallel for schedule(dynamic, 16384)
    for (NID u = 0; u < g.num_nodes(); u++) 
    {
      if (bmap.get_bit(u)) continue; 
      if (comp[u] == c) continue;
      for (NID v: g.out_neigh(u, n_neighbour_rounds)) 
      {
        if (bmap.get_bit(v)) continue; 
        Link(u, v);
      }
      // To support directed graphs, process reverse graph completely
      for (NID v: g.in_neigh(u)) 
      {
        if (bmap.get_bit(v)) continue; 
        Link(u, v);
      }
    }
  }
  // Finally, 'compress' for final convergence
  Compress();

  #if TIME == 1
    auto end = std::chrono::high_resolution_clock::now();
    aff_times.push_back(std::chrono::duration_cast<time_unit>(end - start).count());
  #endif
}

/**
 * @brief Parallel Degree Decrement
 * Once the degree decrements are computed, iterate over the decrement
 * array in parallel, decrementing each vertex's degree.
 * Each thread maintains a vector of identified singletons, which are written
 * to the final permutation array. 
 * 
 */
void SlashBurn::par_decr() 
{
  std::vector<std::vector<uint32_t>> thread_private_singletons(num_threads);
  std::vector<uint32_t>              singleton_offsets(num_threads + 1);

  #pragma omp parallel num_threads(num_threads)
  {
    // init a private vector that will store singleton ids
    uint tid = omp_get_thread_num();
    std::vector<uint32_t>& singletons = thread_private_singletons[tid];
    uint32_t offset = 0;

    // overallocate thread-private singletons vector to avoid pushback
    singletons.resize(num_nodes / num_threads);

    #pragma omp for schedule(static)
    for (uint32_t i = 0; i < num_nodes; i++) 
    {
      if (decrs[i] == 0) continue;
      uint32_t decrement = decrs[i];

      // reset decrement for next iteration
      decrs[i] = 0;
      degrees[i] = (decrement > degrees[i])? 0 : (degrees[i] - decrement);

      // vertex now singleton, remove it
      if (degrees[i] == 0 && !vertex_inactive(i)) 
      {
        singletons[offset++] = i;
        bmap.set_bit_atomic(i);
      }
    }
    singleton_offsets[tid + 1] = offset;
  }

  // compute singleton offset prefix sum
  uint32_t accum = 0;
  for (uint i = 0; i < num_threads + 1; ++i) 
  {
    accum += singleton_offsets[i];
    singleton_offsets[i] = accum;
  }
  uint32_t n_singletons = singleton_offsets[num_threads];

  active_vertex_set_size -= n_singletons;
  spokes_end -= n_singletons;
  for (uint i = 0; i < num_threads + 1; ++i) 
  {
    singleton_offsets[i] += spokes_end;
  }

  // write singletons to permutation array
  #pragma omp parallel num_threads(num_threads)
  {
    uint tid = omp_get_thread_num();
    uint32_t start = singleton_offsets[tid];
    uint32_t end = singleton_offsets[tid + 1];
    for (uint32_t offset = start; offset < end; offset++) 
    {
      perm[offset] = thread_private_singletons[tid][offset - start];
    }
  }
}


/**
 * @brief Verify whether the degree decrements computed using the parallel
 * reduction are correct by sequentially computing a separate (ground-truth)
 * vector of degree decrements and comparing the two vector for equality
 * 
 * @return true 
 * @return false 
 */
bool SlashBurn::verif_decrs(pvector<uint32_t> &vertices_to_decrement) 
{
  pvector<uint32_t> gt_decrs(num_nodes, 0);

  for (uint32_t i = 0; i < vertices_to_decrement.size(); i++) 
  {
    uint32_t u = vertices_to_decrement[i];
    for (uint32_t v: g.out_neigh(u)) 
    {
      if (vertex_inactive(v)) continue;
      gt_decrs[v]++;
    }
    gt_decrs[u] = degrees[u];
  }

  for (uint32_t i = 0; i < num_nodes; ++i) 
  {
    if (gt_decrs[i] != decrs[i]) 
    {
      fmt::print("Invalid Degree Decrement: {} {} {}\n", i, gt_decrs[i], decrs[i]);
      return false;
    }
  }

  return true;
}

/**
 * Given a vector of vertices that will be removed from the graph, 
 * iterate over each vertex's neighbourhood.
 * Accumulate degree decrements across all vertices' neighbourhood.
 * Use sparse array reduction to accumulate decrements in parallel.
 * 
 * @param vertices_to_decrement 
 */
void SlashBurn::compute_decrs(pvector<uint32_t>& vertices_to_decrement) 
{
  #if TIME == 1
    auto start = std::chrono::high_resolution_clock::now();
  #endif

  BlockReductionBWIDTH<uint32_t> decr_red(block_reduction_totalsize, decrs);

  #pragma omp parallel for schedule(dynamic, 16384) reduction(+ : decr_red)
  for (uint32_t i = 0; i < vertices_to_decrement.size(); i++) 
  {
    uint32_t u = vertices_to_decrement[i];
    for (uint32_t v: g.out_neigh(u)) 
    {
      if (vertex_inactive(v)) continue;
      decr_red[v] += 1;
    }
    decr_red[u] = degrees[u];
  }

  #if TIME == 1
   auto end = std::chrono::high_resolution_clock::now();
   decr_time += std::chrono::duration_cast<time_unit>(end - start).count();
  #endif
}

/**
 * @brief Assign k-highest degree vertices to the permutation array
 * 
 */
void SlashBurn::remove_k_hubs() 
{
  uint32_t avss_decrement = 0;

  #pragma omp parallel for schedule(static) reduction(+ : avss_decrement)
  for (uint32_t i = 0; i < k; ++i) 
  {
    uint32_t hub_id = vids[i];
    top_k[i] = hub_id;
    perm[hub_start + i] = hub_id;
    bmap.set_bit_atomic(hub_id);
    ++avss_decrement;
  }

  active_vertex_set_size -= avss_decrement;
  hub_start += k;
}

/**
 * @brief Initialize all data structures and containers
 * 
 */
void SlashBurn::init() 
{
  bmap.reset();
  degrees.resize(num_nodes);
  vids.resize(num_nodes);
  top_k.resize(k);
  perm.resize(num_nodes);
  comp.resize(num_nodes);
  perm.fill(-1);
  hub_start = 0;
  spokes_end = num_nodes;
  active_vertex_set_size = num_nodes;
  block_reduction_totalsize = ((num_nodes / BWIDTH) + 1) * BWIDTH;
  uint32_t Alignment = 64;
  cc_sizes = (uint32_t *) aligned_alloc(Alignment, block_reduction_totalsize * sizeof(uint32_t));
  decrs = (uint32_t *) aligned_alloc(Alignment, block_reduction_totalsize * sizeof(uint32_t));
  cc_size_structs = new cc_size_t[num_nodes]();

  #pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; ++i) 
  {
    cc_size_structs[i].size = 0;
    cc_size_structs[i].id = i;
    vids[i] = i;
  }

  #pragma omp parallel for
  for (uint32_t i = 0; i < block_reduction_totalsize; ++i) 
  {
    cc_sizes[i] = 0;
    decrs[i] = 0;
  }

  populate_degrees();
}

/**
 * @brief Initialize an array of vertex degrees
 * 
 */
void SlashBurn::populate_degrees() 
{
  #pragma omp parallel for schedule(static)
  for (uint32_t v = 0; v < num_nodes; ++v) 
  {
    // graph is undirected, so out_degree == degree
    uint32_t d = g.out_degree(v);
    degrees[v] = d;
  }
}

/**
 * @brief Use external vertex degree array to sort array of vertex IDs
 * 
 */
void SlashBurn::sort_by_degrees() 
{
  ips4o::parallel::sort(vids.begin(), vids.end(), [&](const uint32_t &a, const uint32_t &b) { return (degrees[a] == degrees[b])? (a < b) : (degrees[a] > degrees[b]); });
}

/**
 * @brief 
 * 
 * @param vid vertex ID
 * @return true if vid has been removed from the graph
 * @return false otherwise
 */
bool SlashBurn::vertex_inactive(uint32_t vid) 
{
  return bmap.get_bit(vid);
}

/**
 * @brief Output progress of SlashBurn algorithm in user-friendly format
 * 
 */
void SlashBurn::print_progress() 
{
  fmt::print("i{:<7} | gcc: {:^14}|", n_iters, gcc_size);
  fmt::print(" avss: {:^14}| start: {:^14}| end: {:^14} | ", active_vertex_set_size, hub_start, spokes_end);
  fmt::print("n_cs: {:^14}|\n", n_components);
}


/**
 * @brief Write the final SlashBurn vertex IDs using the following format:
 * 
 * <number of nodes>
 * <number of edges>
 * 0 <vertex 0's new vertex ID>
 * 1 <vertex 1's new vertex ID>
 * ...
 * n-1 <vertex n-1's new vertex ID>
 * 
 * @param path 
 */
void SlashBurn::write_permutation(std::string path) 
{
  std::ofstream outfile(path);
  outfile << fmt::format("{}\n", num_nodes);
  outfile << fmt::format("{}\n", g.num_edges());

  // create the map and sort by original id
  pvector<std::pair<uint32_t, uint32_t>> p(num_nodes);

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < num_nodes; ++i) 
  {
    p[i] = {perm[i], i};
  }

  ips4o::parallel::sort(p.begin(), p.end());

  for (uint64_t i = 0; i < num_nodes; ++i) 
  {
    outfile << fmt::format("{} {}\n", p[i].first, p[i].second);
  }

  outfile.close();
}

/**
 * @brief Use the slashburn ordering to translate the edges of the graph
 * Write the translated edgelist to output path
 * @param path 
 */
void SlashBurn::translate_edge_list(std::string path) 
{
  std::ofstream outfile(path);

  pvector<std::pair<uint32_t, uint32_t>> el(g.num_edges());
  uint64_t k = 0;
  for (uint32_t u = 0; u < g.num_nodes(); u++) 
  {
    uint32_t src = perm[u];
    for (auto v: g.out_neigh(u)) 
    {
      uint32_t dest = perm[v];
      el[k++] = {src, dest};
    }
  }
  ips4o::parallel::sort(el.begin(), el.end());

  for (const auto &e: el) 
  {
    outfile << e.first << " " << e.second << "\n";
  }

  outfile.close();
}

/**
 * @brief Incrementally write the permutation array as the algorithm progresses
 * for visualization and debugging
 */
void SlashBurn::incr_write_perm() 
{
  std::string dir = "./scratch/incr_perm";
  std::string hubs_fname = fmt::format("{}/hubs_{}", dir, n_iters);
  std::string spokes_fname = fmt::format("{}/spokes_{}", dir, n_iters);
  std::ofstream h_outfile(hubs_fname);
  std::ofstream s_outfile(spokes_fname);

  for (uint32_t i = 0; i < hub_start; ++i) 
  {
    h_outfile << perm[i] << "\n";
  }

  for (uint32_t i = spokes_end; i < num_nodes; ++i) 
  {
    s_outfile << perm[i] << "\n";
  }

  h_outfile.close();
  s_outfile.close();
}


/**
 * @brief Use the slashburn ordering to translate the edges of the graph
 * @return relabeled graph stored in coo format 
 */
void SlashBurn::translate_coo(uint32_t* row_ind, uint32_t* col_ind) 
{
  uint64_t k = 0;
  for (uint32_t u = 0; u < g.num_nodes(); u++) 
  {
    for (auto v: g.out_neigh(u)) 
    {
      row_ind[k] = perm[u];
      col_ind[k] = perm[v];
      ++k;
    }
  }
}