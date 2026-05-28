//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//
#include "../include/rabbit_order.h"

typedef std::pair<vint, float> edge;
typedef std::vector<std::vector<std::pair<vint, float> > > adjacency_list;

template<typename RandomAccessRange>
adjacency_list make_adj_list(const vint n, const RandomAccessRange& es) 
{
  using std::get;

  // Symmetrize the edge list and remove self-loops simultaneously
  std::vector<std::tuple<vint, vint, float>> ss(boost::size(es) * 2);
  #pragma omp parallel for
  for (size_t i = 0; i < boost::size(es); ++i) 
  {
    auto& e = es[i];
    if (get<0>(e) != get<1>(e)) 
    {
      ss[i * 2    ] = std::make_tuple(get<0>(e), get<1>(e), 1.0f);
      ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), 1.0f);
    } 
    else 
    {
      // Insert zero-weight edges instead of loops they are ignored in making an adjacency list
      ss[i * 2    ] = std::make_tuple(0, 0, 0.0f);
      ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
    }
  }

  // Sort the edges
  __gnu_parallel::sort(ss.begin(), ss.end());

  // Convert to an adjacency list
  adjacency_list adj(n);
  #pragma omp parallel
  {
    // Advance iterators to a boundary of a source vertex
    const auto adv = [](auto it, const auto first, const auto last) 
    {
      while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
        ++it;
      return it;
    };

    // Compute an iterator range assigned to this thread
    const int    p      = omp_get_max_threads();
    const size_t t      = static_cast<size_t>(omp_get_thread_num());
    const size_t ifirst = ss.size() / p * (t)   + std::min(t,   ss.size() % p);
    const size_t ilast  = ss.size() / p * (t+1) + std::min(t+1, ss.size() % p);
    auto         it     = adv(ss.begin() + ifirst, ss.begin(), ss.end());
    const auto   last   = adv(ss.begin() + ilast,  ss.begin(), ss.end());

    // Reduce edges and store them in std::vector
    while (it != last) 
    {
      const vint s = get<0>(*it);

      // Obtain an upper bound of degree and reserve memory
      const auto maxdeg = std::find_if(it, last, [s](auto& x) {return get<0>(x) != s;}) - it;
      adj[s].reserve(maxdeg);

      while (it != last && get<0>(*it) == s) 
      {
        const vint t = get<1>(*it);
        float      w = 0.0;
        while (it != last && get<0>(*it) == s && get<1>(*it) == t)
          w += get<2>(*it++);
        if (w > 0.0)
          adj[s].push_back({t, w});
      }

      // The actual degree can be smaller than the upper bound
      adj[s].shrink_to_fit();
    }
  }

  return adj;
}

adjacency_list read_graph_mtx(const int* row_ind,
                              const int* col_ind,
                              const int  nrow,
                              const int  nnz) 
{
  std::vector<std::tuple<vint, vint, float>> edges;
  edges.reserve(nnz);

  for (int i=0; i<nnz; ++i)
  {
    edges.push_back({row_ind[i], col_ind[i], 1.0});
  }

  return make_adj_list(nrow, edges);
}


std::unique_ptr<std::pair<vint, vint>[]> merge_order(const graph& g) 
{
  // Co-locating vertex ID and its degree shows better locality
  auto ord = std::make_unique<std::pair<vint, vint>[]>(g.n());
  #pragma omp parallel for
  for (vint v = 0; v < g.n(); ++v)
  {
    ord[v] = {v, static_cast<vint>(g.es[v].size())};
  }

  __gnu_parallel::sort(&ord[0], &ord[g.n()], [](auto x, auto y) {return x.second < y.second;});
  return ord;
}


vint trace_com(const vint v, graph* const g) 
{
  vint com = v;
  for (;;) {
    const vint c = g->coms[com];
    if (c == com) break;
    com = c;
  }

  if (v != com && g->coms[v] != com)
    g->coms[v] = com;

  return com;
}

bool is_toplevel(const graph& g, const vint v) 
{
  return g.vs[v].a->str >= 0.0 && g.vs[v].sibling == vmax && g.coms[v] == v;
}

vint find_best(const graph& g, const vint v, const double vstr) 
{
  double dmax = 0.0;
  vint   best = v;
  for (const edge e : g.es[v]) 
  {
    const double d = static_cast<double>(e.second) - vstr * g.vs[e.first].a->str / g.tot_wgt;
    if (dmax < d) 
    {
      dmax = d;
      best = e.first;
    }
  }
  return best;
}

//
// Checks result of incremental aggregation using `assert`
//
bool check_result(graph* const pg) 
{
  static_cast<void>(pg);

#ifndef NDEBUG
  auto&      g     = *pg;
  const auto vall  = boost::irange(static_cast<vint>(0), g.n());
  const auto istop = [&g](const vint v) {return is_toplevel(g, v);};

  // For all vertex `v`, `trace_com(v)` is in `g.tops`
  {
    std::unordered_set<vint> topids;
    for (vint v = 0; v < g.n(); ++v) topids.insert(trace_com(v, &g));
    std::vector<vint> got(topids.begin(), topids.end());
    auto              ans = *g.tops;
    assert(boost::equal(boost::sort(ans), boost::sort(got)));
  }

  // `g.tops` includes only top-level vertices
  assert(boost::algorithm::all_of(*g.tops, istop));
  // The number of the top-level vertices is equal to the size of `g.tops`,
  // i.e., `g.tops` includes all the top-level vertices
  assert(boost::count_if(vall, istop) == static_cast<intmax_t>(g.tops->size()));

  // All the remaining communities are top-level
  assert(boost::algorithm::all_of(vall, [&g](auto v) {
    const vint c = trace_com(v, &g);
    return is_toplevel(g, c);
  }));

  // Every vertex `v` is consistent as a top-level vertex or a merged vertex
  assert(boost::algorithm::all_of(vall, [&g](auto v) {
    return is_toplevel(g, v) || is_merged(g, v);
  }));
#endif

  return true;
}


inline bool is_merged(const graph& g, const vint v) 
{
  return g.vs[v].a->str < 0.0 && g.coms[v] != v;
}

void unite(const vint v, std::vector<edge>* const nbrs, graph* const g) 
{
  ptrdiff_t icmb = 0;

  nbrs->clear();

  const auto push_edges = [v, nbrs, g, &icmb](const vint u) {
    const size_t     cap  = nbrs->capacity();
    constexpr size_t npre = 8;  // TODO: tuning parameter
    auto&            es   = g->es[u];

    for (size_t i = 0; i < es.size() && i < npre; ++i)
      __builtin_prefetch(&g->coms[es[i].first], 0, 3);
    for (size_t i = 0; i < es.size(); ++i) {
      if (i + npre < es.size())
        __builtin_prefetch(&g->coms[es[i + npre].first], 0, 3);
      const vint c = trace_com(es[i].first, g);
      if (c != v)  // Remove a self-loop edge
        nbrs->push_back({c, es[i].second});
    }

#ifdef DEBUG
    if (nbrs->size() > cap)
      std::cerr << "WARNING: edge accumulation buffer is reallocated\n";
#else
    static_cast<void>(cap);
#endif

    // combine edges before uncombined edges overflows a L2 cache
    // TODO: tuning
    if (nbrs->size() - icmb >= 2048) {
      const auto it = nbrs->begin() + icmb;
      icmb = compact(it, nbrs->end(), it) - nbrs->begin();
      nbrs->resize(icmb);
    }
  };

  push_edges(v);

  // `child` may be modified if another thread merges a vertex into `v`, but
  // this function is not responsible for prohibiting modification of `child`.
  while (g->vs[v].united_child != g->vs[v].a->child) {
    // The vertices in the list connected by `sibling` are already merged, and
    // hence they are never be modified by the other threads.
    const vint c = g->vs[v].a->child;
    vint       w;
    for (w = c; w != vmax && w != g->vs[v].united_child; w = g->vs[w].sibling)
      push_edges(w);

    // `c` and the descendants of `c` have been merged into `v`
    g->vs[v].united_child = c;
  }

  g->tot_nbrs.fetch_add(nbrs->size());

  g->es[v].clear();
  compact(nbrs->begin(), nbrs->end(), std::back_inserter(g->es[v]));
}

vint merge(const vint v, std::vector<edge>* const nbrs, graph* const g) 
{
  // Aggregate edges of the members of community `v`
  // Aggregating before locking `g[v]` shortens the locking time
  unite(v, nbrs, g);

  // `.str < 0.0` means that modification of `g[v]` is prohibited (locked)
  const float vstr = g->vs[v].a->str.exchange(-1);

  // If `.child` was modified between the previous call of `unite()` and the
  // lock, aggregate edges again
  if (g->vs[v].a->child != g->vs[v].united_child) 
  {
    unite(v, nbrs, g);
    g->n_reunite.fetch_add(1);
  }

  const vint u = find_best(*g, v, vstr);
  if (u == v) {
    // Rollback the strength if there is no neighbor that improves modularity
    g->vs[v].a->str = vstr;
  } else {
    // Rollback if `u` has a negative strength (i.e., `u` is locked)
    atom ua = g->vs[u].a;  // atomic load
    if (ua.str < 0.0) {
      g->vs[v].a->str = vstr;
      g->n_fail_lock.fetch_add(1);
      return vmax;
    }

    // `.sibling` can be accessed immediately by `unite()` after letting
    // `g->vs[u].a->child = v`, and so set `.sibling` properly in advance
    g->vs[v].sibling = ua.child;

    // Abort and rollback if CAS failed due to modification of `u`
    const atom _ua(ua.str + vstr, v);
    if (!g->vs[u].a.compare_exchange_weak(ua, _ua)) {
      g->vs[v].sibling = vmax;
      g->vs[v].a->str  = vstr;
      g->n_fail_cas.fetch_add(1);
      return vmax;
    }

    // Update the community of `v`
    g->coms[v] = u;
  }

  assert(u != v || is_toplevel(*g, v));
  assert(u == v || is_merged(*g, v));

  return u;
}

graph aggregate(std::vector<std::vector<edge>>  adj) 
{
  graph      g(std::move(adj));
  const auto ord   = merge_order(g);
  const int  np    = omp_get_max_threads();
  size_t     npend = 0;
  double     tmax  = 0.0, ttotal = 0.0;
  std::vector<std::deque<vint> > topss(np);

  #pragma omp parallel reduction(+: npend) reduction(max: tmax) reduction(+: ttotal)
  {
    const double     tstart = now_sec();
    const int        tid    = omp_get_thread_num();
    std::deque<vint> tops, pends;

    std::vector<edge> nbrs;
    nbrs.reserve(g.n() * 2);  // heuristic value   TODO: tuning

    #pragma omp for schedule(static, 1)
    for (vint i = 0; i < g.n(); ++i) {
      pends.erase(boost::remove_if(pends, [&g, &tops, &nbrs](auto w) {
        const vint u = merge(w, &nbrs, &g);
        if (u == w) tops.push_back(w);
        return u != vmax;  // remove if the merge successed
      }), pends.end());

      const vint v = ord[i].first;
      const vint u = merge(v, &nbrs, &g);
      if      (u == v)    tops.push_back(v);
      else if (u == vmax) pends.push_back(v);
    }

    ttotal = now_sec() - tstart;
    tmax   = ttotal;

    // Merge the vertices in the pending state 
    #pragma omp barrier
    #pragma omp critical
    {
      npend = pends.size();
      for (const vint v : pends) {
        const vint u = merge(v, &nbrs, &g);
        if (u == v) tops.push_back(v);
        assert(u != vmax);  // The merge never fails
      }
      topss[tid] = std::move(tops);
    }
  }

  g.tops = join(topss);

  // `tops` does not have duplicated elements
  assert(([&g]() {
    auto tops = *g.tops;
    return g.tops->size() == boost::size(boost::unique(boost::sort(tops)));
  })());

  //std::cerr << "CPU time utilization rate: " <<  ttotal / (tmax * np)
  //          << "\nvertices left to be pended: " << npend
  //          << "\n`unite()` calls after lock: " << g.n_reunite
  //          << "\nmerge failures by negative-strength: " << g.n_fail_lock
  //          << "\nmerge failures by compare-and-swap: " << g.n_fail_cas
  //          << "\ntot_nbrs = " << g.tot_nbrs << std::endl;
  static_cast<void>(npend);  // suppress Wunused-but-set-variable
  static_cast<void>(tmax);

  assert(check_result(&g));
  return g;
}

std::unique_ptr<vint[]> compute_perm(const graph& g) 
{
  auto              perm = std::make_unique<vint[]>(g.n());
  auto              coms = std::make_unique<vint[]>(g.n());
  const vint        ncom = static_cast<vint>(g.tops->size());
  std::vector<vint> offsets(ncom + 1);

  const int  np    = omp_get_max_threads();
  const vint ntask = std::min<vint>(ncom, 128 * np);
  #pragma omp parallel
  {
    std::deque<vint> stack;

    #pragma omp for schedule(dynamic, 1)
    for (vint i = 0; i < ntask; ++i) {
      for (vint comid = i; comid < ncom; comid += ntask) {
        vint newid = 0;

        descendants(g, (*g.tops)[comid], std::back_inserter(stack));

        while (!stack.empty()) {
          const vint v = stack.back();
          stack.pop_back();

          coms[v] = comid;
          perm[v] = newid++;

          if (g.vs[v].sibling != vmax)
            descendants(g, g.vs[v].sibling, std::back_inserter(stack));
        }

        offsets[comid + 1] = newid;
      }
    }
  }

  boost::partial_sum(offsets, offsets.begin());
  assert(offsets.back() == g.n());

  #pragma omp parallel for schedule(static)
  for (vint v = 0; v < g.n(); ++v)
    perm[v] += offsets[coms[v]];

  // `perm` must contain `[0, g.n())`
  assert(([&g, &perm]() {
    std::vector<vint> sorted(&perm[0], &perm[g.n()]);
    return boost::equal(boost::sort(sorted),
                        boost::irange(static_cast<vint>(0), g.n()));
  })());

  return perm;
}


void rabbit_order(int*          row_ind, 
                  int*          col_ind, 
                  int           nrow, 
                  int           nnz,
                  unsigned int* permutation) 
{
  auto adj = read_graph_mtx(row_ind, col_ind, nrow, nnz);
  auto g = aggregate(std::move(adj));
  auto p = compute_perm(g);
  for(int i=0; i<nrow; ++i)
  {
    permutation[i] = p[i];
  }
}

