"""[v6: also saves, per representative orientation, its surviving leaves in the representative's OWN labelling (for union LPs).]
[v5: + the tail lemma 26.2 (t_c = 0 => d_c in {0,2,3,6}) when there is NO sister pair (Sigma = empty).]
[v6.1 (note 43): 26.2 in Sigma-form at every block with t_c = 0: w_c = d_c + sigma_c in {0,2,3,6} / {1,3,6} / {2,6} for sigma_c = 0/1/2.]
[v4: A-orbit placeholders are no longer "anything": an n-vector is allowed iff it keeps the profile of (A, j, k) feasible for
SOME third block k (union over the 5 possible k; comps of A unknown = one component).  Still sound.]
[v3: a degree-6 block A (all six pairs anchored at A) is kept GENERIC: its D-structure is not enumerated, its 12 orbits
enter the free graphs of the other blocks as placeholders with unknown triple (any n-vector allowed), and G_A counts as one
component.  Leaves are NON-A PARTS; the LP stage then keeps A variable (lambda over its D-graphs).  Sound (weaker checks only).]
[v2: + SISTER PAIRS (Sigma given): sister pairs are self-matched at both blocks (not in D-graphs, never free), and give
singleton components {b} of G_c for {c,b} sister (mu_c = tau_c = sigma on +-b).  Orientation pruning only by 19.1 (f != 2;
f = 4 excludes the two-sister and K22 patterns); the tail lemma 26.2 is NOT used.]
General SOUND block filter for Sigma = empty, one-block-square configurations with ANY twisted subgraph (note 35).
Skeleton = orientation (twisted set + heads of anchored pairs) + D-graph at every pivot block.
Checks (all proved): at every block j, the free orbits admit a 2-regular graph with even cycles >= 4, no sister edge, sisters with
<= 1 common neighbour (19.1), typed rule (n_k <= 1, note 30), and every orbit's n-vector keeps its single-orbit profile feasible
(27.1: F1 + balance on the components of G_c that are CERTAIN: the closed D-cycles at c, i.e. components of the D-graph at c
without twisted pair, and their complement; unassigned blocks: a single component).  Partial skeletons are checked with the
information available (sound pruning).  Symmetry: Aut(twisted graph) on orientations and on surviving skeletons.
Usage: python mix_filter.py <twisted class index> <k> <K> <seconds>"""
import sys, time, itertools, json
from collections import Counter, defaultdict
sys.path.insert(0, "scripts")
import mix_enum as ME, m1_blocks as MB
from m1_blocks3 import two_factor_exists
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
N = 7; P = ME.P

def twisted_classes():
    basis = [{(0, a), (0, b), (a, b)} for a, b in itertools.combinations(range(1, N), 2)]
    seen = []
    for mask in range(1 << 15):
        s = set()
        for t in range(15):
            if mask >> t & 1: s ^= basis[t]
        G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(s)
        key = (G.number_of_edges(), tuple(sorted(d for _, d in G.degree())))
        if any(k2 == key and nx.is_isomorphic(G, H) for k2, H in seen): continue
        seen.append((key, G))
    out = []
    for key, G in sorted(seen, key=lambda kg: (kg[0][0], kg[0][1])):
        H = G.copy(); H.remove_nodes_from([v for v in list(H) if H.degree(v) == 0])
        single = H.number_of_nodes() > 0 and all(d == 2 for _, d in H.degree()) and nx.is_connected(H)
        if G.number_of_edges() in (0, 21) or single: continue
        out.append(sorted(tuple(sorted(e)) for e in G.edges()))
    return out

def automorphisms(twE):
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(twE)
    return [dict(m) for m in GraphMatcher(G, G).isomorphisms_iter()]

SIS = set()
LEAVES = []
def comps_of(c, dg, assigned):
    rest = set(range(N)) - {c}
    sing = [{b} for b in range(N) if b != c and P(c, b) in SIS]
    if sing:
        rest2 = rest - {b for S_ in sing for b in S_}
        if c not in assigned: return sing + ([rest2] if rest2 else [])
        edges, twl = dg
        Gd = nx.MultiGraph(); Gd.add_nodes_from({x for e in edges for x in e}); Gd.add_edges_from(edges)
        out = list(sing); used = set()
        for comp in nx.connected_components(Gd):
            if not (comp & twl): out.append(set(comp)); used |= comp
        left = rest2 - used
        return out + ([left] if left else [])
    if c not in assigned: return [rest]
    edges, twl = dg
    Gd = nx.MultiGraph(); Gd.add_nodes_from({x for e in edges for x in e}); Gd.add_edges_from(edges)
    out = []; used = set()
    for comp in nx.connected_components(Gd):
        if not (comp & twl): out.append(set(comp)); used |= comp
    if not out: return [rest]
    return out + [rest - used] if rest - used else out

def block_ok(j, orbit_list, comps, memo, Ablk=None):
    free = [o for o in orbit_list if o[1] == j]
    if Ablk is not None and j != Ablk:
        free = free + [(Ablk, j, None), (Ablk, j, None)]
    if not free: return True
    ids = list(range(len(free))); typ = {t: free[t][0] for t in ids}
    sis_of = {}
    for a, b in itertools.combinations(ids, 2):
        if typ[a] == typ[b]: sis_of[a] = b; sis_of[b] = a
    present = sorted(set(typ.values())); allowed = {}
    for t in ids:
        i, jj, k = free[t][:3]; al = set()
        if k is None:
            for r1, r2 in itertools.combinations_with_replacement(present, 2):
                nv = Counter([r1, r2])
                if any(nv[r] > sum(1 for u in ids if typ[u] == r) - (1 if r == typ[t] else 0) for r in nv): continue
                if r1 == r2 == typ[t]: continue
                for kk in range(N):
                    if kk in (i, jj): continue
                    if r1 == r2 == kk: continue                          # typed rule for this k
                    key = (i, jj, kk, r1, r2, tuple(tuple(sorted(C)) for c in range(N) if c not in (i, jj) for C in comps[c]))
                    if key not in memo:
                        fixed = {r: nv.get(r, 0) for r in range(N) if r not in (i, jj)}
                        memo[key] = MB.feasible_exact(i, jj, kk, comps, fixed)
                    if memo[key]: al.add((r1, r2)); break
            if not al: return False
            allowed[t] = al; continue
        for r1, r2 in itertools.combinations_with_replacement(present, 2):
            nv = Counter([r1, r2])
            if any(nv[r] > sum(1 for u in ids if typ[u] == r) - (1 if r == typ[t] else 0) for r in nv): continue
            if r1 == r2 == typ[t]: continue
            if r1 == r2 == k: continue                                  # typed rule
            key = (i, jj, k, r1, r2, tuple(tuple(sorted(C)) for c in range(N) if c not in (i, jj) for C in comps[c]))
            if key not in memo:
                fixed = {r: nv.get(r, 0) for r in range(N) if r not in (i, jj)}
                memo[key] = MB.feasible_exact(i, jj, k, comps, fixed)
            if memo[key]: al.add((r1, r2))
        if not al: return False
        allowed[t] = al
    return two_factor_exists(ids, typ, allowed, sis_of)

def orbits_from(dgs):
    out = []
    for c, (edges, twl) in dgs.items():
        for (x, y) in edges: out += [(c, x, y), (c, y, x)]
    return out

def canon_skel(orbs, auts):
    return min(tuple(sorted(tuple(g[v] for v in z) for z in orbs)) for g in auts)

# 26.2 in Sigma-form (C1 notes/130 par. 2, checked in C2 note 43): at a block c with no twisted pair, Out(c) is a union of components
# of G_c (anchored classes on In(c) are closed, sister partners are singletons), so an outgoing arc c -> i forces o_c >= 3 by the
# tail lemma 25.1; with d_c != 1 (a lone anchored pair would be self-paired): w_c = d_c + sigma_c in W_OK[sigma_c].
W_OK = {0: (0, 2, 3, 6), 1: (1, 3, 6), 2: (2, 6)}

def orientations2(twE, sisE):
    tw = {P(a, b) for a, b in twE}; sis = {P(a, b) for a, b in sisE}
    tdeg = Counter(v for e in tw for v in e); sdeg = Counter(v for e in sis for v in e)
    rest = [P(a, b) for a, b in itertools.combinations(range(N), 2) if P(a, b) not in tw and P(a, b) not in sis]
    for bits in range(1 << len(rest)):
        In = {c: [] for c in range(N)}; Out = {c: [] for c in range(N)}
        for t, R in enumerate(rest):
            a, b = sorted(R)
            head, tail = (a, b) if (bits >> t) & 1 else (b, a)
            In[head].append(tail); Out[tail].append(head)
        ok = True
        for c in range(N):
            out = len(Out[c]); f = 2 * out + tdeg[c]
            if f == 2 or (f == 4 and out >= 1): ok = False; break     # 19.1: digon; 4-cycle with two sisters; K22
            if tdeg[c] == 0 and len(In[c]) + sdeg[c] not in W_OK.get(sdeg[c], ()): ok = False; break   # 26.2, Sigma-form (note 43)
        if ok: yield In, tw

def automorphisms2(twE, sisE):
    G = nx.Graph(); G.add_nodes_from(range(N))
    for e in twE: G.add_edge(*e, kind='t')
    for e in sisE: G.add_edge(*e, kind='s')
    em = lambda a, b: a['kind'] == b['kind']
    return [dict(m) for m in GraphMatcher(G, G, edge_match=em).isomorphisms_iter()]

def run(twE, k, K, budget, sisE=()):
    global SIS
    SIS = {P(a, b) for a, b in sisE}
    auts = automorphisms2(twE, sisE); t0 = time.time()
    reps = {}
    for In, tw in orientations2(twE, sisE):
        arcs = [(a, b) for b in range(N) for a in In[b]]
        key = min(tuple(sorted((g[a], g[b]) for a, b in arcs)) for g in auts)
        if key not in reps: reps[key] = In
    reps = list(reps.values())
    tw = {P(a, b) for a, b in twE}
    surv = {}; stats = Counter(); done = True; per_orient = []
    for idx, In in enumerate(reps):
        if idx % K != k: continue
        stats['orient'] += 1
        opts = {}
        for c in range(N):
            twl = {x for x in range(N) if P(c, x) in tw}
            opts[c] = [(g, twl) for g in ME.dgraphs(In[c], sorted(twl))]
        # pivots of the orbits free at j: blocks c with {c,j} anchored at c, or twisted
        Ablk = next((c for c in range(N) if len(In[c]) == 6), None)
        need = {j: {c for c in range(N) if c != j and c != Ablk and (j in In[c] or P(c, j) in tw)} for j in range(N)}
        order = sorted([c for c in range(N) if c != Ablk], key=lambda c: len(opts[c]))
        memo = {}
        def dfs(pos, dgs):
            nonlocal done
            if time.time() - t0 > budget: done = False; return
            assigned = set(dgs)
            comps = {c: comps_of(c, dgs.get(c), assigned) for c in range(N)}
            orbs = orbits_from(dgs)
            for j in range(N):
                if need[j] <= assigned and (need[j] or Ablk is not None) and j != Ablk and not block_ok(j, orbs, comps, memo, Ablk):
                    stats['pruned', pos] += 1; return
            if pos == len(order):
                key = canon_skel(orbs, auts); stats['leaf'] += 1
                LEAVES.append(tuple(sorted(orbs)))
                if key not in surv: surv[key] = {"In": {str(a): b for a, b in In.items()}, "A": Ablk}
                return
            c = order[pos]
            for o in opts[c]:
                dgs[c] = (o[0], o[1]); dfs(pos + 1, dgs); del dgs[c]
                if not done: return
        leaves_here = []
        def dfs_wrap():
            pass
        dfs(0, {})
        per_orient.append({"In": {str(a): b for a, b in In.items()}, "A": Ablk, "leaves": sorted(set(LEAVES))})
        LEAVES.clear()
        if not done: break
    return reps, surv, stats, done, per_orient

if __name__ == "__main__":
    ci, k, K, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    sisE = [tuple(map(int, e)) for e in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] else []
    if ci.startswith("edges="): twE = [tuple(map(int, e)) for e in ci[6:].split(",") if e]; ci = "E" + ci[6:].replace(",", "_")
    else: ci = int(ci); twE = twisted_classes()[ci]
    reps, surv, stats, done, per_orient = run(twE, k, K, budget, sisE)
    print("twisted class", ci, twE, "| orientation classes", len(reps), "| worker", k, dict(stats), "| surviving skeleton classes", len(surv), "done" if done else "TIME CAP")
    json.dump({"twE": twE, "done": done, "classes": [[list(map(list, key)), In] for key, In in surv.items()],
               "per_orient": [{"In": po["In"], "A": po["A"], "leaves": [list(map(list, L)) for L in po["leaves"]]} for po in per_orient]},
              open("logs/mix6_cls_%s_S%s_%d.json" % (ci, "_".join("%d%d" % e for e in sisE), k), "w"))
