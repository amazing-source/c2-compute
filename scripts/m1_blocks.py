# Block-level SOUND filter for M1 (on top of 28.x and single-orbit profiles):
# at a block j, the free orbits form a 2-regular graph (19.1): cycles of length >= 4 (alternating m/pi, so even), no edge
# between sister orbits, sister orbits have <= 1 common neighbour. Each free orbit x (with known (i,j,k)) has its n-vector
# (n_r = # of its two neighbours lying in pair {j,r}) determined by the free graph; that n-vector must make x's profile
# feasible (note 27.1). A block with f_j <= 8 free orbits is checked exhaustively; blocks with f_j > 8 and orbits whose (i,j,k)
# is not determined (pivot = A, whose structure is kept generic) are left unconstrained (conservative).
import itertools, sys, time
from collections import Counter
N = 7
src = open("scripts/m1_profiles.py", encoding="utf-8").read()
exec(src.split("def run(n, t0, budget):")[0].split("N = 7", 1)[1])

def feasible_fixed_n(i, j, k, comps, nfix):
    # same system as feasible(), with the n_r fixed (n_r for r not in nfix is 0)
    others = [r for r in range(N) if r not in (i, j)]
    fc = {r: nfix.get(r, 0) for r in others}
    # emulate: domains of {j,r} collapse to the fixed value
    return feasible_exact(i, j, k, comps, fc)

def feasible_exact(i, j, k, comps, fixed):
    others = [r for r in range(N) if r not in (i, j)]
    var_pairs = [frozenset(p) for p in itertools.combinations(others, 2)]
    dom = {P: range(0, 3) for P in var_pairs}
    def const(c, r):
        P = frozenset((c, r))
        if i in P: return 2 if P == frozenset((i, k)) else 0
        if j in P: return fixed[c if r == j else r]
        return None
    cons = []
    for c in others:
        for C in comps[c]:
            tgt = len(C) - len({i, j} & C); cst = 0; ps = []
            for r in C:
                if frozenset((c, r)) == frozenset((i, j)): continue
                v = const(c, r)
                if v is not None: cst += v
                else: ps.append(frozenset((c, r)))
            cons.append((ps, cst, tgt))
    cons.append((var_pairs, 0, 8))
    order = sorted(var_pairs, key=lambda P: -sum(P in c[0] for c in cons))
    val = {}
    def okp():
        for ps, cst, tgt in cons:
            s = cst; mx = 0
            for P in ps:
                if P in val: s += val[P]
                else: mx += 2
            if s > tgt or s + mx < tgt: return False
        return True
    if not okp(): return False
    def dfs(t):
        if t == len(order): return True
        P = order[t]
        for v in dom[P]:
            val[P] = v
            if okp() and dfs(t + 1): return True
        del val[P]; return False
    return dfs(0)

def two_regular_graphs(V, sisters):
    # all 2-regular simple graphs on V with all cycles of even length >= 4, no sister edge; yields adjacency dicts
    V = list(V); res = []
    def rec(rem, edges):
        if not rem: res.append(edges[:]); return
        first = rem[0]; rest = rem[1:]
        for L in range(4, len(rem) + 1, 2):
            if len(rem) - L not in (0,) and len(rem) - L < 4: continue
            for comb in itertools.combinations(rest, L - 1):
                for perm in itertools.permutations(comb):
                    if perm[0] > perm[-1]: continue
                    cyc = (first,) + perm
                    es = [(cyc[t], cyc[(t + 1) % L]) for t in range(L)]
                    if any(frozenset(e) in sisters for e in es): continue
                    rec([v for v in rem if v not in cyc], edges + es)
    rec(V, []); return res

def run(n, t0, budget):
    T = list(range(n))
    tw = {frozenset((T[m], T[(m + 1) % n])) for m in range(n)}
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N) if frozenset((u, v)) not in tw]
    out = Counter()
    for bits in range(1 << len(pairs)):
        if time.time() - t0 > budget: return out, False
        In = {c: [] for c in range(N)}
        for t, (u, v) in enumerate(pairs):
            a, b = (u, v) if (bits >> t) & 1 else (v, u)
            In[b].append(a)
        d = {c: len(In[c]) for c in range(N)}
        A = [c for c in range(N) if d[c] == 6]
        if len(A) > 1: continue
        if any((c < n and d[c] > 2) or (c >= n and d[c] not in (0, 2, 3, 6)) for c in range(N)): continue
        Ab = A[0] if A else None
        Tw = {c: ([T[(c - 1) % n], T[(c + 1) % n]] if c < n else []) for c in range(N)}
        outdeg = {c: 6 - d[c] - len(Tw[c]) for c in range(N)}
        f = {c: 2 * outdeg[c] + len(Tw[c]) for c in range(N)}
        def tail(a, b):
            if frozenset((a, b)) in tw: return None
            return a if a in In[b] else b
        opts = []
        for c in range(N):
            if c == Ab: opts.append([('A', None)]); continue
            if c >= n: opts.append([('C', tuple(sorted(In[c])))]); continue
            I = In[c]
            if d[c] == 0: opts.append([('P', ())])
            elif d[c] == 1: opts.append([('P', (I[0],))])
            else: opts.append([('L', tuple(sorted(I))), ('P', (I[0], I[1])), ('P', (I[1], I[0]))])
        cc = sum(1 for c in range(n, N) if d[c] == 3)
        survived = False
        for choice in itertools.product(*opts):
            # D-edges per pivot: list of (pivot, pairblock_x, pairblock_y) ; orbit of {pivot,x} matched to orbit of {pivot,y}
            dedges = []; linked = {}; cert42 = {}; cert33 = {}
            for c in range(N):
                ch = choice[c]
                if ch[0] == 'A': continue
                if ch[0] == 'C':
                    I = list(ch[1])
                    if len(I) == 2:
                        a, b = I; dedges += [(c, a, b), (c, a, b)]; linked[c] = set(I); cert42[c] = set(I)
                    elif len(I) == 3:
                        a, b, e = I; dedges += [(c, a, b), (c, b, e), (c, e, a)]; cert33[c] = set(I)
                    continue
                P1, P2 = Tw[c]
                if ch[0] == 'L':
                    a, b = ch[1]; dedges += [(c, a, b), (c, a, b), (c, P1, P2)]; linked[c] = {a, b}; cert42[c] = {a, b}
                else:
                    seq = [P1] + list(ch[1]) + [P2]
                    for x, y in zip(seq, seq[1:]): dedges.append((c, x, y))
            orbits = []
            for (c, x, y) in dedges: orbits += [(c, x, y), (c, y, x)]
            bad = any(k in cert33 and j not in cert33[k] for (i, j, k) in orbits)
            if not bad:
                for L in linked.values():
                    a, b = sorted(L); t_ = tail(a, b)
                    if t_ is not None and f[t_] < 8: bad = True; break
            if not bad:
                for L in linked.values():
                    for (i, j, k) in orbits:
                        if {i, k} == L and j not in L and k in cert42 and j not in cert42[k]: bad = True; break
                    if bad: break
            if bad: continue
            comps = {c: components_for(c, choice, In, Tw, n, d) for c in range(N)}
            # free orbits at each block j: entries (orbit_id, r, (i,j,k) or None)
            # orbit ids: each D-edge endpoint is one orbit: orbit of pair {c, x} matched at pivot c -> free at x, triple (c, x, y)
            blockok = True
            for j in range(N):
                if f[j] == 0 or f[j] > 8: continue
                free = []   # (id, r, triple)
                for idx, (c, x, y) in enumerate(dedges):
                    if x == j: free.append((('e', idx, 0), c, (c, j, y)))
                    if y == j: free.append((('e', idx, 1), c, (c, j, x)))
                # orbits of pairs {j, A} (via A): two sisters, triple unknown
                if Ab is not None and j != Ab:
                    free.append((('A', j, 0), Ab, None)); free.append((('A', j, 1), Ab, None))
                if len(free) != f[j]:
                    continue    # bookkeeping mismatch -> skip block (conservative)
                ids = [fr[0] for fr in free]
                byid = {fr[0]: fr for fr in free}
                # sisters: two free orbits of the same pair {j, r} with r anchored (r in Out(j))
                sis = set()
                for a_, b_ in itertools.combinations(free, 2):
                    if a_[1] == b_[1] and frozenset((j, a_[1])) not in tw: sis.add(frozenset((a_[0], b_[0])))
                graphs = two_regular_graphs(ids, sis)
                cache = {}
                found = False
                for es in graphs:
                    nb = {v: [] for v in ids}
                    for (u, w) in es: nb[u].append(w); nb[w].append(u)
                    if any(len(set(nb[a_]) & set(nb[b_])) > 1 for s_ in sis for (a_, b_) in [tuple(s_)]): continue
                    good = True
                    for v in ids:
                        trip = byid[v][2]
                        if trip is None: continue
                        nvec = Counter(byid[w][1] for w in nb[v])
                        key = (trip, tuple(sorted(nvec.items())))
                        if key not in cache:
                            i, jj, k = trip
                            fixed = {r: nvec.get(r, 0) for r in range(N) if r not in (i, jj)}
                            cache[key] = feasible_exact(i, jj, k, comps, fixed)
                        if not cache[key]: good = False; break
                    if good: found = True; break
                if not found: blockok = False; break
            if blockok: survived = True; break
        if survived: out[(len(A), cc, n)] += 1
    return out, True

if __name__ == "__main__":
    t0 = time.time(); tot = Counter()
    for n in range(3, 8):
        r, done = run(n, t0, 285)
        tot.update(r); print("n =", n, dict(sorted(r.items())), "" if done else "(TIME CAP: partial)"); sys.stdout.flush()
        if not done: break
    print("block-level survivors:", dict(sorted(tot.items())), "time %.0fs" % (time.time() - t0))
