# Block-level SOUND filter, v2: all blocks (any number of free orbits), via a typed 2-factor search.
# For each free orbit v at block j with known triple (i,j,k): allowed unordered neighbour-type pairs {r1,r2} (r = the other block of
# the neighbour's pair) are those whose n-vector keeps v's profile feasible (note 27.1). Orbits with unknown triple (pivot A,
# A kept generic) accept any types. Search a 2-regular graph on the free orbits: no sister edge, sisters <= 1 common neighbour,
# every cycle even and of length >= 4 (19.1). Sound and conservative (A's structure and A's components are not used).
import itertools, sys, time
from collections import Counter
sys.path.insert(0, "scripts")
import m1_blocks as MB
from lib_cycles import cycle_structures
N = 7

def two_factor_exists(ids, typ, allowed, sis_of):
    # ids: vertices; typ[v] = block r of v's pair; allowed[v] = set of frozenset/tuple sorted (r1,r2) or None (any)
    # sis_of[v] = sister vertex or None
    idx = {v: t for t, v in enumerate(ids)}
    nb = {v: [] for v in ids}
    def pair_ok(v):
        if len(nb[v]) < 2 or allowed[v] is None: return True
        return tuple(sorted((typ[nb[v][0]], typ[nb[v][1]]))) in allowed[v]
    def partial_ok(v):
        if allowed[v] is None or len(nb[v]) == 0: return True
        if len(nb[v]) == 1:
            t = typ[nb[v][0]]
            return any(t in pr for pr in allowed[v])
        return pair_ok(v)
    def sis_ok(v):
        s = sis_of.get(v)
        if s is None: return True
        return len(set(nb[v]) & set(nb[s])) <= 1
    def cycles_ok():
        seen = set()
        for v in ids:
            if v in seen: continue
            L = 0; prev = None; cur = v
            while True:
                seen.add(cur); L += 1
                nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
                prev, cur = cur, nxt
                if cur == v: break
            if L < 4 or L % 2: return False
        return True
    def dfs():
        # pick vertex with fewest neighbours
        cand = [v for v in ids if len(nb[v]) < 2]
        if not cand: return cycles_ok()
        v = min(cand, key=lambda u: len(nb[u]))
        for w in ids:
            if w == v or len(nb[w]) >= 2 or w in nb[v] or sis_of.get(v) == w: continue
            if idx[w] < idx[v] and len(nb[v]) == 0 and False: continue
            nb[v].append(w); nb[w].append(v)
            if partial_ok(v) and partial_ok(w) and sis_ok(v) and sis_ok(w) and \
               all(sis_ok(s) for s in (sis_of.get(v), sis_of.get(w)) if s is not None):
                # forbid closing a short cycle early: if v-w path closes a cycle of length < 4 it's caught at end; prune triangles
                if dfs(): return True
            nb[v].pop(); nb[w].pop()
        return False
    return dfs()

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
            if c == Ab: opts.append([('A', cs) for cs in cycle_structures(In[c])]); continue
            if c >= n: opts.append([('C', tuple(sorted(In[c])))]); continue
            I = In[c]
            if d[c] == 0: opts.append([('P', ())])
            elif d[c] == 1: opts.append([('P', (I[0],))])
            else: opts.append([('L', tuple(sorted(I))), ('P', (I[0], I[1])), ('P', (I[1], I[0]))])
        cc = sum(1 for c in range(n, N) if d[c] == 3)
        survived = False
        for choice in itertools.product(*opts):
            dedges = []; linked = {}; cert42 = {}; cert33 = {}
            for c in range(N):
                ch = choice[c]
                if ch[0] == 'A':
                    for cyc in ch[1]:
                        if len(cyc) == 2:
                            a, b = cyc; dedges += [(c, a, b), (c, a, b)]; linked[('A', a, b)] = {a, b}
                        else:
                            for x, y in zip(cyc, cyc[1:] + cyc[:1]): dedges.append((c, x, y))
                    continue
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
            comps = {c: MB.components_for(c, choice, In, Tw, n, d) for c in range(N)}
            if Ab is not None:
                comps[Ab] = [set(cyc) for cyc in choice[Ab][1]]
            blockok = True; memo = {}
            for j in range(N):
                if f[j] == 0: continue
                free = []
                for idx, (c, x, y) in enumerate(dedges):
                    if x == j: free.append((('e', idx, 0), c, (c, j, y)))
                    if y == j: free.append((('e', idx, 1), c, (c, j, x)))
                if len(free) != f[j]: continue
                ids = [fr[0] for fr in free]; typ = {fr[0]: fr[1] for fr in free}
                sis_of = {}
                for a_, b_ in itertools.combinations(free, 2):
                    if a_[1] == b_[1] and frozenset((j, a_[1])) not in tw: sis_of[a_[0]] = b_[0]; sis_of[b_[0]] = a_[0]
                types_present = sorted(set(typ.values()))
                allowed = {}
                for fr in free:
                    trip = fr[2]
                    if trip is None: allowed[fr[0]] = None; continue
                    i, jj, k = trip
                    al = set()
                    for r1, r2 in itertools.combinations_with_replacement(types_present, 2):
                        nvec = Counter([r1, r2])
                        if any(nvec[r] > sum(1 for t in typ.values() if t == r) - (1 if r == fr[1] else 0) for r in nvec): continue
                        if r1 == r2 == fr[1]: continue
                        key = (trip, r1, r2)
                        if key not in memo:
                            fixed = {r: nvec.get(r, 0) for r in range(N) if r not in (i, jj)}
                            memo[key] = MB.feasible_exact(i, jj, k, comps, fixed)
                        if memo[key]: al.add((r1, r2))
                    allowed[fr[0]] = al
                    if not al: break
                if any(v is not None and not v for v in allowed.values()): blockok = False; break
                if not two_factor_exists(ids, typ, allowed, sis_of): blockok = False; break
            if blockok: survived = True; break
        if survived: out[(len(A), cc, n)] += 1
    return out, True

if __name__ == "__main__":
    t0 = time.time(); tot = Counter()
    for n in [int(a) for a in sys.argv[1:]] or range(3, 7):
        r, done = run(n, t0, 285)
        tot.update(r); print("n =", n, dict(sorted(r.items())), "" if done else "(TIME CAP: partial)"); sys.stdout.flush()
        if not done: break
    print("block-level v3 (A enumerated) survivors:", dict(sorted(tot.items())), "time %.0fs" % (time.time() - t0))
