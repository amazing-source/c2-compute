# Stronger SOUND filter for M1 survivors: full single-orbit balance profiles (note 27.1), on top of 28.x.
# For each orbit x = (i, j, k) (pivot i, free block j, third block k), unknowns phi(P) for pairs P not containing i
# (the 15 pairs of [7]\{i}), except the own pair {i,j}:
#   - pairs {j,r}: n_r in [0, number of free orbits of {j,r} at j]; sum_r n_r = 2 (R_j-neighbours of x)
#   - other pairs: phi in {0,1,2}
#   - F1: phi({i,r}) = 0 for r != k, phi({i,k}) = 2 (these are constants, entered in the block sums)
# Constraints: for each block c not in {i,j} and each component C of G_c, sum_{r in C} phi({c,r}) = b_C - |{i,j} & C|.
#   Components used only where they are CERTAIN: non-T blocks (closed cycle on In + the rest), T-blocks with a closed linked
#   pair (alpha: In + rest). Elsewhere (beta T-blocks, A) only the total (= 4 roots) is used. Sound, conservative.
# A skeleton (orientation) survives if some choice of D-structures passes 28.x AND every orbit's profile is feasible.
import itertools, sys, time
from collections import Counter
N = 7

def feasible(i, j, k, comps, freecount):
    # comps: dict c -> list of (set_of_blocks) partitioning [7]\{c}; freecount[r] = # free orbits of {j,r} at j (r != i)
    others = [r for r in range(N) if r not in (i, j)]
    var_pairs = [frozenset((j, r)) for r in others] + [frozenset(p) for p in itertools.combinations(others, 2)]
    dom = {}
    for P in var_pairs:
        if j in P:
            r = next(iter(P - {j})); dom[P] = range(0, min(2, freecount.get(r, 0)) + 1)
        else:
            dom[P] = range(0, 3)
    def const(c, r):
        P = frozenset((c, r))
        if i in P:
            return 2 if P == frozenset((i, k)) else 0
        return None
    cons = []   # (list of pairs, constant part, target)
    for c in others:
        for C in comps[c]:
            tgt = len(C) - len({i, j} & C)
            cst = 0; ps = []
            for r in C:
                v = const(c, r)
                if v is not None: cst += v
                elif frozenset((c, r)) == frozenset((i, j)): pass
                else: ps.append(frozenset((c, r)))
            cons.append((ps, cst, tgt))
    nj = [frozenset((j, r)) for r in others]
    cons.append((nj, 0, 2))
    far = [P for P in var_pairs if j not in P]
    cons.append((far, 2 if k != j else 0, 8))   # far pairs of x: those avoiding {i,j}; includes {i,k}? no ({i,k} contains i)
    # note: far pairs avoiding {i,j} = var pairs without j (10) ; sum must be 8
    cons[-1] = (far, 0, 8)
    order = sorted(var_pairs, key=lambda P: -sum(P in c[0] for c in cons))
    val = {}
    def ok_partial():
        for ps, cst, tgt in cons:
            s = cst; free = 0; mx = 0
            for P in ps:
                if P in val: s += val[P]
                else: mx += max(dom[P])
            if s > tgt or s + mx < tgt: return False
        return True
    def dfs(t):
        if t == len(order): return True
        P = order[t]
        for v in dom[P]:
            val[P] = v
            if ok_partial() and dfs(t + 1): return True
        del val[P]; return False
    return dfs(0)

def components_for(c, choice, In, Tw, n, d):
    rest = [r for r in range(N) if r != c]
    ch = choice[c]
    if ch[0] == 'C' and len(ch[1]) in (2, 3):
        S = set(ch[1]); return [S, set(rest) - S]
    if ch[0] == 'L':
        S = set(ch[1]); return [S, set(rest) - S]
    return [set(rest)]

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
            if c == Ab: opts.append([('A', None)]); continue           # A kept generic (conservative)
            if c >= n: opts.append([('C', tuple(sorted(In[c])))]); continue
            I = In[c]
            if d[c] == 0: opts.append([('P', ())])
            elif d[c] == 1: opts.append([('P', (I[0],))])
            else: opts.append([('L', tuple(sorted(I))), ('P', (I[0], I[1])), ('P', (I[1], I[0]))])
        cc = sum(1 for c in range(n, N) if d[c] == 3)
        survived = False
        for choice in itertools.product(*opts):
            orbits = []; linked = {}; cert42 = {}; cert33 = {}
            for c in range(N):
                ch = choice[c]
                if ch[0] == 'A': continue
                if ch[0] == 'C':
                    I = list(ch[1])
                    if len(I) == 2:
                        a, b = I; orbits += [(c, a, b), (c, b, a)]; linked[c] = set(I); cert42[c] = set(I)
                    elif len(I) == 3:
                        a, b, e = I
                        for (x, y) in ((a, b), (b, e), (e, a)): orbits += [(c, x, y), (c, y, x)]
                        cert33[c] = set(I)
                    continue
                P1, P2 = Tw[c]
                if ch[0] == 'L':
                    a, b = ch[1]; orbits += [(c, a, b), (c, b, a), (c, P1, P2), (c, P2, P1)]
                    linked[c] = {a, b}; cert42[c] = {a, b}
                else:
                    seq = [P1] + list(ch[1]) + [P2]
                    for x, y in zip(seq, seq[1:]): orbits += [(c, x, y), (c, y, x)]
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
            allok = True
            for (i, j, k) in set(orbits):
                # free orbits of {j,r} at j: 2 if r in Out(j) (anchored at r), 1 if {j,r} twisted, 0 if r in In(j)
                fc = {}
                for r in range(N):
                    if r in (i, j): continue
                    if frozenset((j, r)) in tw: fc[r] = 1
                    elif r in In[j]: fc[r] = 0
                    else: fc[r] = 2
                if not feasible(i, j, k, comps, fc): allok = False; break
            if allok: survived = True; break
        if survived: out[(len(A), cc, n)] += 1
    return out, True

t0 = time.time(); tot = Counter()
for n in range(3, 8):
    r, done = run(n, t0, 280)
    tot.update(r); print("n =", n, dict(sorted(r.items())), "" if done else "(TIME CAP: partial)"); sys.stdout.flush()
    if not done: break
print("profile-filtered survivors (labelled orientations):", dict(sorted(tot.items())), "time %.0fs" % (time.time() - t0))
