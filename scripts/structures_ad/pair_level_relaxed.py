"""Pair-level AD model with the FREE structures encoded by partner variables instead of enumerated.
  - blocks k with d_k >= 4 (D-structure at k unknown): each orbit W of a pair km anchored at k gets a partner pair km2
    (binary part[W, km2], sum 1), phi_W(km2) = 2 part, consistency sum_s part[(km,s), km2] = sum_s part[(km2,s), km];
    balance at k only on the union M_k = In(k) and on F_k (valid relaxation: both are unions of components);
  - dd pairs: each dd orbit W gets a partner dd pair P2 disjoint from its pair (binary), 2 part <= phi_W(P2),
    consistency sum_s part[(P,s), P2] = sum_s part[(P2,s), P];
  - dichotomy with the partner indicator: sum_Z (y2[Z,P] - part[Z,P] + 3 y3[Z,P]) = 2 - n_s(P), y3 <= part,
    y2 + y3 >= part (part[Z,P] = 1 iff D(Z) lies in P; constant 1 for fixed partners).
Blocks with d = 2 / 3 keep their forced digon / triangle (known components).  Infeasible => dead for every choice."""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pair_level import N, PAIRS, PIDX, pr, cycle_matching


def build_relaxed(In, dd, dichotomy=True, fixedDs=None):
    orbits = [(p, s) for p in PAIRS for s in (0, 1)]
    anchor = {}
    for k in range(N):
        for m in In[k]:
            anchor[pr(k, m)] = k
    for p in dd:
        anchor[p] = None
    assert all(p in anchor for p in PAIRS)
    fixedD = {}
    freeblocks = []
    fixedDs = fixedDs or {}
    compsM = {}
    for k in range(N):
        d = len(In[k])
        if d in (2, 3):
            for (m, s), (m2, s2) in cycle_matching(tuple(In[k])).items():
                fixedD[(pr(k, m), s)] = (pr(k, m2), s2)
            compsM[k] = [set(In[k])]
        elif d >= 4 and k in fixedDs:
            for cyc in fixedDs[k]:
                for (m, s), (m2, s2) in cycle_matching(tuple(cyc)).items():
                    fixedD[(pr(k, m), s)] = (pr(k, m2), s2)
            compsM[k] = [set(c) for c in fixedDs[k]]
        elif d >= 4:
            freeblocks.append(k)
            compsM[k] = [set(In[k])]
        elif d == 1:
            return None
        else:
            compsM[k] = []
    cnt = [0]

    def new(n=1):
        a = cnt[0]
        cnt[0] += n
        return a
    phi = {(o, P): new() for o in orbits for P in PAIRS}
    ns = {p: new() for p in dd}
    part = {}
    for k in freeblocks:
        for m in In[k]:
            for s in (0, 1):
                for m2 in In[k]:
                    if m2 != m:
                        part[((pr(k, m), s), pr(k, m2))] = new()
    for P in dd:
        for s in (0, 1):
            for P2 in dd:
                if not (set(P) & set(P2)):
                    part[((P, s), P2)] = new()
    for o, (P2, s2) in fixedD.items():
        part[(o, P2)] = None
    y = {}
    if dichotomy:
        for o in orbits:
            for P in PAIRS:
                if P != o[0]:
                    y[(o, P)] = new(4)
    nvar = cnt[0]
    rows, lo, hi = [], [], []
    lb = np.zeros(nvar)
    ub = np.full(nvar, 3.0)

    def add(coefs, l, h):
        rows.append(coefs)
        lo.append(l)
        hi.append(h)
    for o in orbits:
        P, s = o
        a, b = P
        if P in ns:
            add([(phi[(o, P)], 1), (ns[P], -1)], 0, 0)
        else:
            ub[phi[(o, P)]] = 0
        for t in (a, b):
            add([(phi[(o, pr(t, m))], 1) for m in range(N) if m != t], 2, 2)
            other = P[0] + P[1] - t
            if anchor[P] == t:
                if o in fixedD:
                    P2 = fixedD[o][0]
                    m2 = P2[0] + P2[1] - t
                    for m in range(N):
                        if m == t:
                            continue
                        j = phi[(o, pr(t, m))]
                        if m == m2:
                            lb[j] = ub[j] = 2
                        else:
                            ub[j] = 0
                else:
                    add([(part[(o, pr(t, m2))], 1) for m2 in In[t] if m2 != other], 1, 1)
                    for m in range(N):
                        if m == t:
                            continue
                        j = phi[(o, pr(t, m))]
                        if m in In[t] and m != other:
                            add([(j, 1), (part[(o, pr(t, m))], -2)], 0, 0)
                        else:
                            ub[j] = 0
            else:
                for m in In[t]:
                    ub[phi[(o, pr(t, m))]] = 0
        for k in range(N):
            if k in P:
                continue
            comps = list(compsM[k])
            Fk = set(range(N)) - {k} - set(In[k])
            if Fk:
                comps.append(Fk)
            for S in comps:
                add([(phi[(o, pr(k, m))], 1) for m in S], len(S - {a, b}), len(S - {a, b}))
        if anchor[P] is None:
            cands = [P2 for P2 in dd if not (set(P) & set(P2))]
            if not cands:
                return 'no dd partner'
            add([(part[(o, P2)], 1) for P2 in cands], 1, 1)
            for P2 in cands:
                add([(phi[(o, P2)], 1), (part[(o, P2)], -2)], 0, 3)
    for k in freeblocks:
        for m, m2 in itertools.combinations(In[k], 2):
            add([(part[((pr(k, m), s), pr(k, m2))], 1) for s in (0, 1)]
                + [(part[((pr(k, m2), s), pr(k, m))], -1) for s in (0, 1)], 0, 0)
    for P, P2 in itertools.combinations(dd, 2):
        if set(P) & set(P2):
            continue
        add([(part[((P, s), P2)], 1) for s in (0, 1)] + [(part[((P2, s), P)], -1) for s in (0, 1)], 0, 0)
    for P, P2 in itertools.combinations(PAIRS, 2):
        add([(phi[((P, 0), P2)], 1), (phi[((P, 1), P2)], 1), (phi[((P2, 0), P)], -1), (phi[((P2, 1), P)], -1)], 0, 0)
    for p in dd:
        ub[ns[p]] = 1
    for key, j in part.items():
        if j is not None:
            ub[j] = 1
    if dichotomy:
        for (o, P), yy in y.items():
            for f in range(4):
                ub[yy + f] = 1
            add([(yy + f, 1) for f in range(4)], 1, 1)
            add([(phi[(o, P)], 1)] + [(yy + f, -f) for f in range(4)], 0, 0)
            pk = part.get((o, P), 'none')
            if pk == 'none':
                ub[yy + 3] = 0
            elif pk is None:
                add([(yy + 2, 1), (yy + 3, 1)], 1, 1)
            else:
                add([(yy + 3, 1), (pk, -1)], -1, 0)
                add([(yy + 2, 1), (yy + 3, 1), (pk, -1)], 0, 1)
        for P in PAIRS:
            coefs = []
            const = 0
            for o in orbits:
                if o[0] == P:
                    continue
                yy = y[(o, P)]
                pk = part.get((o, P), 'none')
                coefs += [(yy + 2, 1), (yy + 3, 3)]
                if pk is None:
                    const += 1
                elif pk != 'none':
                    coefs.append((pk, -1))
            if P in ns:
                coefs.append((ns[P], 1))
            add(coefs, 2 + const, 2 + const)
    A = np.zeros((len(rows), nvar))
    for r, cs in enumerate(rows):
        for j, c in cs:
            A[r, j] += c
    return A, np.array(lo, float), np.array(hi, float), lb, ub, nvar


def feasible_relaxed(In, dd, dichotomy=True, time_limit=120, fixedDs=None):
    out = build_relaxed(In, dd, dichotomy, fixedDs)
    if out is None or isinstance(out, str):
        return None
    A, lo, hi, lb, ub, nvar = out
    if np.any(lb > ub):
        return None
    res = milp(c=np.zeros(nvar), constraints=[LinearConstraint(A, lo, hi)], integrality=np.ones(nvar),
               bounds=Bounds(lb, ub), options={'time_limit': time_limit})
    if res.status == 0:
        return res.x
    if res.status == 2:
        return None
    return 'unknown(%d)' % res.status
