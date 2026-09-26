"""Pair-level model of an AD configuration (C2 note 41 section 4): every pair anchored ('one block' squares at its anchor) or
doubly-disjoint (dd).  Unknowns phi[W][P'] = number of neighbours of a representative w of orbit W among the 4 roots of pair P'.
Only proved facts are imposed:
  (own)   phi[W][P_W] = n_s(P_W) (0 for anchored pairs; n_s in {0,1} for dd pairs, same for both orbits);
  (F1)    for t in supp W: sum_{P' ni t} phi[W][P'] = 2;
  (zeros) at t in supp W: if P_W anchored at t, both through-t neighbours are in D(W): phi[W][t m'] = 2, other t-pairs 0;
          otherwise (W free at t) phi[W][t m] = 0 for m in In(t)  (39.4: M_t, F_t unions of components);
  (bal)   for k not in supp W and every union S of components of G_k: sum_{m in S} phi[W][k m] = |S minus supp W|
          (balance lemma, note 23 par. 8; components: the D-cycles at k inside In(k), and F_k);
  (D)     dd orbit: phi[W][pair of D(W)] in {2,3};
  (sym)   sum_s phi[(P,s)][P'] = sum_s phi[(P',s)][P]  (= e(P,P')/2).
Optional extras (flags): mu-budget 16.1, pair dichotomy.  Uses scipy milp (HiGHS) as a small feasibility test."""
import itertools, json, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 7
PAIRS = [(a, b) for a in range(N) for b in range(a + 1, N)]
PIDX = {p: i for i, p in enumerate(PAIRS)}
def pr(a, b): return (a, b) if a < b else (b, a)

def two_regular_graphs(vs):
    """all 2-regular multigraph-free 'D-structures' on vertex list vs (each vertex degree 2, cycles of length >= 2; a
    2-cycle = digon).  Returned as list of (cycles, orbit matching) where the matching pairs orbit (m, s) with (m', s')."""
    vs = list(vs); out = []
    def rec(rem, cyc):
        if not rem:
            out.append(list(cyc)); return
        f = rem[0]
        others = rem[1:]
        for L in range(1, len(others) + 1):
            for comb in itertools.combinations(others, L):
                if L == 1:
                    cyc.append((f,) + comb); rec([x for x in others if x not in comb], cyc); cyc.pop()
                else:
                    for perm in itertools.permutations(comb):
                        if perm[0] > perm[-1]: continue          # cycle direction
                        cyc.append((f,) + perm); rec([x for x in others if x not in perm], cyc); cyc.pop()
    rec(vs, [])
    return out

def cycle_matching(cyc):
    """orbits of the pairs {pivot, m}: named (m, 0), (m, 1).  Along the cycle m_0 m_1 ... m_{L-1}: (m_i, 1) -- (m_{i+1}, 0).
    For a digon (m, m'): (m,0)-(m',0) and (m,1)-(m',1) (naming WLOG)."""
    L = len(cyc); M = {}
    if L == 2:
        a, b = cyc
        M[(a, 0)] = (b, 0); M[(b, 0)] = (a, 0); M[(a, 1)] = (b, 1); M[(b, 1)] = (a, 1)
        return M
    for i in range(L):
        a, b = cyc[i], cyc[(i + 1) % L]
        M[(a, 1)] = (b, 0); M[(b, 0)] = (a, 1)
    return M

def build(In, dd, ddmatch, Dstruct, extra_split=None, mu_budget=False, dichotomy=False, fix=None):
    """In: dict block -> list of blocks anchored at it.  dd: list of dd pairs.  ddmatch: dict orbit -> orbit for dd orbits
    (orbit = (pair, s)).  Dstruct: dict pivot -> list of cycles (D-structure at the pivot on In(pivot)).
    extra_split: dict block -> list of blocks-sets partitioning F_k into components (default: F_k one union)."""
    orbits = [(p, s) for p in PAIRS for s in (0, 1)]
    OIDX = {o: i for i, o in enumerate(orbits)}
    anchor = {}
    for k in range(N):
        for m in In[k]: anchor[pr(k, m)] = k
    for p in dd: anchor[p] = None
    assert all(p in anchor for p in PAIRS)
    # D partner of each orbit
    D = {}
    for k, cycles in Dstruct.items():
        for cyc in cycles:
            M = cycle_matching(cyc)
            for (m, s), (m2, s2) in M.items():
                D[(pr(k, m), s)] = (pr(k, m2), s2)
    D.update(ddmatch)
    assert len(D) == 42 and all(D[D[o]] == o for o in orbits)
    comps = {}
    for k in range(N):
        cs = [set(c) for c in Dstruct.get(k, [])]
        Fk = set(range(N)) - {k} - set(In[k])
        if extra_split and k in extra_split: cs += [set(c) for c in extra_split[k]]
        elif Fk: cs.append(Fk)
        comps[k] = cs
    nv = 42 * 21
    ddl = list(dd)
    nsidx = {p: nv + i for i, p in enumerate(ddl)}
    nvar = nv + len(ddl)
    def v(o, p): return OIDX[o] * 21 + PIDX[p]
    rows, lo, hi = [], [], []
    def add(coefs, l, h):
        r = np.zeros(nvar)
        for j, c in coefs: r[j] += c
        rows.append(r); lo.append(l); hi.append(h)
    ub = np.full(nvar, 3.0); lb = np.zeros(nvar)
    for o in orbits:
        P, s = o; a, b = P
        # own pair
        if P in nsidx: add([(v(o, P), 1), (nsidx[P], -1)], 0, 0)
        else: ub[v(o, P)] = 0
        for t in (a, b):
            add([(v(o, pr(t, m)), 1) for m in range(N) if m != t], 2, 2)
            if anchor[P] == t:
                P2 = D[o][0]; m2 = P2[0] + P2[1] - t
                for m in range(N):
                    if m == t: continue
                    j = v(o, pr(t, m))
                    if m == m2: lb[j] = ub[j] = 2
                    else: ub[j] = 0
            else:
                for m in In[t]:
                    ub[v(o, pr(t, m))] = 0
        for k in range(N):
            if k in P: continue
            for S in comps[k]:
                add([(v(o, pr(k, m)), 1) for m in S], len(S - {a, b}), len(S - {a, b}))
        if anchor[P] is None:
            j = v(o, D[o][0]); lb[j] = max(lb[j], 2)
    for P, P2 in itertools.combinations(PAIRS, 2):
        add([(v((P, 0), P2), 1), (v((P, 1), P2), 1), (v((P2, 0), P), -1), (v((P2, 1), P), -1)], 0, 0)
    for p in ddl: ub[nsidx[p]] = 1
    if dichotomy:
        # pair dichotomy (= mu-budget 16.1): for every pair P,
        #   #{Z not in P, Z not a D-partner of an orbit of P : phi_Z(P) = 2} + 2 #{Z : phi_Z(P) = 3} = 2 - n_s(P);
        # phi_Z(P) = 3 only if Z is the D-partner of an orbit of P.  Indicators y_f (f = 0..3) per (Z, P), Z not in P.
        base_n = nvar
        yidx = {}
        for o in orbits:
            for P in PAIRS:
                if P == o[0]: continue
                yidx[(o, P)] = base_n; base_n += 4
        extra = base_n - nvar
        rows = [np.concatenate([r, np.zeros(extra)]) for r in rows]
        lb = np.concatenate([lb, np.zeros(extra)]); ub = np.concatenate([ub, np.ones(extra)])
        nvar0 = nvar; nvar = base_n
        def add2(coefs, l, h):
            r = np.zeros(nvar)
            for j, c in coefs: r[j] += c
            rows.append(r); lo.append(l); hi.append(h)
        for (o, P), y in yidx.items():
            add2([(y + f, 1) for f in range(4)], 1, 1)
            add2([(v(o, P), 1)] + [(y + f, -f) for f in range(4)], 0, 0)
            if D[o][0] != P: ub[y + 3] = 0
        for P in PAIRS:
            Dp = {D[(P, 0)], D[(P, 1)]}
            coefs = []
            for o in orbits:
                if o[0] == P: continue
                if o not in Dp: coefs.append((yidx[(o, P)] + 2, 1))
                coefs.append((yidx[(o, P)] + 3, 2))
            if P in nsidx: coefs.append((nsidx[P], 1)); add2(coefs, 2, 2)
            else: add2(coefs, 2, 2)
    if fix:
        for (o, p), val in fix.items(): lb[v(o, p)] = ub[v(o, p)] = val
    A = np.array([r if len(r) == nvar else np.concatenate([r, np.zeros(nvar - len(r))]) for r in rows])
    return A, np.array(lo), np.array(hi), lb, ub, nvar, v, nsidx, orbits

def feasible(In, dd, ddmatch, Dstruct, **kw):
    A, lo, hi, lb, ub, nvar, v, nsidx, orbits = build(In, dd, ddmatch, Dstruct, **kw)
    if np.any(lb > ub): return None
    res = milp(c=np.zeros(nvar), constraints=[LinearConstraint(A, lo, hi)], integrality=np.ones(nvar),
               bounds=Bounds(lb, ub), options={'time_limit': 60})
    if res.status == 0:
        return res.x, v, nsidx, orbits
    return None if res.status == 2 else 'unknown(%d)' % res.status

def parse_survivors(path):
    reps = []
    cur = None
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line.startswith('rep '):
            cur = {'rep': int(line.split()[1])}; reps.append(cur)
        elif line.startswith('In(k):'):
            cur['In'] = {int(k): v for k, v in eval(line[len('In(k):'):]).items()}
    return reps
