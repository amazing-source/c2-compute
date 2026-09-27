"""Step 1 of the two-step formulation of the local system at a block c (C2 notation mu = mates, tau = pi o h):
(iii) says exactly: iota := Phi o sigma o Phi^{-1} is a fixed-point-free involution on the 60 non-mu pairs such that, for
labels b != d,
    #{iota-pairs {e, e'} : b in e, d in e'  or  b in e', d in e} = K3[b,d]        (and K3[b,b] = #{e : b in e and b in iota e}).
At a full block (mu = tau = m): K3[b,b] = 0, so e and iota(e) are disjoint, and the 30 quadruples e u iota(e) cover every
pair of labels 3 times, pi-pairs twice, m-pairs once (the 'near-design').
Step 2: Phi = a bijection sigma-pairs -> iota-pairs with orientation, subject to (ii) (and (iv), (v) at a full block).
This script: CP-SAT for step 1 alone (optionally enumerating several iota), then step 2 as a small MILP for each iota found.
Usage: python iota_design.py cfg.json n_iota time_per_solve"""
import itertools, json, sys, time
import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

L = list(range(12))
PI = [v ^ 1 for v in L]


def K2K3(mu, tau):
    K2 = np.full((12, 12), 2, int); K3 = np.full((12, 12), 2, int)
    for a in L:
        K2[a, a] -= 1; K2[a, PI[a]] -= 1; K2[a, mu[a]] -= 1; K2[a, tau[PI[a]]] -= 1
    for b in L:
        K3[b, PI[b]] -= 1; K3[b, tau[b]] -= 1; K3[b, tau[mu[b]]] -= 1; K3[b, mu[tau[b]]] -= 1
    return K2, K3


def step1(mu, tau, n_sol, tl, forbid=(), c1lemma=None):
    K2, K3 = K2K3(mu, tau)
    edges = [e for e in itertools.combinations(L, 2) if mu[e[0]] != e[1]]
    E = len(edges)
    md = cp_model.CpModel()
    P = {}
    for i in range(E):
        for j in range(i + 1, E):
            e, f = edges[i], edges[j]
            common = set(e) & set(f)
            # contribution to K3 diagonal: each common label b gets 2 (e and iota e both contain b)
            if any(K3[b, b] < 2 for b in common):
                continue
            P[i, j] = md.NewBoolVar('p%d_%d' % (i, j))
    for i in range(E):
        md.AddExactlyOne(P[min(i, j), max(i, j)] for j in range(E) if j != i and (min(i, j), max(i, j)) in P)
    acc = {}
    for (i, j), v in P.items():
        e, f = edges[i], edges[j]
        for b in e:
            for d in f:
                key = (min(b, d), max(b, d)) if b != d else (b, b)
                acc.setdefault(key, []).append(v)
    for b in L:
        for d in L:
            if d < b:
                continue
            target = int(K3[b, d]) if b != d else int(K3[b, b]) // 2
            md.Add(sum(acc.get((b, d), [])) == target)
    for sol in forbid:
        md.AddBoolOr([P[k].Not() for k in sol])
    if c1lemma is not None:                      # C1's lemma at a full block: c1lemma = list of (special labels Sp_p, bound)
        for Sp, bound in c1lemma:
            big = []
            for (i, j), v in P.items():
                k = len((set(edges[i]) | set(edges[j])) & Sp)
                if k == 4:
                    md.Add(v == 0)
                elif k == 3:
                    big.append(v)
            md.Add(sum(big) <= bound)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 2
    st = s.Solve(md)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return s.StatusName(st), None, edges
    sol = [k for k, v in P.items() if s.Value(v)]
    return 'FEASIBLE', sol, edges


def step2(mu, tau, iota_pairs, edges, tl=60, extras=None):
    """Phi: sigma-pairs (z, sigma z) -> iota-pairs (e, e') oriented; (ii) exact.  MILP."""
    K2, K3 = K2K3(mu, tau)
    far = [z for z in itertools.combinations(L, 2) if z[0] // 2 != z[1] // 2]
    reps, seen = [], set()
    for z in far:
        if z in seen:
            continue
        sz = tuple(sorted((PI[z[0]], PI[z[1]])))
        seen |= {z, sz}
        reps.append((z, sz))
    var = []
    for Zi, (z, sz) in enumerate(reps):
        for Pi_, (i, j) in enumerate(iota_pairs):
            for (e, f) in ((edges[i], edges[j]), (edges[j], edges[i])):
                if extras is not None and not extras(z, sz, e, f):
                    continue
                var.append((Zi, Pi_, e, f))
    nv = len(var)
    ri, ci, vv, lo = [], [], [], []
    r = [0]

    def row(ent, val):
        for j, c in ent:
            ri.append(r[0]); ci.append(j); vv.append(c)
        lo.append(val); r[0] += 1
    byZ, byP, acc = {}, {}, {}
    for k, (Zi, Pi_, e, f) in enumerate(var):
        byZ.setdefault(Zi, []).append(k); byP.setdefault(Pi_, []).append(k)
        z, sz = reps[Zi]
        for a in z:
            for b in e:
                acc.setdefault((a, b), []).append(k)
        for a in sz:
            for b in f:
                acc.setdefault((a, b), []).append(k)
    for Zi in range(30):
        row([(k, 1) for k in byZ.get(Zi, [])], 1)
    for Pi_ in range(30):
        row([(k, 1) for k in byP.get(Pi_, [])], 1)
    for a in L:
        for b in L:
            row([(k, 1) for k in acc.get((a, b), [])], int(K2[a, b]))
    A = sp.csr_matrix((np.array(vv, float), (np.array(ri), np.array(ci))), shape=(r[0], nv))
    res = milp(np.zeros(nv), constraints=[LinearConstraint(A, lo, lo)], integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={'time_limit': tl})
    return {0: 'FEASIBLE', 2: 'INFEASIBLE'}.get(res.status, 'unknown(%d)' % res.status), nv


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    n_iota = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 60
    if 'cycles' in cfg:
        from fullblock_rigid import mates_from_D
        m = mates_from_D([tuple(c) for c in cfg['cycles']])
        mu = tau = m
    else:
        mu, tau = cfg['mu'], cfg['tau']
    found = []
    t0 = time.time()
    for k in range(n_iota):
        st, sol, edges = step1(mu, tau, 1, tl, forbid=found)
        print('step1 #%d' % k, st, '%.1fs' % (time.time() - t0), flush=True)
        if sol is None:
            break
        found.append(sol)
        st2, nv = step2(mu, tau, sol, edges, tl=tl)
        print('   step2 (ii) for this iota:', st2, '| vars', nv, '%.1fs' % (time.time() - t0), flush=True)
