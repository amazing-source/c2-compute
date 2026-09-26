"""Local model at a FULL block c (every orbit through c pivots at c: tau_c = eta_c = m, the mate matching of N(c+)), and the
q:7 rigid-row test of C1 (notes/131 par. 7) on its solutions.

Labels v = 2i + s (i = 0..5 the six other blocks, s = 0 for +, 1 for -), pi(v) = v ^ 1.  Far roots z = {a, a'} (labels of two
different blocks), sigma z = {pi a, pi a'}.  Phi(z) = {v : z ~ (c+, v)} (two labels).  Facts used (derivations in
NOTE_S29_BLOC_PLEIN_LIGNES_RIGIDES.md):
  m      : gauge-fixed from the D-structure at c (a D-cycle p1..pl gives the m-edges p_j+ -- p_{j+1}-);
  (i)    Phi is a bijection far roots -> non-m pairs;
  (ii)   #{z containing a : b in Phi(z)} = K2[a,b] = 2 - [a=b] - [b=pi a] - [b=m(a)] - [b=m(pi a)];
  (iii)  #{z : b in Phi(z), d in Phi(sigma z)} = K3[b,d] = 2 - [d=m(b)] - [d=pi b] - 2[d=b];
  (iv)   for a in z: not (a in Phi(z) and pi a in Phi(sigma z)), not (pi a in Phi(z) and a in Phi(sigma z))  (unique neighbour
         of z through a and through pi a);
  (v)    balance on each D-cycle C: |blocks(z) & C| + |Phi(z) & lab(C)| + |Phi(sigma z) & lab(C)| = |C|;
  (vi)   optional (AD residues): a far pair anchored at its block e has Phi(z), Phi(sigma z) disjoint from lab(e).
Rigid-row test (mod 7): Z = 3 delta + 5 B^T B + st^T (5I + 4 P_pi) st, st = s + 2(1_z - 1_{sigma z}); f rigid iff Z_ff = 0;
then T_{f,f'} = Z_{f,f'} for every far f', and at a full block the row must have exactly 6 entries +-1 and 23 zeros
(no shadow cancellation: |supp s_f| = 4, and 10 - 4 = 6 far neighbours at q = 1)."""
import itertools, json, sys, time, random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

P7 = 7
L = list(range(12))
pi = [v ^ 1 for v in L]


def mates_from_D(cycles):
    """cycles: list of block cycles (tuples of block indices 0..5), a digon given as (p, q)."""
    m = [None] * 12
    for cyc in cycles:
        l = len(cyc)
        for j in range(l):
            p, q = cyc[j], cyc[(j + 1) % l]
            a, b = 2 * p, 2 * q + 1
            if l == 2 and j == 1:
                a, b = 2 * p, 2 * q + 1          # digon: p+ -- q-, q+ -- p-
            m[a], m[b] = b, a
    assert all(x is not None for x in m) and all(m[m[v]] == v for v in L) and all(m[v] != pi[v] for v in L)
    return m


def K2K3(m):
    K2 = np.full((12, 12), 2, int)
    K3 = np.full((12, 12), 2, int)
    for a in L:
        K2[a, a] -= 1; K2[a, pi[a]] -= 1; K2[a, m[a]] -= 1; K2[a, m[pi[a]]] -= 1
        K3[a, m[a]] -= 1; K3[a, pi[a]] -= 1; K3[a, a] -= 2
    assert (K2 >= 0).all() and (K3 >= 0).all()
    return K2, K3


def far_roots():
    far = [z for z in itertools.combinations(L, 2) if z[0] // 2 != z[1] // 2]
    reps, seen = [], set()
    for z in far:
        if z in seen:
            continue
        sz = tuple(sorted((pi[z[0]], pi[z[1]])))
        seen |= {z, sz}
        reps.append((z, sz))
    return far, reps                                   # 60 far roots, 30 sigma-pairs (z, sigma z)


def lab(blocks):
    return {2 * b + s for b in blocks for s in (0, 1)}


def admissible(z, sz, e, e2, m, cycles, anchor):
    if set(e) & set(e2):
        return False
    for a in z:                                       # (iv)
        if (a in e and pi[a] in e2) or (pi[a] in e and a in e2):
            return False
    for C in cycles:                                  # (v)
        lc = lab(C)
        if len({a // 2 for a in z} & set(C)) + len(set(e) & lc) + len(set(e2) & lc) != len(C):
            return False
    if anchor is not None:                            # (vi)
        la = lab([anchor])
        if set(e) & la or set(e2) & la:
            return False
    return True


def build(m, cycles, anchors, rng=None):
    """anchors: dict frozenset({i, j}) -> anchor block or None (dd / unknown)."""
    K2, K3 = K2K3(m)
    far, reps = far_roots()
    nonm = [e for e in itertools.combinations(L, 2) if m[e[0]] != e[1]]
    var = []
    for Zi, (z, sz) in enumerate(reps):
        anc = anchors.get(frozenset((z[0] // 2, z[1] // 2)))
        for e in nonm:
            for e2 in nonm:
                if e != e2 and admissible(z, sz, e, e2, m, cycles, anc):
                    var.append((Zi, e, e2))
    nv = len(var)
    ri, ci, vv, lo = [], [], [], []
    r = [0]

    def row(entries, val):
        for j, c in entries:
            ri.append(r[0]); ci.append(j); vv.append(c)
        lo.append(val); r[0] += 1
    byZ, byE, acc2, acc3 = {}, {}, {}, {}
    for j, (Zi, e, e2) in enumerate(var):
        z, sz = reps[Zi]
        byZ.setdefault(Zi, []).append(j)
        byE.setdefault(e, []).append(j)
        byE.setdefault(e2, []).append(j)
        for a in z:
            for b in e:
                acc2.setdefault((a, b), []).append(j)
        for a in sz:
            for b in e2:
                acc2.setdefault((a, b), []).append(j)
        for b in e:
            for d in e2:
                acc3.setdefault((b, d), []).append(j)
        for b in e2:
            for d in e:
                acc3.setdefault((b, d), []).append(j)
    for Zi in range(len(reps)):
        row([(j, 1) for j in byZ.get(Zi, [])], 1)
    for e in nonm:
        row([(j, 1) for j in byE.get(e, [])], 1)
    for a in L:
        for b in L:
            row([(j, 1) for j in acc2.get((a, b), [])], int(K2[a, b]))
    for b in L:
        for d in L:
            row([(j, 1) for j in acc3.get((b, d), [])], int(K3[b, d]))
    A = sp.csr_matrix((np.array(vv, float), (np.array(ri), np.array(ci))), shape=(r[0], nv))
    return var, reps, A, np.array(lo, float)


def solve(var, A, lo, tl, seed=None, forbid=()):
    nv = len(var)
    c = np.zeros(nv) if seed is None else np.random.default_rng(seed).standard_normal(nv)
    cons = [LinearConstraint(A, lo, lo)]
    for sol in forbid:                                 # no-good cuts: differ from previous solutions
        row = np.zeros(nv); row[list(sol)] = 1
        cons.append(LinearConstraint(row[None, :], -np.inf, len(sol) - 1))
    res = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1), options={'time_limit': tl})
    if res.status == 0:
        return 'FEASIBLE', sorted(np.nonzero(res.x > 0.5)[0].tolist())
    return {2: 'INFEASIBLE'}.get(res.status, 'unknown(%d)' % res.status), None


def zmatrix(var, reps, chosen):
    """Z (30 x 30 mod 7) with rows/columns indexed by the sigma-pairs, representative z = reps[Zi][0]."""
    W = 5 * np.eye(12, dtype=int)
    for v in L:
        W[v, pi[v]] = 4
    n = len(reps)
    ST = np.zeros((n, 12), int)
    B = np.zeros((n, 6), int)
    for j in chosen:
        Zi, e, e2 = var[j]
        z, sz = reps[Zi]
        s = np.zeros(12, int)
        s[list(e)] += 1; s[list(e2)] -= 1
        d = np.zeros(12, int)
        d[list(z)] += 1; d[list(sz)] -= 1
        ST[Zi] = s + 2 * d
        for a in z:
            B[Zi, a // 2] = 1 if a % 2 == 0 else -1
    Z = (3 * np.eye(n, dtype=int) + 5 * B @ B.T + ST @ W @ ST.T) % P7
    return Z


def rigid_report(Z, reps, anchors):
    n = Z.shape[0]
    rig = [f for f in range(n) if Z[f, f] == 0]
    bad = []
    for f in rig:
        row = [int(Z[f, g]) for g in range(n) if g != f]
        nz = [x for x in row if x]
        ok = len(nz) == 6 and all(x in (1, 6) for x in nz)
        z = reps[f][0]
        sis = next(g for g in range(n) if g != f and {reps[g][0][0] // 2, reps[g][0][1] // 2} == {z[0] // 2, z[1] // 2})
        anc = anchors.get(frozenset((z[0] // 2, z[1] // 2)))
        if anc is not None and Z[f, sis] != 0:
            ok = False
        if not ok:
            bad.append((f, len(nz), sorted(set(nz))))
    return rig, bad


if __name__ == '__main__':
    cfg = json.loads(sys.argv[1])
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    nsol = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cycles = [tuple(c) for c in cfg['cycles']]
    anchors = {frozenset(map(int, k.split(','))): v for k, v in cfg.get('anchors', {}).items()}
    m = mates_from_D(cycles)
    t0 = time.time()
    var, reps, A, lo = build(m, cycles, anchors)
    print('D-cycles', cycles, '| anchored far pairs', len([1 for v in anchors.values() if v is not None]),
          '| variables', len(var), '| rows', A.shape[0], '| build %.0fs' % (time.time() - t0), flush=True)
    found = []
    for k in range(nsol):
        st, chosen = solve(var, A, lo, tl, seed=k, forbid=found)
        print('solve', k, st, '%.0fs' % (time.time() - t0), flush=True)
        if chosen is None:
            break
        found.append(chosen)
        Z = zmatrix(var, reps, chosen)
        rig, bad = rigid_report(Z, reps, anchors)
        print('  rigid far orbits', len(rig), rig, '| rigid rows violating (6 entries +-1, sister 0):', len(bad), bad[:6],
              flush=True)
