"""Certified pair-level verdicts for full skeletons (general_model.build), in exact rational arithmetic.

certify(model) -> {'status': 'dead', 'level': 'LP' | 'branch', 'tree': T} or {'status': 'alive' | 'unknown', ...}
A tree node is either a leaf {'cert': [y_r as strings]} or a split {'var': j, 'le': k, 'lo': T1, 'hi': T2} meaning
x_j <= k (T1) and x_j >= k + 1 (T2).  A leaf certificate y proves that no x in the node's box satisfies lo <= A x <= hi:
    U(y) = sum_r (y_r hi_r if y_r > 0 else y_r lo_r)  <  L(y) = sum_j min(c_j lb_j, c_j ub_j),   c = A^T y.
check(model, tree) re-verifies everything from scratch (rebuild the model from the skeleton, then this function)."""
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog


def _phase1(A, lo, hi, lb, ub):
    import scipy.sparse as sp
    m, n = A.shape
    As = A.tocsr() if hasattr(A, 'tocsr') else sp.csr_matrix(A)
    I = sp.identity(m, format='csr')
    Aub = sp.vstack([sp.hstack([As, I, -I]), sp.hstack([-As, -I, I])]).tocsr()
    bub = np.concatenate([hi, -lo])
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    bounds = [(lb[j], ub[j]) for j in range(n)] + [(0, None)] * (2 * m)
    return linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method='highs')


def _cert_from(res, m, denom):
    dual = res.ineqlin.marginals
    yhi, ylo = -dual[:m], -dual[m:2 * m]
    return [Fraction(round(a * denom), denom) - Fraction(round(b * denom), denom) for a, b in zip(yhi, ylo)]


def _row(A, r):
    if hasattr(A, 'indptr'):
        s, e = A.indptr[r], A.indptr[r + 1]
        return zip(A.indices[s:e], A.data[s:e])
    nz = np.nonzero(A[r])[0]
    return zip(nz, A[r][nz])


def leaf_ok(A, lo, hi, lb, ub, y):
    U = Fraction(0)
    for r, yr in enumerate(y):
        if yr > 0:
            U += yr * int(hi[r])
        elif yr < 0:
            U += yr * int(lo[r])
    c = {}
    for r, yr in enumerate(y):
        if yr == 0:
            continue
        for j, a in _row(A, r):
            if a != 0:
                c[j] = c.get(j, Fraction(0)) + yr * int(round(a))
    L = sum(min(cj * int(lb[j]), cj * int(ub[j])) for j, cj in c.items() if cj != 0)
    return L > U


def certify(A, lo, hi, lb, ub, maxnodes=400, denom=10**6):
    stats = {'nodes': 0}

    def rec(lb, ub, depth):
        stats['nodes'] += 1
        if stats['nodes'] > maxnodes:
            return 'limit'
        if np.any(lb > ub):
            return {'cert': []}          # empty box: trivially infeasible (checked by check())
        m = A.shape[0]
        res = _phase1(A, lo, hi, lb, ub)
        if res.status == 0 and res.fun > 1e-7:
            for dn in (denom, 10**9, 10**12):
                y = _cert_from(res, m, dn)
                if leaf_ok(A, lo, hi, lb, ub, y):
                    return {'cert': [str(v) for v in y]}
            return 'rounding'
        if res.status != 0:
            return 'lp-error'
        x = res.x[:A.shape[1]]
        frac = [(abs(x[j] - round(x[j])), j) for j in range(len(x)) if lb[j] < ub[j] and abs(x[j] - round(x[j])) > 1e-6]
        if not frac:
            return ('integral', x)
        _, j = max(frac)
        k = int(np.floor(x[j]))
        out = {'var': int(j), 'le': k}
        ub1 = ub.copy(); ub1[j] = k
        lb2 = lb.copy(); lb2[j] = k + 1
        for key, (l_, u_) in (('lo', (lb, ub1)), ('hi', (lb2, ub))):
            sub = rec(l_, u_, depth + 1)
            if not isinstance(sub, dict):
                return sub
            out[key] = sub
        return out
    t = rec(lb.copy(), ub.copy(), 0)
    if isinstance(t, dict):
        return {'status': 'dead', 'level': 'LP' if 'cert' in t else 'branch', 'nodes': stats['nodes'], 'tree': t}
    if isinstance(t, tuple):
        return {'status': 'alive', 'nodes': stats['nodes']}
    return {'status': 'unknown', 'why': t, 'nodes': stats['nodes']}


def check(A, lo, hi, lb, ub, tree):
    if 'cert' in tree:
        if not tree['cert']:
            return bool(np.any(lb > ub))
        return leaf_ok(A, lo, hi, lb, ub, [Fraction(v) for v in tree['cert']])
    j, k = tree['var'], tree['le']
    ub1 = ub.copy(); ub1[j] = min(ub1[j], k)
    lb2 = lb.copy(); lb2[j] = max(lb2[j], k + 1)
    return check(A, lo, hi, lb, ub1, tree['lo']) and check(A, lo, hi, lb2, ub, tree['hi'])
