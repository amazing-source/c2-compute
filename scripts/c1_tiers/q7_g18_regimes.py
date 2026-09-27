"""g- = 18: regimes with torsion (t >= 1), from g = 18 + t - 2 r0 + rho and Witt (notes/132 sec. 1):
  t = 1: r0 = 1, rho = 1                 -> full balance; T_FF = Z + lam k k^T, so Z_ff in -lam {0, squares} (lam in {1, 3})
  t = 2: (r0, rho) = (2, 2)               -> full balance
         (1, 0), split reduced ball only  -> balance on a line W of the torsion (8 lines)
  t = 3: (2, 1) (the (3, 3) regime dies by full balance, notes/132) -> balance on a plane W (57 planes)
  t = 4: (3, 2) ((4, 4) dies by full balance; (2, 0) excluded by Witt: all t = 4 classes are non-split)
         -> balance on a hyperplane W (400)
For each class and each choice (lam or W): unary filters, then the local model (i)+(ii)+(iii) (small ILP).
A class survives if some choice survives."""
import sys, json, time, itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
from q7_g16_test import (classes_t_ge, ball_disc, split, ball, kernel_mod, ginv_sym, orbit_options, K2, K3, FAR, L, P7)

SQ = {1, 2, 4}


def z_vector():
    v = np.zeros(19, dtype=np.int64)
    v[6] = 3
    v[7:] = 1
    return v


def torsion_basis(Pm):
    """basis {phi_1..phi_t} of ker P modulo z_c (as 19-vectors)."""
    K = kernel_mod(Pm.tolist())
    z = z_vector()
    assert not np.any((Pm @ z) % P7)
    basis = [z]
    for v in K:
        M = np.array(basis + [v]) % P7
        if np.linalg.matrix_rank(M) > len(basis):          # rank over Q is a proxy; confirm mod 7 below
            pass
        from q7_g16_test import rank_mod
        if rank_mod(M.tolist()) > len(basis):
            basis.append(v % P7)
    return basis[1:]


def proj_points(t):
    pts = []
    for v in itertools.product(range(P7), repeat=t):
        if not any(v):
            continue
        k = next(x for x in v if x)
        if k == 1:
            pts.append(np.array(v, dtype=np.int64))
    return pts


_CACHE = {}


def class_data(tau, eta, tb):
    key = (tuple(tau), tuple(eta))
    if key not in _CACHE:
        Pm = ball(tau, eta)
        Pp = ginv_sym(Pm)
        Tb = np.array(tb) % P7
        zv = z_vector()
        data = []
        for (opts, Xc) in orbit_options(tau, eta, K2(tau, eta)):
            base = ((zv @ Xc) % P7 == 0)
            beta = (Tb @ Xc) % P7
            zz = (3 + np.einsum('ij,ij->j', Xc, (Pp @ Xc) % P7)) % P7
            data.append((opts, Xc, base, beta, zz))
        _CACHE.clear()
        _CACHE[key] = data
    return _CACHE[key]


def filtered(tau, eta, tb, pred=None, zset=None):
    """options balanced for z_c and satisfying pred(beta) (beta = (x_f . phi_i)_i, t x n), and Z_ff in zset if given
    (Z_ff is only meaningful when pred imposes the full balance)."""
    res = []
    for (opts, Xc, base, beta, zz) in class_data(tau, eta, tb):
        keep = base.copy()
        if pred is not None:
            keep &= pred(beta)
        if zset is not None:
            keep &= np.isin(zz, list(zset))
        idx = np.nonzero(keep)[0]
        res.append(([opts[j] for j in idx], Xc[:, idx]))
    return res


def local_ilp(tau, eta, res, tl=30):
    if any(len(r[0]) == 0 for r in res):
        return 'dead-unary'
    k2, k3 = K2(tau, eta), K3(tau, eta)
    var = [(fi, j) for fi, (opts, Xc) in enumerate(res) for j in range(len(opts))]
    vid = {v: i for i, v in enumerate(var)}
    rows, cols, vals, lo = [], [], [], []
    r = 0
    for fi, (opts, Xc) in enumerate(res):
        for j in range(len(opts)):
            rows.append(r); cols.append(vid[(fi, j)]); vals.append(1)
        lo.append(1); r += 1
    nonmu = [e for e in itertools.combinations(L, 2) if tau[e[0]] != e[1]]
    use = {e: [] for e in nonmu}
    acc2, acc3 = {}, {}
    for fi, (opts, Xc) in enumerate(res):
        z, sz = FAR[fi]
        for j, (e, e2) in enumerate(opts):
            v = vid[(fi, j)]
            use[e].append(v); use[e2].append(v)
            for a in z:
                for b in e:
                    acc2.setdefault((a, b), []).append(v)
            for a in sz:
                for b in e2:
                    acc2.setdefault((a, b), []).append(v)
            for b in e:
                for d in e2:
                    acc3.setdefault((b, d), []).append(v)
            for b in e2:
                for d in e:
                    acc3.setdefault((b, d), []).append(v)
    for e in nonmu:
        for v in use[e]:
            rows.append(r); cols.append(v); vals.append(1)
        lo.append(1); r += 1
    for a in L:
        for b in L:
            for v in acc2.get((a, b), []):
                rows.append(r); cols.append(v); vals.append(1)
            lo.append(int(k2[a, b])); r += 1
            for v in acc3.get((a, b), []):
                rows.append(r); cols.append(v); vals.append(1)
            lo.append(int(k3[a, b])); r += 1
    A = sp.csr_matrix((np.array(vals, float), (np.array(rows), np.array(cols))), shape=(r, len(var)))
    lo = np.array(lo, float)
    m = milp(np.zeros(len(var)), constraints=[LinearConstraint(A, lo, lo)], integrality=np.ones(len(var)),
             bounds=Bounds(0, 1), options={'time_limit': tl})
    return {0: 'local-FEASIBLE', 2: 'dead-local'}.get(m.status, 'local-unknown(%d)' % m.status)


def regime_choices(tau, eta, t, sp_):
    Pm = ball(tau, eta)
    tb = torsion_basis(Pm)
    assert len(tb) == t, (len(tb), t)
    full = lambda beta: np.all(beta == 0, axis=0)
    ch = []
    if t == 1:
        for lam in (1, 3):
            ch.append(('t1 lam=%d' % lam, tb, full, {0} | {(-lam * s) % P7 for s in SQ}))
    elif t == 2:
        ch.append(('t2 (2,2)', tb, full, None))
        if sp_:
            for c in proj_points(2):
                ch.append(('t2 (1,0) line', tb, (lambda cc: (lambda beta: (cc @ beta) % P7 == 0))(c), None))
    else:
        # W = a-perp in coefficient space (dim t - 1): balanced iff beta in span(a)
        for a in proj_points(t):
            def pred(beta, a=a):
                # beta_f parallel to a: all 2x2 minors a_i beta_j - a_j beta_i vanish
                ok = np.ones(beta.shape[1], dtype=bool)
                for i in range(t):
                    for j in range(i + 1, t):
                        ok &= ((a[i] * beta[j] - a[j] * beta[i]) % P7 == 0)
                return ok
            ch.append(('t%d (%d,%d) hyperplane' % (t, t - 1, 2 * (t - 1) - t), tb, pred, None))
    return ch


if __name__ == '__main__':
    tsel = [int(x) for x in sys.argv[1].split(',')]
    start, end = int(sys.argv[2]), int(sys.argv[3])
    tcap = float(sys.argv[4]) if len(sys.argv) > 4 else 270
    t0 = time.time()
    cls = [c for c in classes_t_ge(1) if c[2] in tsel][start:end]
    print('classes t in', tsel, ':', len(cls), flush=True)
    out = []
    for n, (tau, eta, t, osz) in enumerate(cls):
        if time.time() - t0 > tcap:
            print('TIME CAP at', n, flush=True); break
        r_, leg = ball_disc(tau, eta)
        sp_ = split(r_, leg)
        stats = []
        alive = False
        for name, tb, pred, zset in regime_choices(tau, eta, t, sp_):
            res = filtered(tau, eta, tb, pred, zset)
            st = local_ilp(tau, eta, res)
            stats.append((name, st))
            if st != 'dead-unary' and st != 'dead-local':
                alive = True
        from collections import Counter
        summary = Counter(s for _, s in stats)
        print(n, 't=%d' % t, tau, eta, 'split' if sp_ else ('non-split' if sp_ is False else 'odd'), dict(summary),
              'ALIVE' if alive else 'dead', '(%.0fs)' % (time.time() - t0), flush=True)
        out.append({'tau': tau, 'eta': eta, 't': t, 'split': sp_, 'stats': stats, 'alive': alive})
    json.dump(out, open('g18_regimes_%s_%d_%d.json' % ('-'.join(map(str, tsel)), start, end), 'w'))
    print('alive classes:', sum(1 for o in out if o['alive']), 'of', len(out))
