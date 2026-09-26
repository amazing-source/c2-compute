"""Odd far-far block T_FF at a FULL block c, given the local solution Phi (shadows s_f, block signs B_f of the 30 far orbits).
Exact integer consequences of S^2 + S = 12I (odd sector, 49 x 49; rows f far, columns o_v / d_e):
  (E1) for every far f and label v:  sum_{f''} T[f,f''] s_{f''}(v) = -s_f(v) - sign(v) B[blk(v), f]
       (uses S_{o_w,o_v} = 0 at a full block: [m(w) = v] - [h(w) = pi v] with h = pi m);
  (E2) for every far f and block e != c:  sum_{f''} T[f,f''] B[e,f''] = s_f(e-) - s_f(e+);
  (E3) diagonal: 2 + 4 + #{f'' : T[f,f''] != 0} = 12, i.e. exactly 6 nonzero entries per row;
  off-diagonal (quadratic, checked on solutions): B_f.B_f' + s_f.s_f' + (T^2)[f,f'] + T[f,f'] = 0.
Plus: T symmetric, zero diagonal, entries in {-1, 0, 1}; T = 0 between the two orbits of an anchored pair (n_s = 0);
q:7 rigid rows (C1): Z_ff = 0 (mod 7) => T[f,.] = Z[f,.] (mod 7), which fixes the row exactly.
Mod-7 residue (checked on solutions): T - Z = K K^T of rank exactly 2, anisotropic (g- = 20 at a full block)."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

P7 = 7


def rank_mod(M):
    M = [[int(x) % P7 for x in r] for r in M]
    r = 0
    ncol = len(M[0]) if M else 0
    for c in range(ncol):
        p = next((i for i in range(r, len(M)) if M[i][c]), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = pow(M[r][c], P7 - 2, P7)
        M[r] = [x * inv % P7 for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % P7 for a, b in zip(M[i], M[r])]
        r += 1
    return r


def local_data(var, reps, chosen):
    n = len(reps)
    S = np.zeros((n, 12), int)
    B = np.zeros((n, 6), int)
    for j in chosen:
        Zi, e, e2 = var[j]
        z, sz = reps[Zi]
        S[Zi, list(e)] += 1
        S[Zi, list(e2)] -= 1
        for a in z:
            B[Zi, a // 2] = 1 if a % 2 == 0 else -1
    return S, B


def tff_ilp(S, B, Z, sisters_zero, tl=60, use_rigid=True):
    """S: 30 x 12 shadows, B: 30 x 6 block signs, Z: 30 x 30 mod 7.  Returns (status, T)."""
    n = S.shape[0]
    pairs = [(f, g) for f in range(n) for g in range(f + 1, n)]
    pidx = {p: i for i, p in enumerate(pairs)}
    npair = len(pairs)
    nv = 2 * npair                                    # p (T = +1), q (T = -1)
    lb = np.zeros(nv); ub = np.ones(nv)

    def tvars(f, g):
        i = pidx[(min(f, g), max(f, g))]
        return i, npair + i
    for f, g in sisters_zero:
        i, k = tvars(f, g)
        ub[i] = ub[k] = 0
    if use_rigid:
        for f in range(n):
            if Z[f, f] % P7 == 0:
                for g in range(n):
                    if g == f:
                        continue
                    zv = int(Z[f, g]) % P7
                    i, k = tvars(f, g)
                    if zv == 0:
                        ub[i] = ub[k] = 0
                    elif zv == 1:
                        lb[i] = 1; ub[k] = 0
                    elif zv == 6:
                        lb[k] = 1; ub[i] = 0
                    else:
                        return 'INFEASIBLE (rigid row %d has Z = %d)' % (f, zv), None
    if (lb > ub).any():
        return 'INFEASIBLE (bounds)', None
    ri, ci, vv, lo, hi = [], [], [], [], []
    r = [0]

    def row(entries, a, b):
        for j, c in entries:
            if c:
                ri.append(r[0]); ci.append(j); vv.append(c)
        lo.append(a); hi.append(b); r[0] += 1
    for f in range(n):
        others = [g for g in range(n) if g != f]
        for v in range(12):                           # (E1)
            ent = []
            for g in others:
                i, k = tvars(f, g)
                ent += [(i, int(S[g, v])), (k, -int(S[g, v]))]
            sign = 1 if v % 2 == 0 else -1
            rhs = -int(S[f, v]) - sign * int(B[f, v // 2])
            row(ent, rhs, rhs)
        for e in range(6):                            # (E2)
            ent = []
            for g in others:
                i, k = tvars(f, g)
                ent += [(i, int(B[g, e])), (k, -int(B[g, e]))]
            rhs = int(S[f, 2 * e + 1]) - int(S[f, 2 * e])
            row(ent, rhs, rhs)
        ent = []                                      # (E3) diagonal: 6 nonzeros
        for g in others:
            i, k = tvars(f, g)
            ent += [(i, 1), (k, 1)]
        row(ent, 6, 6)
    for i in range(npair):                            # p + q <= 1
        row([(i, 1), (npair + i, 1)], 0, 1)
    A = sp.csr_matrix((np.array(vv, float), (np.array(ri), np.array(ci))), shape=(r[0], nv))
    res = milp(np.zeros(nv), constraints=[LinearConstraint(A, np.array(lo, float), np.array(hi, float))],
               integrality=np.ones(nv), bounds=Bounds(lb, ub), options={'time_limit': tl})
    if res.status != 0:
        return {2: 'INFEASIBLE'}.get(res.status, 'unknown(%d)' % res.status), None
    x = np.round(res.x).astype(int)
    T = np.zeros((n, n), int)
    for (f, g), i in pidx.items():
        T[f, g] = T[g, f] = x[i] - x[npair + i]
    return 'FEASIBLE', T


def post_checks(S, B, T, Z):
    n = S.shape[0]
    E3 = B @ B.T + S @ S.T + T @ T + T - 12 * np.eye(n, dtype=int)
    R = (T - Z) % P7
    return {'E3_offdiag_violations': int((E3 != 0).sum()), 'rank7(T - Z)': rank_mod(R.tolist())}
