"""
Exact eigen-lattices of a (lambda,mu)=(1,2) srg with an involution t (one fixed vertex), as Z[t]-modules.

For each eigenvalue theta:  M = ker_Z(A - theta I) (primitive in Z^n, standard form),
  L+ = M cap (Z^n)^t,  L- = M cap (Z^n)^{-t}  (both primitive, standard form restricted).
Reports: rank, det (factored), Smith invariants of the Gram matrices, evenness, the glue index
[M : L+ (+) L-] = sqrt(det L+ det L- / det M), and the F2[t]-structure of M mod 2 (is it free?).
Controls: Paley(9) = K3 x K3 (m = 2), BvLS srg(243,22,1,2) (m = 11).
"""
import sys, os, itertools
from collections import Counter
import numpy as np
import flint
from flint import fmpz_mat, fmpz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def int_kernel(C):
    """rows = Z-basis of {x in Z^n : C x = 0}, C an fmpz_mat (m x n)."""
    n = C.ncols()
    H, U = C.transpose().hnf(transform=True)
    rows = [i for i in range(H.nrows()) if all(H[i, j] == 0 for j in range(H.ncols()))]
    return fmpz_mat([[U[i, j] for j in range(n)] for i in rows])


def factor(n):
    n = int(n); f = {}
    if n < 0: n = -n
    p = 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: f[n] = f.get(n, 0) + 1
    return f


def fstr(f):
    return '*'.join('%d^%d' % (p, e) if e > 1 else str(p) for p, e in sorted(f.items())) or '1'


def smith_summary(G):
    D = G.snf()
    inv = Counter(int(D[i, i]) for i in range(min(D.nrows(), D.ncols())))
    return dict(sorted(inv.items()))


def np2fmpz(M):
    return fmpz_mat(M.tolist())


def analyze(name, A, tperm, thetas):
    n = A.shape[0]
    T = np.zeros((n, n), dtype=np.int64)
    for v in range(n):
        T[tperm[v], v] = 1
    I = np.eye(n, dtype=np.int64)
    out = {}
    for th in thetas:
        C = A - th * I
        M = int_kernel(np2fmpz(C))
        Lp = int_kernel(np2fmpz(np.vstack([C, I - T])))
        Lm = int_kernel(np2fmpz(np.vstack([C, I + T])))
        res = {}
        for lab, L in (('M', M), ('L+', Lp), ('L-', Lm)):
            G = L * L.transpose()
            d = G.det()
            even = all(G[i, i] % 2 == 0 for i in range(G.nrows()))
            res[lab] = dict(rank=L.nrows(), det=int(d), fac=factor(d), smith=smith_summary(G), even=even, basis=L, gram=G)
            print("%s theta=%d %-3s rank %3d det %s smith %s even %s" % (name, th, lab, L.nrows(), fstr(factor(d)), res[lab]['smith'], even), flush=True)
        idx2 = res['L+']['det'] * res['L-']['det'] // res['M']['det']
        r = int(round(idx2 ** 0.5))
        assert r * r == idx2, idx2
        # F2[t]-structure of M mod 2: dim of fixed subspace of the reduction
        Mm = np.array(M.tolist(), dtype=np.int64) % 2
        # fixed subspace of the code K = rowspace(Mm) under t: solve within K
        K = Mm.copy()
        # K^t = {x in K : x T = x}; compute dim via rank of the map x -> x(I-T) on K
        rank_K = gf2_rank(K)
        img = (K @ ((I - T) % 2)) % 2
        rank_img = gf2_rank(img)
        dimfix = rank_K - rank_img
        print("%s theta=%d glue index [M:L+ (+) L-] = %s ; dim_F2 K = %d, dim K^t = %d (free iff K^t = K/2 -> %s), 2^(dimK - dimK^t) = %s"
              % (name, th, fstr(factor(r)), rank_K, dimfix, dimfix * 2 == rank_K, fstr(factor(2 ** (rank_K - dimfix)))), flush=True)
        out[th] = res
    return out


def gf2_rank(Mat):
    M = (np.array(Mat, dtype=np.int64) % 2).astype(np.uint8)
    M = M.copy(); r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i; break
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r


def paley9():
    V = [(a, b) for a in range(3) for b in range(3)]
    idx = {v: i for i, v in enumerate(V)}
    A = np.zeros((9, 9), dtype=np.int64)
    for u in V:
        for v in V:
            if u != v and ((u[0] == v[0]) != (u[1] == v[1])):
                A[idx[u], idx[v]] = 1
    t = [idx[((-v[0]) % 3, (-v[1]) % 3)] for v in V]
    return A, t


def bvls():
    from inv import bvls_in_labelling
    cfg, A = bvls_in_labelling()
    t = [cfg.vid[cfg.t(v)] for v in cfg.V]
    return A.astype(np.int64), t


if __name__ == '__main__':
    which = sys.argv[1:] or ['paley9', 'bvls']
    if 'paley9' in which:
        A, t = paley9()
        analyze('Paley9', A, t, [1, -2])
    if 'bvls' in which:
        A, t = bvls()
        analyze('BvLS', A, t, [4, -5])
