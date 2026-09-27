"""g- = 16 is local (C1, 26/09 night).  From g = 18 + t_c - 2 r0_c + rho_c (notes/129 sec. 5 bis) at g = 16 every block c has
  t_c = 2, r0_c = 2, rho_c = 0   -> torsion fully balanced and R_c = 0: T_FF = Z (mod 7) on all 30 far orbits;
  t_c = 3, r0_c = 3, rho_c = 1   -> fully balanced, R_c = lambda k k^T (k^T k = 0, X k = 0);
  t_c = 4, (r0_c, rho_c) = (4, 2) or (3, 0).
Here: regime t = 2 (and the balance filter for t = 3, 4 with r0 = t).  For a block type (tau, eta) (ours; = C2's (mu, tau)),
each far orbit f needs a shadow option (Phi(z), Phi(sigma z)) with
  (bal)  x_f orthogonal to ker P (P = ball of N = S - 3I, 19 x 19; x_f = column of f: block entries B, star entries s_f),
  (Z)    Z_ff = 3 + x_f^T P^+ x_f = 0 (mod 7)                          [t = 2 only]
(both unary: they only involve f's own shadow), then (i)+(ii)+(iii) of C2's free-star lemma (structures' full_local.py)
and the pairwise T_ff' = Z_ff' in {0, +-1} with 10 - |supp s_f| nonzero entries per row [t = 2].
Exact mod-7 linear algebra; P^+ from a nonsingular principal submatrix of maximal size (P symmetric)."""
import itertools, sys, time, json
import numpy as np

P7 = 7
L = list(range(12))
PI = [x ^ 1 for x in L]


def fpf_involutions(pts):
    if not pts:
        yield {}
        return
    a = pts[0]
    for i in range(1, len(pts)):
        b = pts[i]
        rest = pts[1:i] + pts[i + 1:]
        for m in fpf_involutions(rest):
            m = dict(m)
            m[a] = b
            m[b] = a
            yield m


ALL = [tuple(m[x] for x in L) for m in fpf_involutions(L)]
G = []
for perm in itertools.permutations(range(6)):
    for flips in range(64):
        G.append(tuple(2 * perm[i // 2] + ((i % 2) ^ ((flips >> (i // 2)) & 1)) for i in range(12)))


def conj(t, g):
    ginv = [0] * 12
    for i, gi in enumerate(g):
        ginv[gi] = i
    return tuple(g[t[ginv[x]]] for x in L)


def rank_mod(M):
    M = [[int(x) % P7 for x in r] for r in M]
    n, m = len(M), len(M[0])
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, n) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], P7 - 2, P7)
        M[r] = [x * inv % P7 for x in M[r]]
        for i in range(n):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % P7 for a, b in zip(M[i], M[r])]
        r += 1
    return r


def inv_mod(A):
    n = len(A)
    M = [[int(x) % P7 for x in r] + [1 if i == j else 0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        piv = next((i for i in range(c, n) if M[i][c]), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        inv = pow(M[c][c], P7 - 2, P7)
        M[c] = [x * inv % P7 for x in M[c]]
        for i in range(n):
            if i != c and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % P7 for a, b in zip(M[i], M[c])]
    return np.array([r[n:] for r in M], dtype=np.int64)


def kernel_mod(A):
    """basis of the right kernel of A (mod 7), as rows."""
    A = [[int(x) % P7 for x in r] for r in A]
    n, m = len(A), len(A[0])
    M = [r[:] for r in A]
    pivcols = []
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, n) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], P7 - 2, P7)
        M[r] = [x * inv % P7 for x in M[r]]
        for i in range(n):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % P7 for a, b in zip(M[i], M[r])]
        pivcols.append(c)
        r += 1
    free = [c for c in range(m) if c not in pivcols]
    basis = []
    for fc in free:
        v = [0] * m
        v[fc] = 1
        for i, pc in enumerate(pivcols):
            v[pc] = (-M[i][fc]) % P7
        basis.append(v)
    return np.array(basis, dtype=np.int64).reshape(len(basis), m)


def ginv_sym(Pm):
    """generalized inverse of a symmetric matrix mod 7 via a maximal nonsingular principal submatrix (greedy)."""
    n = Pm.shape[0]
    S = []
    for i in range(n):
        T = S + [i]
        if rank_mod(Pm[np.ix_(T, T)].tolist()) == len(T):
            S = T
    r = rank_mod(Pm.tolist())
    if len(S) != r:
        # greedy may fail in principle; try swaps
        for T in itertools.combinations(range(n), r):
            if rank_mod(Pm[np.ix_(T, T)].tolist()) == r:
                S = list(T)
                break
    Ginv = np.zeros((n, n), dtype=np.int64)
    Ginv[np.ix_(S, S)] = inv_mod(Pm[np.ix_(S, S)].tolist())
    assert np.array_equal((Pm @ Ginv @ Pm) % P7, Pm % P7)
    return Ginv


def K2(mu, tau):
    K = np.full((12, 12), 2) - np.eye(12, dtype=int)
    for a in L:
        K[a, PI[a]] -= 1
        K[a, mu[a]] -= 1
        K[a, tau[PI[a]]] -= 1
    return K


def K3(mu, tau):
    K = np.full((12, 12), 2)
    for b in L:
        for f in (PI, [tau[mu[x]] for x in L], [mu[tau[x]] for x in L], tau):
            K[b, f[b]] -= 1
    return K


FAR = []
_seen = set()
for z in itertools.combinations(L, 2):
    if z[0] // 2 == z[1] // 2 or z in _seen:
        continue
    sz = tuple(sorted((PI[z[0]], PI[z[1]])))
    _seen |= {z, sz}
    FAR.append((z, sz))


def ball(tau, eta):
    Pm = np.zeros((19, 19), dtype=np.int64)
    for b in range(7):
        Pm[b, b] = -4
    for l in L:
        g = 7 + l
        Pm[6, g] = Pm[g, 6] = 1
        sgn = 1 if l % 2 == 0 else -1
        Pm[l // 2, g] = Pm[g, l // 2] = sgn
        Pm[g, 7 + tau[l]] += 1
        Pm[g, 7 + eta[l]] -= 1
        Pm[g, g] -= 3
    return Pm % P7


POINTWISE = False     # set True (or env Q7_POINTWISE=1) to add the pointwise lambda/mu filters (L1), (S1), (S2)


def pointwise_ok(z, sz, e, e2, tau, eta):
    """pointwise local conditions (lambda = 1 / mu = 2), valid at every block (C1, 27/09):
    (L1) for every label a of the two blocks of z: not (a in Phi(z) and sigma a in Phi(sigma z));
    (S1) for l in Phi(z): not (l in z and eta(l) in Phi(sigma z))            [lambda on the edge (c+,l)-z];
    (S2) for l not in Phi(z): not (l in z and tau(l) in Phi(z) and eta(l) in Phi(sigma z))   [mu = 2];
    and (S1), (S2) for sigma z with (Phi(sigma z), Phi(z))."""
    blocks = {a // 2 for a in z}
    for a in L:
        if a // 2 in blocks and a in e and PI[a] in e2:
            return False
    for (zz, E, E2) in ((z, e, e2), (sz, e2, e)):
        for l in L:
            if l in zz:
                if l in E:
                    if eta[l] in E2:
                        return False
                elif tau[l] in E and eta[l] in E2:
                    return False
    return True


def orbit_options(tau, eta, k2):
    """per far orbit: list of (e, e2) and the matrix of columns x (19 x n_opt)."""
    import os
    pw = POINTWISE or os.environ.get('Q7_POINTWISE') == '1'
    nonmu = [e for e in itertools.combinations(L, 2) if tau[e[0]] != e[1]]
    out = []
    for (z, sz) in FAR:
        ok1 = [e for e in nonmu if all(k2[a, b] > 0 for a in z for b in e)]
        ok2 = [e for e in nonmu if all(k2[a, b] > 0 for a in sz for b in e)]
        opts = [(e, e2) for e in ok1 for e2 in ok2 if e != e2]
        if pw:
            opts = [(e, e2) for (e, e2) in opts if pointwise_ok(z, sz, e, e2, tau, eta)]
        Xc = np.zeros((19, len(opts)), dtype=np.int64)
        for a in z:
            Xc[a // 2, :] = 1 if a % 2 == 0 else -1
        for j, (e, e2) in enumerate(opts):
            for l in e:
                Xc[7 + l, j] += 1
            for l in e2:
                Xc[7 + l, j] -= 1
        out.append((opts, Xc % P7))
    return out


def unary_filter(tau, eta, regime_t2=True):
    """returns (t, per-orbit surviving option lists with their columns, P^+)."""
    Pm = ball(tau, eta)
    kerP = kernel_mod(Pm.tolist())          # 1 + t vectors
    Pp = ginv_sym(Pm)
    k2 = K2(tau, eta)
    res = []
    for (opts, Xc) in orbit_options(tau, eta, k2):
        if len(opts) == 0:
            res.append(([], Xc[:, :0]))
            continue
        bal = np.all((kerP @ Xc) % P7 == 0, axis=0)
        keep = bal
        if regime_t2:
            zz = (3 + np.einsum('ij,ij->j', Xc, (Pp @ Xc) % P7)) % P7
            keep = keep & (zz == 0)
        idx = np.nonzero(keep)[0]
        res.append(([opts[j] for j in idx], Xc[:, idx]))
    return kerP.shape[0] - 1, res, Pp


def classes_t_ge(tmin=2):
    """(tau, eta) up to Aut(X, pi) with t >= tmin; validity K2, K3 >= 0 (C2's mu = tau, C2's tau = eta)."""
    seen = set()
    reps_tau = []
    for t in ALL:
        if t in seen:
            continue
        orb = {conj(t, g) for g in G}
        seen |= orb
        reps_tau.append(t)
    out = []
    for tau in reps_tau:
        stab = [g for g in G if conj(tau, g) == tau]
        seen_eta = set()
        for eta in ALL:
            if eta in seen_eta:
                continue
            orb = {conj(eta, g) for g in stab}
            seen_eta |= orb
            Q = [[(tau[w] == v) - (eta[w] == v) - 2 * (PI[w] == v) - (w == v) for v in L] for w in L]
            t = 12 - rank_mod(Q)
            if t < tmin:
                continue
            if (K2(tau, eta) < 0).any() or (K3(tau, eta) < 0).any():
                continue
            out.append((tau, eta, t, len(orb)))
    return out


def congruence_diag(M):
    """diagonalize a symmetric matrix mod 7 by congruence; returns the list of nonzero diagonal entries."""
    A = [[int(x) % P7 for x in row] for row in M]
    n = len(A)
    diag = []
    active = list(range(n))
    while active:
        i = next((k for k in active if A[k][k]), None)
        if i is None:
            pair = next(((k, l) for k in active for l in active if k != l and A[k][l]), None)
            if pair is None:
                break                                   # remaining block is zero
            k, l = pair                                 # e_k <- e_k + e_l makes A_kk = 2 A_kl != 0
            for m in range(n):
                A[k][m] = (A[k][m] + A[l][m]) % P7
            for m in range(n):
                A[m][k] = (A[m][k] + A[m][l]) % P7
            i = k
        piv = A[i][i]
        inv = pow(piv, P7 - 2, P7)
        for j in active:
            if j != i and A[j][i]:
                f = A[j][i] * inv % P7
                for m in range(n):
                    A[j][m] = (A[j][m] - f * A[i][m]) % P7
                for m in range(n):
                    A[m][j] = (A[m][j] - f * A[m][i]) % P7
        diag.append(piv)
        active.remove(i)
    return diag


def ball_disc(tau, eta):
    """(rank, Legendre symbol of the discriminant of the nondegenerate part) of the ball N_I = S_I - 3I (mod 7)."""
    d = congruence_diag(ball(tau, eta).tolist())
    det = 1
    for x in d:
        det = det * x % P7
    return len(d), (1 if pow(det, 3, P7) == 1 else -1)


def split(r, leg):
    """even-dimensional nondegenerate part is hyperbolic iff disc = (-1)^(r/2) mod squares (-1 is a non-square mod 7)."""
    if r % 2:
        return None
    return leg == (1 if (r // 2) % 2 == 0 else -1)


if __name__ == '__main__':
    t0 = time.time()
    cls = classes_t_ge(2)
    print('valid (tau, eta) classes with t >= 2:', len(cls), ' by t:',
          {t: sum(1 for c in cls if c[2] == t) for t in (2, 3, 4)}, '(%.0fs)' % (time.time() - t0), flush=True)
    summary = []
    witt_dead = 0
    for (tau, eta, t, osz) in cls:
        r, leg = ball_disc(tau, eta)
        assert r == 18 - t
        sp = split(r, leg)
        if t == 2 and sp is False:
            witt_dead += 1
            summary.append({'tau': tau, 'eta': eta, 't': t, 'orbit_size': osz, 'witt': 'non-split', 'min_opts': 0})
            continue
        tt, res, Pp = unary_filter(tau, eta, regime_t2=(t == 2))
        counts = [len(r[0]) for r in res]
        summary.append({'tau': tau, 'eta': eta, 't': t, 'orbit_size': osz, 'witt': {None: 'odd', True: 'split', False: 'non-split'}[sp],
                        'min_opts': min(counts), 'n_empty': sum(1 for c in counts if c == 0), 'counts': counts})
    print('t = 2 classes killed by Witt (non-split reduced ball):', witt_dead, flush=True)
    dead = [s for s in summary if s['min_opts'] == 0]
    print('classes killed by the unary filters (some far orbit has no option):', len(dead), 'of', len(summary),
          '(%.0fs)' % (time.time() - t0), flush=True)
    for s in summary:
        if s['min_opts'] > 0:
            print('ALIVE after unary:', s['t'], s['tau'], s['eta'], 'counts', s['counts'])
    json.dump(summary, open('g16_unary_summary.json', 'w'))
