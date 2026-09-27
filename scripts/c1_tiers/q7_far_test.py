"""q:7 far test at a block c with t_c = 0 (notes/129 sec. 5, notes/131 sec. 7), for a local solution (mu, tau, Phi) in the
format of C2's free-star lemma / structures' freestar/*.json:
  labels 0..11 = block i//2 (the 6 blocks other than c), sign + if i even, sigma(i) = i ^ 1;
  mu  = C2's mu = our tau (mates in U: (c+, l) ~ (c+, mu l));
  tau = C2's tau = our eta ((c+, l) ~ (c-, sigma(eta l)));
  Phi[z] = the 2 labels l with z ~ (c+, l), for each of the 60 far vertices z = {a, a'} (labels of two other blocks).
Builds N = S - 3I (mod 7) on the ball (7 blocks + 12 star orbits) and the ball x far block X, solves P U = X and returns
Z = 3I + X^T U: T_FF = Z + R_c (mod 7), R_c = K r K^T of rank g - 18, X K = 0, K^T K = 0.
Exact integer identities used in checks: row f of T has exactly 10 - |supp s_f| nonzero far entries."""
import itertools
import numpy as np

P7 = 7


def sig(i):
    return i ^ 1


def rank_mod(M, p=P7):
    M = [[int(x) % p for x in r] for r in M]
    rows = len(M)
    ncols = len(M[0]) if rows else 0
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, rows) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [x * inv % p for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % p for a, b in zip(M[i], M[r])]
        r += 1
    return r


def det_mod(M, p=P7):
    M = [[int(x) % p for x in r] for r in M]
    n = len(M)
    det = 1
    for c in range(n):
        piv = next((i for i in range(c, n) if M[i][c]), None)
        if piv is None:
            return 0
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            det = -det
        det = det * M[c][c] % p
        inv = pow(M[c][c], p - 2, p)
        for i in range(c + 1, n):
            if M[i][c]:
                f = M[i][c] * inv % p
                M[i] = [(a - f * b) % p for a, b in zip(M[i], M[c])]
    return det % p


def solve_mod(A, Bm, p=P7):
    """one solution U of A U = B (mod p), or None if inconsistent."""
    A = np.array(A, dtype=np.int64) % p
    Bm = np.array(Bm, dtype=np.int64) % p
    n, m = A.shape
    k = Bm.shape[1]
    M = np.concatenate([A, Bm], axis=1)
    piv_cols = []
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, n) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = M[r] * pow(int(M[r, c]), p - 2, p) % p
        for i in range(n):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        piv_cols.append(c)
        r += 1
    if np.any(M[r:, m:] % p):
        return None
    U = np.zeros((m, k), dtype=np.int64)
    for i, c in enumerate(piv_cols):
        U[c] = M[i, m:]
    return U % p


def legendre(a, p=P7):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def build(mu, tau, Phi):
    """mu, tau: dict label -> label (fixed-point-free involutions); Phi: dict frozenset({a,a'}) -> set of 2 labels."""
    L = list(range(12))
    Qc = np.zeros((12, 12), dtype=np.int64)
    for l in L:
        Qc[l, mu[l]] += 1
        Qc[l, tau[l]] -= 1
        Qc[l, sig(l)] -= 2
        Qc[l, l] -= 1
    # far orbits: reps z = (a, a') with a < a', and z not yet seen as sigma of an earlier rep
    far, seen = [], set()
    for z in itertools.combinations(L, 2):
        if z[0] // 2 == z[1] // 2:
            continue
        if z in seen:
            continue
        sz = tuple(sorted((sig(z[0]), sig(z[1]))))
        seen |= {z, sz}
        far.append((z, sz))
    nF = len(far)
    # ball coordinates: blocks 0..5 (others), 6 = c; star orbits g_l, l = 0..11 (rep (c+, l))
    nb = 7 + 12
    Pm = np.zeros((nb, nb), dtype=np.int64)
    for b in range(7):
        Pm[b, b] = -4
    for l in L:
        g = 7 + l
        Pm[6, g] = Pm[g, 6] = 1                      # rep (c+, l) ~ c+
        sgn = 1 if l % 2 == 0 else -1                # rep ~ label l of block l//2
        Pm[l // 2, g] = Pm[g, l // 2] = sgn
        for m in L:
            Pm[g, 7 + m] += (1 if m == mu[l] else 0) - (1 if m == tau[l] else 0)
        Pm[g, g] -= 3
    X = np.zeros((nb, nF), dtype=np.int64)
    S = np.zeros((12, nF), dtype=np.int64)
    BF = np.zeros((6, nF), dtype=np.int64)
    for j, (z, sz) in enumerate(far):
        for a in z:
            sgn = 1 if a % 2 == 0 else -1
            X[a // 2, j] = sgn
            BF[a // 2, j] = sgn
        Pz, Psz = set(Phi[frozenset(z)]), set(Phi[frozenset(sz)])
        for l in L:
            v = (1 if l in Pz else 0) - (1 if l in Psz else 0)
            S[l, j] = v
            X[7 + l, j] = v
    return dict(Qc=Qc, P=Pm, X=X, S=S, BF=BF, far=far)


def analyze(mu, tau, Phi, verbose=True):
    D = build(mu, tau, Phi)
    Qc = D['Qc']
    t = 12 - rank_mod(Qc.tolist())
    det = det_mod(Qc.tolist()) if t == 0 else 0
    out = dict(t=t, det=det, leg=legendre(det) if det else 0)
    U = solve_mod(D['P'], D['X'])
    out['balanced'] = U is not None
    if U is None:
        return out, D
    X = D['X']
    nF = X.shape[1]
    Z = (3 * np.eye(nF, dtype=np.int64) + X.T @ U) % P7
    out['Z_symmetric'] = bool(np.array_equal(Z, Z.T))
    supp = (D['S'] != 0).sum(axis=0)
    rigid = [f for f in range(nF) if Z[f, f] == 0]
    bad_rows = []
    for f in rigid:
        row = [int(Z[f, g]) for g in range(nF) if g != f]
        nonpm = sum(1 for v in row if v not in (0, 1, 6))
        nz = sum(1 for v in row if v != 0)
        if nonpm or nz != 10 - supp[f]:
            bad_rows.append((f, nonpm, nz, int(10 - supp[f])))
    out.update(rigid=rigid, bad_rigid_rows=bad_rows, shadow_supports=sorted(set(int(x) for x in supp)))
    D['Z'] = Z
    D['U'] = U
    return out, D


def load_structures_json(path):
    import json
    d = json.load(open(path))
    mu = {int(k): int(v) for k, v in d['mu'].items()}
    tau = {int(k): int(v) for k, v in d['tau'].items()}
    Phi = {}
    for k, v in d['Phi (far root -> non-mu pair)'].items():
        a, b = json.loads(k)
        Phi[frozenset((a, b))] = tuple(v)
    return mu, tau, Phi


if __name__ == '__main__':
    import sys
    mu, tau, Phi = load_structures_json(sys.argv[1])
    out, D = analyze(mu, tau, Phi)
    for k, v in out.items():
        print(k, v)


def far_K_search(Z, X, S, det_square, g=20, node_cap=10 ** 6):
    """Search K (30 x 2 over F_7) with T := Z + K r K^T in {0, +-1} off the diagonal, T_ff = 0, the row counts
    #{f' != f : T_ff' != 0} = 10 - |supp s_f|, X K = 0, K^T K = 0, rank K = 2.  r = I (det Q_c square: anisotropic) or
    diag(1, -1) (hyperbolic).  g = 18: K = 0 (T = Z).  Returns (status, info)."""
    nF = Z.shape[0]
    need = [10 - int((S[:, f] != 0).sum()) for f in range(nF)]
    if g == 18:
        T = Z % P7
        ok = all(T[f, f] == 0 for f in range(nF)) and all(T[f, h] in (0, 1, 6) for f in range(nF) for h in range(nF))
        ok = ok and all(sum(1 for h in range(nF) if h != f and T[f, h]) == need[f] for f in range(nF))
        return ('alive' if ok else 'dead'), {}
    rr = np.eye(2, dtype=np.int64) if det_square else np.diag([1, -1]).astype(np.int64)

    def q(u, v):
        return int(u @ rr @ v) % P7
    vecs = [np.array([a, b], dtype=np.int64) for a in range(P7) for b in range(P7)]
    by_norm = {}
    for v in vecs:
        by_norm.setdefault(q(v, v), []).append(v)
    opts = [by_norm[(-int(Z[f, f])) % P7] for f in range(nF)]
    # order: rigid (single option 0 in the anisotropic case) first, then by fewest options
    order = sorted(range(nF), key=lambda f: len(opts[f]))
    K = [None] * nF
    stats = {'nodes': 0, 'fail_entry': 0, 'fail_count': 0, 'leaves': 0, 'fail_final': 0}
    sols = []
    fixed_first = [False]

    def entry(f, h):
        return (int(Z[f, h]) + q(K[f], K[h])) % P7

    def dfs(i):
        stats['nodes'] += 1
        if stats['nodes'] > node_cap:
            return
        if i == nF:
            stats['leaves'] += 1
            Km = np.array(K)
            if np.any((X @ Km) % P7) or np.any((Km.T @ Km) % P7) or rank_mod(Km.T.tolist()) < 2:
                stats['fail_final'] += 1
                return
            sols.append(Km.copy())
            return
        f = order[i]
        cand = opts[f]
        if not fixed_first[0] and det_square and any(np.any(v) for v in cand):
            cand = [next(v for v in cand if np.any(v))]           # SO(r) acts simply transitively on a norm circle
            fixed_first[0] = True
            first_here = True
        else:
            first_here = False
        for v in cand:
            K[f] = v
            good = True
            for j in range(i):
                h = order[j]
                if entry(f, h) not in (0, 1, 6):
                    good = False
                    stats['fail_entry'] += 1
                    break
            if good:
                # row counts: rows of all assigned orbits, bounded by the unassigned remainder
                assigned = order[:i + 1]
                rest = nF - (i + 1)
                for h in assigned:
                    nz = sum(1 for h2 in assigned if h2 != h and entry(h, h2))
                    if nz > need[h] or nz + rest < need[h]:
                        good = False
                        stats['fail_count'] += 1
                        break
            if good:
                dfs(i + 1)
            K[f] = None
        if first_here:
            fixed_first[0] = False

    dfs(0)
    status = 'alive' if sols else ('dead' if stats['nodes'] <= node_cap else 'unknown')
    return status, dict(stats=stats, n_solutions=len(sols), solutions=sols[:200], r=rr)
