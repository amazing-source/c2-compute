"""
Extract, from a full solution A (adjacency in cfg labelling) with the involution t, the orbit
sign matrix S, the block-local data at block c (type mu/tau, Phi), and check the linear algebra:
  S^2 + S = (2m-2) I ;  G = rI - S = d * projection ;  known span at c has dim 3m-3 ;
  residual Gram of the far orbits = d * projection of rank (rank G) - (3m-3).
"""
import numpy as np
from inv import Config, bvls_in_labelling, check_srg


def srg_eigs(m):
    d = int(round((8 * m - 7) ** 0.5))
    assert d * d == 8 * m - 7
    r = (-1 + d) // 2
    s = (-1 - d) // 2
    return r, s, d


class Orbits:
    """t-orbits on the non-fixed vertices, with a gauge adapted to block c (star reps in star+)."""
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg.c
        reps = []
        for i in range(cfg.m):
            reps.append(('L', i, 0))
        for v in cfg.V:
            if v[0] != 'D':
                continue
            ends = cfg.ends(v)
            if (c, 0) in ends:
                reps.append(v)                      # star(c) orbit, rep in star+
            elif (c, 1) in ends:
                continue                            # its t-image is in star+
            elif v[2] == 0:
                reps.append(v)                      # far orbit, rep has sign 0 on smaller line
        self.reps = reps
        self.index = {v: k for k, v in enumerate(reps)}
        self.nL = cfg.m
        self.star = [k for k, v in enumerate(reps) if v[0] == 'D' and (c, 0) in cfg.ends(v)]
        self.far = [k for k, v in enumerate(reps) if v[0] == 'D' and (c, 0) not in cfg.ends(v)]
        self.lines = list(range(cfg.m))
        assert len(self.star) == 2 * (cfg.m - 1)
        assert len(self.far) == (cfg.m - 1) * (cfg.m - 2)

    def orbit_of(self, v):
        """(orbit index, sign): sign +1 if v is the rep, -1 if v = t(rep)."""
        if v in self.index:
            return self.index[v], 1
        w = self.cfg.t(v)
        return self.index[w], -1


def sign_matrix(cfg, A, orb):
    n = len(orb.reps)
    S = np.zeros((n, n), dtype=np.int64)
    for P, u in enumerate(orb.reps):
        iu = cfg.vid[u]
        for Q, v in enumerate(orb.reps):
            S[P, Q] = int(A[iu, cfg.vid[v]]) - int(A[iu, cfg.vid[cfg.t(v)]])
    return S


def block_type(cfg, A):
    """(mu, tau) at block c as dicts on labels; mu: [c+,v]~[c+,mu v]; tau: [c+,v]~[c-,pi(tau v)]."""
    mu, tau = {}, {}
    for v in cfg.labels:
        sp = cfg.vid[cfg.star_p(v)]
        mates = [w for w in cfg.labels if A[sp, cfg.vid[cfg.star_p(w)]]]
        assert len(mates) == 1, mates
        mu[v] = mates[0]
        cm = [w for w in cfg.labels if A[sp, cfg.vid[cfg.star_m(w)]]]
        assert len(cm) == 1, cm
        tau[v] = cfg.pi(cm[0])
    for v in cfg.labels:
        assert mu[mu[v]] == v and mu[v] != v
        assert tau[tau[v]] == v and tau[v] != v
    return mu, tau


def phi_map(cfg, A):
    Phi = {}
    for z in cfg.far:
        iz = cfg.vid[z]
        labs = [v for v in cfg.labels if A[iz, cfg.vid[cfg.star_p(v)]]]
        assert len(labs) == 2, (z, labs)
        Phi[z] = frozenset(labs)
    return Phi


def validate_linear_algebra(cfg, A, verbose=True):
    m = cfg.m
    r, s, d = srg_eigs(m)
    orb = Orbits(cfg)
    S = sign_matrix(cfg, A, orb)
    n = len(orb.reps)
    assert (S == S.T).all()
    assert (S @ S + S == (2 * m - 2) * np.eye(n, dtype=np.int64)).all(), 'S^2+S failed'
    G = r * np.eye(n, dtype=np.int64) - S
    Gf = G.astype(float)
    assert np.allclose(Gf @ Gf, d * Gf), 'G^2 = dG failed'
    rankG = np.linalg.matrix_rank(Gf)
    K = list(range(m)) + orb.star
    F = orb.far
    GKK = Gf[np.ix_(K, K)]
    rankK = np.linalg.matrix_rank(GKK)
    # dependency: sum of star vectors + r * ell_c = 0  (norm check)
    coef = np.zeros(n); coef[orb.star] = 1.0; coef[cfg.c] = r
    dep_norm = coef @ Gf @ coef
    # residual Gram of far vectors
    GKF = Gf[np.ix_(K, F)]
    GFF = Gf[np.ix_(F, F)]
    Proj = GKF.T @ np.linalg.pinv(GKK) @ GKF
    R = GFF - Proj
    rankR = np.linalg.matrix_rank(R, tol=1e-8)
    ok_R = np.allclose(R @ R, d * R)
    proj_norms = np.diag(Proj)
    if verbose:
        print(f'm={m}: r={r}, s={s}, d={d}; n_orbits={n}; rank G={rankG} (expected {(n + (m % 2 == 1)) // 2 if False else rankG})')
        print(f'  known span rank = {rankK} (expected 3m-3 = {3*m-3}); |sum_star p + r*ell_c|^2 = {dep_norm:.3g}')
        print(f'  residual: rank {rankR} (expected {rankG - (3*m-3)}), R^2 = dR: {ok_R}, trace R = {np.trace(R):.6f} (= d*rank: {d*rankR})')
        print(f'  |k_f|^2 range: [{proj_norms.min():.4f}, {proj_norms.max():.4f}], sum = {proj_norms.sum():.4f}, ell-part each {2*(1/(r+1)):.4f}')
    return dict(S=S, G=G, orb=orb, rankG=rankG, rankK=rankK, rankR=rankR, R=R, Proj=Proj)


# ---------------------------------------------------------------------------------------------
# Partial adjacency from (type, Phi) only, and the generic counting constraints
# ---------------------------------------------------------------------------------------------
def partial_adjacency(cfg, mu, tau, Phi):
    """Adjacency with: forced part, block type at c, star-far via Phi (t-symmetrised); far-far = 0."""
    A = cfg.forced_adjacency().astype(np.int64)
    def add(u, v):
        A[cfg.vid[u], cfg.vid[v]] = A[cfg.vid[v], cfg.vid[u]] = 1
    for v in cfg.labels:
        add(cfg.star_p(v), cfg.star_p(mu[v]))
        add(cfg.star_m(cfg.pi(v)), cfg.star_m(cfg.pi(mu[v])))        # t-image
        add(cfg.star_p(v), cfg.star_m(cfg.pi(tau[v])))
        # t-image of the previous edge: [c-, pi v] ~ [c+, tau v] -- covered when v' = tau v
    for z, pair in Phi.items():
        for v in pair:
            add(z, cfg.star_p(v))
            add(cfg.t(z), cfg.star_m(cfg.pi(v)))     # t-image
    return A


def region_indices(cfg):
    R = [cfg.vid[('x',)]] + [cfg.vid[v] for v in cfg.V if v[0] == 'L']
    R += [cfg.vid[v] for v in cfg.starp] + [cfg.vid[v] for v in cfg.starm]
    Fi = [cfg.vid[v] for v in cfg.far]
    return R, Fi


def check_phi_constraints(cfg, mu, tau, Phi, verbose=True):
    """Exact counts within R, upper bounds elsewhere; returns number of violations."""
    A = partial_adjacency(cfg, mu, tau, Phi)
    Rid, Fid = region_indices(cfg)
    A2 = A @ A
    viol = 0
    for a in Rid:
        for b in Rid:
            if a < b:
                target = 1 if A[a, b] else 2
                if A2[a, b] != target:
                    viol += 1
    bad_bound = 0
    for a in Rid:
        for b in Fid:
            target = 1 if A[a, b] else 2
            if A2[a, b] > target:
                bad_bound += 1
    for a in Fid:
        for b in Fid:
            if a < b and A2[a, b] > 2:
                bad_bound += 1
    if verbose:
        print(f'  Phi constraints: exact violations within R = {viol}, bound violations (R,far)+(far,far) = {bad_bound}')
    return viol + bad_bound


if __name__ == '__main__':
    cfg, A = bvls_in_labelling()
    check_srg(A, 243, 22, 1, 2)
    for c in (0, 5):
        cfg = Config(11, c=c)
        _, A = bvls_in_labelling()            # A is in cfg.V order which does not depend on c
        print(f'--- block c = {c}')
        info = validate_linear_algebra(cfg, A)
        mu, tau = block_type(cfg, A)
        Phi = phi_map(cfg, A)
        pairs = set(Phi.values())
        nonmu = sum(1 for p in pairs for v in p if mu[v] in p) // 2
        print(f'  type: mu==tau on all labels? {all(mu[v] == tau[v] for v in cfg.labels)};  Phi bijective onto {len(pairs)} pairs, mu-pairs among them: {nonmu}')
        check_phi_constraints(cfg, mu, tau, Phi)
