"""
Independent model of the involution case of srg(2m^2+1, 2m, 1, 2) (m = 7: Conway 99-graph).

Vertex labelling (Z^m picture, involution t = -1):
  x            : the unique fixed vertex               ('x',)
  (i, s)       : the 2m neighbours of x, line i in 0..m-1, sign s in {0,1}   ('L', i, s)
  [(i,s),(j,u)]: the 2m(m-1) vertices at distance 2, i < j                  ('D', i, s, j, u)
Adjacency forced by lambda=1, mu=2 : x ~ (i,s); (i,0) ~ (i,1); (i,s) ~ [(i,s),(j,u)] and [(j,u),(i,s)].
t swaps (i,s) <-> (i,1-s) and maps [(i,s),(j,u)] -> [(i,1-s),(j,1-u)].

Block c (a line) : labels = the 2(m-1) vertices (e,s), e != c.  Type at c = (mu, tau):
  mu  : perfect matching on labels, [c+,v] ~ [c+, mu v]
  tau : fixed-point-free involution on labels, [c+, v] ~ [c-, pi(tau v)]  (pi = flip sign)
Far roots (w.r.t. c): the D-vertices avoiding line c.  Phi(z) = {v : z ~ [c+, v]} (exactly 2 labels).
"""
import itertools
import numpy as np


class Config:
    def __init__(self, m, c=0):
        self.m = m
        self.c = c
        self.lines = list(range(m))
        self.others = [e for e in range(m) if e != c]
        # labels: (e, s) with e != c
        self.labels = [(e, s) for e in self.others for s in (0, 1)]
        self.lab_index = {v: i for i, v in enumerate(self.labels)}
        self.nlab = len(self.labels)
        # vertices
        V = [('x',)]
        for i in range(m):
            for s in (0, 1):
                V.append(('L', i, s))
        for i in range(m):
            for j in range(i + 1, m):
                for s in (0, 1):
                    for u in (0, 1):
                        V.append(('D', i, s, j, u))
        self.V = V
        self.vid = {v: k for k, v in enumerate(V)}
        self.n = len(V)
        # far roots (D vertices avoiding c), star+ / star- (D vertices containing (c,0)/(c,1))
        self.far = [v for v in V if v[0] == 'D' and v[1] != c and v[3] != c]
        self.starp = []
        self.starm = []
        for v in V:
            if v[0] == 'D':
                ends = self.ends(v)
                if (c, 0) in ends:
                    self.starp.append(v)
                if (c, 1) in ends:
                    self.starm.append(v)
        assert len(self.far) == 2 * (m - 1) * (m - 2)
        assert len(self.starp) == self.nlab == len(self.starm)

    @staticmethod
    def ends(v):
        return [(v[1], v[2]), (v[3], v[4])]

    def D(self, p, q):
        """D vertex with endpoints p=(i,s), q=(j,u)."""
        if p[0] > q[0]:
            p, q = q, p
        assert p[0] != q[0]
        return ('D', p[0], p[1], q[0], q[1])

    def star_p(self, v):   # [c+, v]
        return self.D((self.c, 0), v)

    def star_m(self, v):   # [c-, v]
        return self.D((self.c, 1), v)

    def t(self, v):
        if v[0] == 'x':
            return v
        if v[0] == 'L':
            return ('L', v[1], 1 - v[2])
        return ('D', v[1], 1 - v[2], v[3], 1 - v[4])

    @staticmethod
    def pi(v):
        return (v[0], 1 - v[1])

    def forced_adjacency(self):
        """0/1 matrix of the adjacencies forced by lambda=1, mu=2 around x (all others 0)."""
        A = np.zeros((self.n, self.n), dtype=np.int8)
        x = self.vid[('x',)]
        for v in self.V:
            if v[0] == 'L':
                A[x, self.vid[v]] = A[self.vid[v], x] = 1
                w = self.vid[('L', v[1], 1 - v[2])]
                A[self.vid[v], w] = A[w, self.vid[v]] = 1
            elif v[0] == 'D':
                for p in self.ends(v):
                    q = self.vid[('L',) + p]
                    A[q, self.vid[v]] = A[self.vid[v], q] = 1
        return A


# ----------------------------------------------------------------------------------------
# The BvLS graph srg(243,22,1,2) as a coset graph of the ternary Golay code, in this labelling
# ----------------------------------------------------------------------------------------
def ternary_golay_code():
    """Return the [11,6,5]_3 Golay code as a set (tuples over F_3) and a syndrome map."""
    # generator polynomial g(x) = x^5 + x^4 - x^3 + x^2 - 1  (mod 3); check by computing the code
    g = [2, 0, 1, 2, 1, 1]   # coefficients of x^0..x^5 : -1, 0, 1, -1, 1, 1  (mod 3)
    n, k = 11, 6
    gens = []
    for sh in range(k):
        vec = [0] * n
        for i, coef in enumerate(g):
            vec[i + sh] = coef % 3
        gens.append(vec)
    G = np.array(gens, dtype=np.int64)
    code = set()
    for coefs in itertools.product(range(3), repeat=k):
        w = tuple((np.array(coefs) @ G) % 3)
        code.add(w)
    assert len(code) == 3 ** 6
    wts = sorted({sum(1 for a in w if a) for w in code if any(w)})
    assert min(wts) == 5, wts
    return code


def bvls_graph():
    """Adjacency matrix (243x243) of the coset graph of the ternary Golay code, and coset reps."""
    code = ternary_golay_code()
    code_list = [np.array(w) for w in code]
    # coset representative: the minimum-weight vector (weights <= 2 since the code is perfect)
    n = 11
    # enumerate all 3^11 vectors, assign coset id via canonical rep (lexicographically smallest of min weight)
    # cheaper: build the syndrome via a parity check matrix. We instead use dictionary of coset leaders.
    leaders = {}
    # all vectors of weight <= 2 are leaders (perfect code, covering radius 2)
    cands = [tuple([0] * n)]
    for i in range(n):
        for a in (1, 2):
            v = [0] * n
            v[i] = a
            cands.append(tuple(v))
    for i in range(n):
        for j in range(i + 1, n):
            for a in (1, 2):
                for b in (1, 2):
                    v = [0] * n
                    v[i] = a
                    v[j] = b
                    cands.append(tuple(v))
    assert len(cands) == 243
    coset_of = {}
    for idx, lead in enumerate(cands):
        for w in code_list:
            coset_of[tuple((np.array(lead) + w) % 3)] = idx
    assert len(coset_of) == 3 ** 11
    A = np.zeros((243, 243), dtype=np.int8)
    for idx, lead in enumerate(cands):
        for i in range(n):
            for a in (1, 2):
                v = list(lead)
                v[i] = (v[i] + a) % 3
                jdx = coset_of[tuple(v)]
                assert jdx != idx
                A[idx, jdx] = A[jdx, idx] = 1
    return A, cands


def bvls_in_labelling():
    """Return (cfg, A) with A the BvLS adjacency matrix indexed by cfg.V (m = 11), x = coset 0."""
    A0, cands = bvls_graph()
    m = 11
    cfg = Config(m)
    # map labels: (i, s) -> e_i (s=0) / -e_i (s=1); D vertex -> sum of the two
    def vec(lab):
        v = [0] * m
        v[lab[0]] = 1 if lab[1] == 0 else 2
        return v
    idx_of = {tuple(c): k for k, c in enumerate(cands)}
    perm = []
    for v in cfg.V:
        if v[0] == 'x':
            perm.append(idx_of[tuple([0] * m)])
        elif v[0] == 'L':
            perm.append(idx_of[tuple(vec((v[1], v[2])))])
        else:
            w = (np.array(vec((v[1], v[2]))) + np.array(vec((v[3], v[4])))) % 3
            perm.append(idx_of[tuple(w)])
    perm = np.array(perm)
    A = A0[np.ix_(perm, perm)]
    return cfg, A


def check_srg(A, v, k, lam, mu):
    A = A.astype(np.int64)
    assert A.shape == (v, v)
    assert (A.sum(axis=1) == k).all()
    A2 = A @ A
    J = np.ones((v, v), dtype=np.int64)
    I = np.eye(v, dtype=np.int64)
    assert (A2 == k * I + lam * A + mu * (J - I - A)).all()
    return True


if __name__ == '__main__':
    cfg, A = bvls_in_labelling()
    print('BvLS built:', A.shape, 'srg check', check_srg(A, 243, 22, 1, 2))
    # check forced adjacencies and the involution
    F = cfg.forced_adjacency()
    assert (A[F == 1] == 1).all()
    tperm = np.array([cfg.vid[cfg.t(v)] for v in cfg.V])
    assert (A[np.ix_(tperm, tperm)] == A).all()
    print('forced adjacencies present, t = -1 is an automorphism with 1 fixed point')
