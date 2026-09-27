"""
Per-type solver for the block-local system of the involution case (independent implementation).

Given m and a type (mu, tau) at block c = 0 (labels 0..2m-3, label 2(e-1)+s <-> (e, s), pi(v) = v^1),
search all Phi : far roots -> non-mu pairs with the exact counts K2, K3 and the local bounds, and for
each Phi complete the far-far sign block T through the rank-(rank G - (3m-3)) real residual.
For m = 7 the residual space has dimension 4.

Everything here is derived from lambda = 1, mu = 2 and the t-symmetry only.
"""
import sys, time, json, itertools
import numpy as np
from inv import Config
from analyze import srg_eigs


class TypeInstance:
    def __init__(self, m, mu, tau):
        self.m = m
        self.cfg = Config(m, c=0)
        self.nl = 2 * (m - 1)
        self.mu = list(mu)
        self.tau = list(tau)
        nl = self.nl
        assert len(mu) == nl and len(tau) == nl
        for v in range(nl):
            assert mu[mu[v]] == v and mu[v] != v
            assert tau[tau[v]] == v and tau[v] != v
        self.pi = [v ^ 1 for v in range(nl)]
        self.h = [self.pi[tau[v]] for v in range(nl)]            # [c+,v] ~ [c-, h v]
        self.line = [v // 2 + 1 for v in range(nl)]                # line index 1..m-1
        self.pivot = [mu[v] == tau[v] for v in range(nl)]
        # non-mu pairs
        self.pairs = []
        self.pair_index = {}
        for v in range(nl):
            for w in range(v + 1, nl):
                if mu[v] != w:
                    self.pair_index[(v, w)] = len(self.pairs)
                    self.pairs.append((v, w))
        self.npairs = len(self.pairs)
        # far roots: (a, b) labels on different lines, a < b ; t-image: (pi a, pi b)
        self.roots = []
        self.root_index = {}
        for a in range(nl):
            for b in range(a + 1, nl):
                if self.line[a] != self.line[b]:
                    self.root_index[(a, b)] = len(self.roots)
                    self.roots.append((a, b))
        self.nroots = len(self.roots)
        self.troot = [self.root_index[tuple(sorted((self.pi[a], self.pi[b])))] for (a, b) in self.roots]
        # far orbits: rep = root with a even (sign 0 on the smaller line); a < b so line(a) < line(b)
        self.orbits = [k for k, (a, b) in enumerate(self.roots) if a % 2 == 0]
        self.norb = len(self.orbits)
        assert self.norb == (m - 1) * (m - 2)
        # K2[a][v] and K3[v][w]
        pi, h = self.pi, self.h
        self.K2 = [[2 - (v == a) - (v == pi[a]) - (v == mu[a]) - (h[v] == a) for v in range(nl)] for a in range(nl)]
        mup = [pi[mu[pi[w]]] for w in range(nl)]      # matching at c-
        self.K3 = [[(1 if h[v] == w else 2) - (v == w) - (h[mu[v]] == w) - (h[v] == mup[w]) for w in range(nl)] for v in range(nl)]
        self.valid = all(x >= 0 for row in self.K2 for x in row) and all(x >= 0 for row in self.K3 for x in row)
        # sanity: row sums of K2 = 20 (10 roots x 2 labels) per label a
        for a in range(nl):
            assert sum(self.K2[a]) == 2 * (nl - 2), (a, self.K2[a])
        # ---- linear algebra of the known frame (lines + star(c)) in the s-eigenspace ----
        self.r, self.s, self.d = srg_eigs(m)
        self._build_known_frame()

    # ---------------------------------------------------------------- known frame
    def _build_known_frame(self):
        m, nl, r = self.m, self.nl, self.r
        # orbit coordinates: lines 0..m-1, star orbits P_v (v = 0..nl-1, rep [c+, v])
        nk = m + nl
        S = np.zeros((nk, nk))
        for i in range(m):
            S[i, i] = -1
        # S[line l, star P_v] = A((l,0),[c+,v]) - A((l,0),[c-,pi v]):
        #   line c: [c+,v] contains (c,0): +1 ;  line(v): (l,0) in [c+,v] iff v has sign 0: +1 if v even else -1 (via t-image)
        for v in range(nl):
            P = m + v
            S[0, P] = S[P, 0] = 1
            l = self.line[v]
            S[l, P] = S[P, l] = 1 if v % 2 == 0 else -1
        for v in range(nl):
            for w in range(nl):
                if v != w:
                    # S[P_v,P_w] = A([c+,v],[c+,w]) - A([c+,v],[c-,pi w]) = [mu v = w] - [h v = pi w]
                    S[m + v, m + w] = (1 if self.mu[v] == w else 0) - (1 if self.h[v] == self.pi[w] else 0)
        G = r * np.eye(nk) - S
        self.S_known = S
        self.G_known = G
        w, U = np.linalg.eigh(G)
        keep = w > 1e-9
        self.dimK = int(keep.sum())
        assert self.dimK == 3 * m - 3, self.dimK
        # coordinates of the known vectors: rows of B (nk x dimK) with B B^T = G
        self.B = U[:, keep] * np.sqrt(w[keep])
        self.Bpinv = np.linalg.pinv(self.B)              # dimK x nk ; k = Bpinv^T ... see kvec
        self.Gpinv = np.linalg.pinv(G)
        # frame target on K: sum_f k_f k_f^T = d I_K - sum_known v v^T  (in coordinates)
        self.Omega = self.d * np.eye(self.dimK) - self.B.T @ self.B
        self.resdim = None   # rank(G_full) - dimK, computed from parameters: dim of s-eigenspace^- minus dimK
        # dimension of the antisymmetric s-eigenspace: (mult(s) )/2 ; mult(s) = g
        v = 2 * m * m + 1
        k = 2 * m
        f = ((v - 1) - (2 * k + (v - 1) * (1 - 2)) / (self.r - self.s)) / 2   # multiplicity of r
        g = (v - 1) - f
        self.resdim = int(round(g / 2)) - self.dimK

    def gvec(self, orb_root, sigma):
        """inner products of p_f with the known vectors (lines then star), from the rep root and shadow sigma."""
        m = self.m
        a, b = self.roots[orb_root]
        g = np.zeros(m + self.nl)
        # <p_f, ell_l> = -S[f, L_l] = -(A(z,(l,0)) - A(z,(l,1))) : z contains (line(a), sign a%2) and (line(b), sign b%2)
        for lab in (a, b):
            g[self.line[lab]] = -(1 if lab % 2 == 0 else -1)
        # <p_f, p_{P_v}> = -S[f, P_v] = -sigma(v)
        g[m:] = -np.asarray(sigma, dtype=float)
        return g

    def kvec(self, g):
        """coordinates in R^dimK of the K-component with prescribed inner products g (must be consistent)."""
        k = self.Bpinv @ g            # least-squares solution of B k = g
        return k

    # ---------------------------------------------------------------- shadows
    def shadow(self, pz, ptz):
        """sigma_f from Phi(z) = pair pz, Phi(tz) = pair ptz (pairs as (v,w))."""
        sig = [0] * self.nl
        for v in pz:
            sig[v] += 1
        for v in ptz:
            sig[v] -= 1
        return sig

    def local_ok(self, root, pz, ptz):
        """local bounds for the orbit (z = root, tz) with Phi(z) = pz, Phi(tz) = ptz."""
        a, b = self.roots[root]
        ta, tb = self.pi[a], self.pi[b]
        nl, pi, mu, tau = self.nl, self.pi, self.mu, self.tau
        inz = (a, b)
        intz = (ta, tb)
        # bijection at the orbit: pz != ptz
        if pz == ptz:
            return False
        # cancellation |Phi(z) cap Phi(tz)| <= 1, and a cancelled label must be a non-pivot
        inter = set(pz) & set(ptz)
        if len(inter) > 1:
            return False
        for v in inter:
            if self.pivot[v]:
                return False
        # (label, far) bounds, for z and for tz
        for (z_ends, P, tP) in ((inz, pz, ptz), (intz, ptz, pz)):
            for lab in range(nl):
                if lab in z_ends:
                    if (lab in P) + (pi[lab] in tP) > 1:
                        return False
                else:
                    if (pi[lab] in z_ends) + (lab in P) + (pi[lab] in tP) > 2:
                        return False
            # (star+ [c+,v], far z): [v in z] + [mu v in Phi(z)] + [tau v in Phi(tz)] <= target
            for v in range(nl):
                target = 1 if v in P else 2
                if (v in z_ends) + (mu[v] in P) + (tau[v] in tP) > target:
                    return False
        return True

    def orbit_candidates(self, orb_root):
        cands = []
        for i, pz in enumerate(self.pairs):
            for j, ptz in enumerate(self.pairs):
                if i != j and self.local_ok(orb_root, pz, ptz):
                    cands.append((i, j))
        return cands


# ------------------------------------------------------------------------------------------------
# Residual completion: given assigned far orbits with K-components k_f, residual norms rho_f and
# the allowed T-values per pair, find residual vectors in R^resdim (DFS).
# ------------------------------------------------------------------------------------------------
class Residual:
    def __init__(self, inst, tol=1e-7):
        self.inst = inst
        self.tol = tol
        self.dim = inst.resdim

    def complete(self, ks, rhos, allowedT, max_solutions=1):
        """ks: list of K-coordinates; rhos: residual norms^2; allowedT[i][j]: iterable of allowed T in {-1,0,1}.
        Returns list of (T matrix, residual vectors) solutions found (up to max_solutions)."""
        n = len(ks)
        KK = np.array([[float(ki @ kj) for kj in ks] for ki in ks])
        sols = []
        res = [None] * n
        T = np.zeros((n, n), dtype=int)
        tol = self.tol
        dim = self.dim
        basis = []            # indices of orbits whose residuals form the current basis (Gram-Schmidt frame)
        frame = np.zeros((dim, 0))     # orthonormal frame columns
        coords = [None] * n            # coordinates of residuals in the orthonormal frame (length = frame cols)

        def rec(i, frame, ncols):
            if len(sols) >= max_solutions:
                return
            if i == n:
                sols.append((T.copy(), [c.copy() for c in coords]))
                return
            rho = rhos[i]
            # inner products with previous residuals: R_ij = -T_ij - KK_ij
            # choose T with the basis orbits -> determines the projection of r_i on the frame
            # basis coordinates: previous residual coords c_j (length ncols); we need <r_i, r_j> for j < i.
            # Let C (i x ncols) = coords of previous residuals; unknown c_i (ncols) + possibly a new coordinate.
            prev = list(range(i))
            if ncols == 0:
                # first vector
                if rho < -tol:
                    return
                c = np.zeros(1)
                c[0] = np.sqrt(max(rho, 0.0))
                coords[i] = c
                # extend frame by one column (abstract); ncols becomes 1 if rho > tol
                if rho > tol:
                    rec(i + 1, frame, 1)
                else:
                    coords[i] = np.zeros(0)
                    rec(i + 1, frame, 0)
                return
            C = np.array([np.pad(coords[j], (0, ncols - len(coords[j]))) for j in prev])   # i x ncols
            # choose a set of basis rows: pivot rows of C (rank ncols)
            # enumerate T for all previous j: too many; instead enumerate T for basis rows, deduce c_i, then check others
            # basis rows: greedy independent rows
            q, rr, piv = _qr_pivot(C)
            brows = piv[:ncols]
            options = [allowedT[i][j] for j in brows]
            Cb = C[brows]                     # ncols x ncols, invertible
            Cb_inv = np.linalg.inv(Cb)
            for tvals in itertools.product(*options):
                rhs = np.array([-t - KK[i, j] for t, j in zip(tvals, brows)])
                ci = Cb_inv @ rhs               # coordinates of r_i on the frame
                proj2 = float(ci @ ci)
                extra2 = rho - proj2
                if extra2 < -tol:
                    continue
                if extra2 > tol and ncols >= dim:
                    continue
                # check the other previous orbits
                ok = True
                for j in prev:
                    if j in brows:
                        T[i, j] = T[j, i] = tvals[list(brows).index(j)]
                        continue
                    ip = float(ci @ C[j])
                    tv = -ip - KK[i, j]
                    tr = int(round(tv))
                    if abs(tv - tr) > 1e-6 or tr not in allowedT[i][j]:
                        ok = False
                        break
                    T[i, j] = T[j, i] = tr
                if not ok:
                    continue
                if extra2 > tol:
                    coords[i] = np.append(ci, np.sqrt(extra2))
                    rec(i + 1, frame, ncols + 1)
                else:
                    coords[i] = ci.copy()
                    rec(i + 1, frame, ncols)
                if len(sols) >= max_solutions:
                    return
        rec(0, frame, 0)
        return sols


def _qr_pivot(C):
    """return (Q, R, piv) with piv the row indices giving a maximal independent set (greedy)."""
    # greedy: pick rows increasing rank
    rows = []
    cur = np.zeros((0, C.shape[1]))
    for idx in range(C.shape[0]):
        cand = np.vstack([cur, C[idx]])
        if np.linalg.matrix_rank(cand, tol=1e-8) > cur.shape[0]:
            rows.append(idx)
            cur = cand
        if cur.shape[0] == C.shape[1]:
            break
    return None, None, rows


# ------------------------------------------------------------------------------------------------
# Phi search (orbit by orbit, lines in order), with K2/K3 budgets and pairwise (far,far) bounds
# ------------------------------------------------------------------------------------------------
class PhiSearch:
    def __init__(self, inst, use_residual=True, residual_every_line=True, verbose=1, node_limit=None, time_limit=None):
        self.inst = inst
        self.use_residual = use_residual
        self.verbose = verbose
        self.node_limit = node_limit
        self.time_limit = time_limit
        inst_ = inst
        nl = inst.nl
        # orbit order: by line of the smaller-line label, then by the other line
        orbs = sorted(inst.orbits, key=lambda k: (inst.line[inst.roots[k][0]], inst.line[inst.roots[k][1]], inst.roots[k]))
        self.order = orbs
        self.cands = {k: inst.orbit_candidates(k) for k in orbs}
        # roots containing label a: for K2 accounting
        self.roots_with = [[k for k, (a, b) in enumerate(inst.roots) if lab in (a, b)] for lab in range(nl)]
        # state
        self.pair_used = [False] * inst.npairs
        self.K2rem = [row[:] for row in inst.K2]
        self.K3rem = [row[:] for row in inst.K3]
        self.roots_left = [len(r) for r in self.roots_with]   # unassigned roots containing label a
        self.assign = {}      # root -> pair index
        self.nodes = 0
        self.solutions = []
        self.level_nodes = [0] * (len(orbs) + 1)
        self.t0 = time.time()
        self.residual = Residual(inst)
        self.line_end = {}    # position in order after which a line is complete
        for pos, k in enumerate(orbs):
            e = inst.line[inst.roots[k][0]]
            self.line_end[e] = pos
        self.stop = False

    def _apply(self, root, pz_idx, ptz_idx, sign):
        inst = self.inst
        pz = inst.pairs[pz_idx]
        ptz = inst.pairs[ptz_idx]
        troot = inst.troot[root]
        for (rt, P, tP) in ((root, pz, ptz), (troot, ptz, pz)):
            a, b = inst.roots[rt]
            for lab in (a, b):
                self.roots_left[lab] -= sign
                for v in P:
                    self.K2rem[lab][v] -= sign
            for v in P:
                for w in tP:
                    self.K3rem[v][inst.pi[w]] -= sign
        self.pair_used[pz_idx] = (sign == 1)
        self.pair_used[ptz_idx] = (sign == 1)
        if sign == 1:
            self.assign[root] = pz_idx
            self.assign[troot] = ptz_idx
        else:
            del self.assign[root]
            del self.assign[troot]

    def _feasible_counts(self):
        inst = self.inst
        for lab in range(inst.nl):
            row = self.K2rem[lab]
            s = 0
            for x in row:
                if x < 0:
                    return False
                s += x
            if s != 2 * self.roots_left[lab]:
                return False
        for row in self.K3rem:
            for x in row:
                if x < 0:
                    return False
        return True

    def _pair_ok(self, root, pz, ptz):
        """(far,far) bounds against already assigned orbits; returns dict of forced non-adjacencies."""
        inst = self.inst
        troot = inst.troot[root]
        a, b = inst.roots[root]
        ta, tb = inst.roots[troot]
        for rt2, p2 in self.assign.items():
            P2 = inst.pairs[p2]
            c2, d2 = inst.roots[rt2]
            for (zends, P, tP) in ((inst.roots[root], pz, ptz), (inst.roots[troot], ptz, pz)):
                shared = len(set(zends) & {c2, d2})
                # common star+ : Phi(z) cap Phi(z2) ; common star- : Phi(tz) cap Phi(t z2)
                tp2 = inst.pairs[self.assign[inst.troot[rt2]]]
                common = shared + len(set(P) & set(P2)) + len(set(tP) & set(tp2))
                if common > 2:
                    return False
        return True

    def run(self):
        self._rec(0)
        return self.solutions

    def _rec(self, pos):
        if self.stop:
            return
        inst = self.inst
        self.nodes += 1
        self.level_nodes[pos] += 1
        if self.node_limit and self.nodes > self.node_limit:
            self.stop = True
            return
        if self.time_limit and time.time() - self.t0 > self.time_limit:
            self.stop = True
            return
        if pos == len(self.order):
            sol = dict(self.assign)
            if self.use_residual:
                ok = self._residual_check(final=True)
                if not ok:
                    return
            self.solutions.append(sol)
            if self.verbose:
                print('  SOLUTION found at node', self.nodes)
            return
        root = self.order[pos]
        for (pi_idx, pj_idx) in self.cands[root]:
            if self.pair_used[pi_idx] or self.pair_used[pj_idx]:
                continue
            pz, ptz = inst.pairs[pi_idx], inst.pairs[pj_idx]
            if not self._pair_ok(root, pz, ptz):
                continue
            self._apply(root, pi_idx, pj_idx, +1)
            if self._feasible_counts():
                go = True
                if self.use_residual:
                    e = inst.line[inst.roots[root][0]]
                    if self.line_end[e] == pos:
                        go = self._residual_check(final=False)
                if go:
                    self._rec(pos + 1)
            self._apply(root, pi_idx, pj_idx, -1)
            if self.stop:
                return

    def _residual_check(self, final):
        inst = self.inst
        orbs = [k for k in self.order if k in self.assign]
        ks, rhos = [], []
        for k in orbs:
            pz = inst.pairs[self.assign[k]]
            ptz = inst.pairs[self.assign[inst.troot[k]]]
            sig = inst.shadow(pz, ptz)
            g = inst.gvec(k, sig)
            kv = inst.kvec(g)
            ks.append(kv)
            rhos.append(inst.r - float(kv @ kv))
            if rhos[-1] < -1e-7:
                return False
        if sum(rhos) > inst.d * inst.resdim + 1e-6:
            return False
        # allowed T values from vertex-level forcing (far,far): if known common neighbours = 2 then non-adjacent
        n = len(orbs)
        allowed = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                allowed[i][j] = self._allowed_T(orbs[i], orbs[j])
                if not allowed[i][j]:
                    return False
        sols = self.residual.complete(ks, rhos, allowed, max_solutions=1)
        return len(sols) > 0

    def _allowed_T(self, f, g):
        """allowed values of T_fg = A(z_f,z_g) - A(z_f, t z_g) given the (far,far) common-neighbour bounds."""
        inst = self.inst
        zf, zg, tzg = f, g, inst.troot[g]
        def known_common(z1, z2):
            a1 = inst.roots[z1]; a2 = inst.roots[z2]
            P1 = inst.pairs[self.assign[z1]]; P2 = inst.pairs[self.assign[z2]]
            tP1 = inst.pairs[self.assign[inst.troot[z1]]]; tP2 = inst.pairs[self.assign[inst.troot[z2]]]
            return len(set(a1) & set(a2)) + len(set(P1) & set(P2)) + len(set(tP1) & set(tP2))
        c1 = known_common(zf, zg)      # z_f vs z_g
        c2 = known_common(zf, tzg)     # z_f vs t z_g
        if c1 > 2 or c2 > 2:
            return []
        adj1 = [0, 1] if c1 < 2 else [0]      # if c1 == 1 adjacency allowed (lambda=1: exactly that one) ... c1<=1 ok
        adj2 = [0, 1] if c2 < 2 else [0]
        # lambda = 1: if adjacent, known common must be <= 1 (already); also both adjacent impossible? z_f ~ z_g and z_f ~ t z_g
        # is allowed (C=2, partner) only if... then T = 0.
        vals = set()
        for x in adj1:
            for y in adj2:
                vals.add(x - y)
        return sorted(vals)


def load_type(path):
    d = json.load(open(path))
    return d['mu'], d['tau']


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../structures/etl_detsq_00.json'
    mu, tau = load_type(path)
    inst = TypeInstance(7, mu, tau)
    print('type valid (K2,K3 >= 0):', inst.valid, ' pivots:', sum(inst.pivot), ' dimK', inst.dimK, ' resdim', inst.resdim)
    srch = PhiSearch(inst, use_residual='--nores' not in sys.argv, verbose=1,
                     time_limit=float(sys.argv[2]) if len(sys.argv) > 2 else 60)
    ncand = [len(srch.cands[k]) for k in srch.order]
    print('candidates per orbit (in search order):', ncand)
    sols = srch.run()
    print('nodes', srch.nodes, 'time %.1fs' % (time.time() - srch.t0), 'stopped' if srch.stop else 'COMPLETE', 'solutions', len(sols))
    print('nodes per level:', srch.level_nodes)
