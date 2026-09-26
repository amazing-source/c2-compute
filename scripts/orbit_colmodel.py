"""ORBIT-level column model for a FIXED skeleton (note 40): Sigma = empty, one-block squares, every orbit given by its triple
(pivot i, free block j, third block k); D(orbit (i,j,k)) = the orbit (i,k,j) (digons: matched in order, WLOG).
Written independently of the 'structures' code (same identity, own enumeration, own rows, own exact check).

Orbit-level column of Z = (i,j,k): vector q in {0,1,2}^42 (q[W] = number of neighbours of a label of Z in W), with
  q[D(Z)] = 2, q = 0 on every other orbit of a pair {i,r} (F1) and on the sister (n_s = 0);
  free part: two distinct orbits F1, F2 free at j, not the sister, not both in the pair {j,k} (typed rule, note 30);
  far part: pair-level far row v on the 10 pairs of the 5 blocks t != i,j with block sums 4 - 2[t = k] - n_t (all proved,
  notes 27-31); v = 2 puts 1 on both orbits (a far pair never contains D(Z)); v = 1 puts 1 on ONE of the two orbits.
Rows:
  one[Z] = 1 ;  S[W]: sum_Z q_Z(W) = 12 ;  C[a,b]: q_a(b) - q_b(a) = 0 (a < b, whenever one side is variable) ;
  I[a,b] (a <= b, doubled): sum_Z 2 q_Z(a) q_Z(b) + q_a(b) + q_b(a) = 2 (12[a = b] + 4 - Sh(a,b)),  Sh = 2 same pair, else
     number of common blocks  (Q^2 + Q = 12 I + 4 J - Sh on the sigma-quotient; eigenvalues 3, -4 off the block space) ;
  cut[c,S]: edges of the free graph R_c inside S <= |S| - 1 for every S whose size is not a union of even cycles >= 4.
Phase-1 LP by column generation (exact pricing: every column enumerated, vectorised); EXACT check of the last duals in
integers over ALL columns: y^T b > sum_Z max_c (A^T y)_c + sum_le max(0, y_r rhs_r)  =>  no solution.
Usage: python orbit_colmodel.py <survivors json> [indices]"""
import sys, json, itertools, time
from collections import defaultdict
import numpy as np
import highspy
from scipy.sparse import csc_matrix
sys.path.insert(0, "scripts")
from gen_colmodel_lam import far_solutions
N = 7

def cycle_union_sizes(f):
    """sizes of sub-unions of the cycles of a 2-regular graph on f vertices with even cycles >= 4"""
    out = set()
    def parts(n, mn):
        if n == 0: yield []; return
        for p in range(mn, n + 1, 2):
            if p >= 4 and n - p >= 0:
                for rest in parts(n - p, p): yield [p] + rest
    for pr in parts(f, 4):
        for r in range(len(pr) + 1):
            for sub in itertools.combinations(pr, r): out.add(sum(sub))
    return out

class OrbitModel:
    def __init__(self, triples, use_I=True):
        self.use_I = use_I
        self.T = [tuple(t) for t in triples]; n = self.n = len(self.T); assert n == 42
        self.pair = [frozenset(t[:2]) for t in self.T]
        self.orbs_of = defaultdict(list)
        for s in range(n): self.orbs_of[self.pair[s]].append(s)
        assert all(len(v) == 2 for v in self.orbs_of.values()) and len(self.orbs_of) == 21
        self.sis = [next(u for u in self.orbs_of[self.pair[s]] if u != s) for s in range(n)]
        byt = defaultdict(list)
        for s, t in enumerate(self.T): byt[t].append(s)
        self.D = [byt[(i, k, j)][byt[(i, j, k)].index(s)] for s, (i, j, k) in enumerate(self.T)]
        assert all(self.D[self.D[s]] == s and self.D[s] != s for s in range(n))
        self.free = {c: [s for s in range(n) if self.T[s][1] == c] for c in range(N)}
        self.Sh = np.array([[2 if self.pair[a] == self.pair[b] else len(self.pair[a] & self.pair[b]) for b in range(n)] for a in range(n)])
        self.build_columns(); self.build_rows()

    def build_columns(self):
        """per orbit Z: int8 matrix of all its orbit-level q vectors"""
        self.cols = []
        for Z, (i, j, k) in enumerate(self.T):
            cand = [F for F in self.free[j] if F != Z and F != self.sis[Z]]
            far_blocks = [t for t in range(N) if t not in (i, j)]
            blocks_of = lambda F: next(iter(self.pair[F] - {j}))
            mats = []
            for F1, F2 in itertools.combinations(cand, 2):
                r1, r2 = blocks_of(F1), blocks_of(F2)
                if r1 == r2 == k: continue                     # typed rule n_k <= 1
                nv = defaultdict(int); nv[r1] += 1; nv[r2] += 1
                deg = {t: 4 - 2 * (t == k) - nv[t] for t in far_blocks}
                if min(deg.values()) < 0: continue
                prs, sols = far_solutions(far_blocks, deg)
                base = np.zeros(42, dtype=np.int8); base[self.D[Z]] = 2; base[F1] += 1; base[F2] += 1
                for s in sols:
                    v = base.copy(); opts = []
                    for R, val in zip(prs, s):
                        o1, o2 = self.orbs_of[R]
                        if val == 2: v[o1] += 1; v[o2] += 1
                        elif val == 1: opts.append((o1, o2))
                    m = len(opts)
                    if m == 0: mats.append(v[None, :]); continue
                    ch = ((np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1).astype(bool)
                    blk = np.repeat(v[None, :], 1 << m, axis=0)
                    o1 = np.array([a for a, b in opts]); o2 = np.array([b for a, b in opts])
                    for t in range(m):
                        blk[~ch[:, t], o1[t]] += 1; blk[ch[:, t], o2[t]] += 1
                    mats.append(blk)
            M = np.concatenate(mats) if mats else np.zeros((0, 42), dtype=np.int8)
            assert (M.sum(axis=1) == 12).all()
            self.cols.append(M)

    def build_rows(self):
        n = self.n; self.rname = []; self.rkind = []; self.rrhs = []; self.rows = {}
        def add(name, kind, rhs):
            self.rows[name] = len(self.rname); self.rname.append(name); self.rkind.append(kind); self.rrhs.append(rhs)
        for Z in range(n): add(('one', Z), 'eq', 1)
        for W in range(n): add(('S', W), 'eq', 12)
        var = [set(np.nonzero(self.cols[a].max(axis=0) != self.cols[a].min(axis=0))[0].tolist()) for a in range(n)]
        for a, b in itertools.combinations(range(n), 2):
            if b in var[a] or a in var[b]: add(('C', a, b), 'eq', 0)
            elif len(self.cols[a]) and len(self.cols[b]) and self.cols[a][0, b] != self.cols[b][0, a]:
                print("  constant coupling conflict", a, b, self.T[a], self.T[b]); add(('C', a, b), 'eq', 0)
        if self.use_I == 'pair':
            self.PL = sorted(self.orbs_of, key=lambda R: tuple(sorted(R))); self.PLi = {R: t for t, R in enumerate(self.PL)}
            for x, R1 in enumerate(self.PL):
                for R2 in self.PL[x:]:
                    rhs = sum(12 * (a == b) + 4 - int(self.Sh[a, b]) for a in self.orbs_of[R1] for b in self.orbs_of[R2])
                    add(('IP', R1, R2), 'eq', 2 * rhs)
        elif self.use_I:
            irel = __import__('os').environ.get('IREL')
            self.Ikept = set()
            for a in range(n):
                for b in range(a, n):
                    rel = 'diag' if a == b else ('sister' if self.pair[a] == self.pair[b] else ('share' if self.pair[a] & self.pair[b] else 'disjoint'))
                    if irel and rel not in irel.split(','): continue
                    add(('I', a, b), 'eq', 2 * (12 * (a == b) + 4 - int(self.Sh[a, b]))); self.Ikept.add((a, b))
        self.cuts = []; self.lazy_pool = []
        lazy = __import__('os').environ.get('LAZYCUTS') is not None
        for c in range(N):
            F = self.free[c]; f = len(F)
            if f == 0: continue
            bad = cycle_union_sizes(f)
            for size in range(3, f):
                if size in bad: continue
                for S in itertools.combinations(F, size):
                    if lazy and size > 3: self.lazy_pool.append((c, S)); continue      # separated later (solve_lazy)
                    add(('cut', c, S), 'le', 2 * (size - 1)); self.cuts.append((c, S))
        # static linear coefficient maps: lin_rows[Z][W] = list of row indices whose coefficient is q_Z(W) (+1) or -q_Z(W)
        self.linmap = []
        for Z in range(n):
            ent = defaultdict(list)
            for W in range(n):
                ent[W].append((self.rows['S', W], 1))
                if W == Z: continue
                a, b = min(Z, W), max(Z, W)
                if ('C', a, b) in self.rows: ent[W].append((self.rows['C', a, b], 1 if Z == a else -1))
                if self.use_I == 'pair':
                    R1, R2 = self.pair[Z], self.pair[W]
                    if R1 != R2: ent[W].append((self.rows['IP', *sorted((R1, R2), key=lambda R: self.PLi[R])], 1))
                elif self.use_I and (a, b) in self.Ikept: ent[W].append((self.rows['I', a, b], 1))
            self.linmap.append(ent)
        self.pairrows = defaultdict(list)
        if getattr(self, 'use_sis5', False):
            for c in range(N):
                F = self.free[c]
                for U in F:
                    Up = self.sis[U]
                    if U < Up and Up in F:
                        r = len(self.rname); name = ('sis5', c, U, Up); self.rows[name] = r
                        self.rname.append(name); self.rkind.append('le'); self.rrhs.append(1)
                        for Z in F:
                            if Z not in (U, Up): self.pairrows[Z].append((r, U, Up))
        cutrows = defaultdict(list)
        for (c, S) in self.cuts:
            r = self.rows['cut', c, S]
            for Z in S:
                for W in S:
                    if W != Z: self.linmap[Z][W].append((r, 1))
        self.Iidx = np.zeros((n, n), dtype=np.int64)
        if self.use_I == 'pair':
            self.Pagg = np.zeros((21, n), dtype=np.int64)
            for a in range(n): self.Pagg[self.PLi[self.pair[a]], a] = 1
            self.IPidx = np.zeros((21, 21), dtype=np.int64)
            for x, R1 in enumerate(self.PL):
                for w, R2 in enumerate(self.PL[x:], x): self.IPidx[x, w] = self.IPidx[w, x] = self.rows['IP', R1, R2]
        elif self.use_I:
            self.Imask = np.zeros((n, n), dtype=bool)
            for (a, b) in self.Ikept: self.Iidx[a, b] = self.Iidx[b, a] = self.rows['I', a, b]; self.Imask[a, b] = self.Imask[b, a] = True

    def score_parts(self, y, Z):
        """const, lin (42), Ysym (42x42) such that (A^T y)_c = const + q.lin + q^T Y q for every column q of Z"""
        n = self.n
        if self.use_I == 'pair':
            yP = np.array(y[self.IPidx], dtype=y.dtype); np.fill_diagonal(yP, 2 * np.diag(y[self.IPidx]))
            Y = self.Pagg.T.astype(y.dtype) @ yP @ self.Pagg.astype(y.dtype)
        elif self.use_I:
            yI = np.where(self.Imask, y[self.Iidx], 0).astype(y.dtype); Y = np.array(yI, dtype=y.dtype); np.fill_diagonal(Y, 2 * np.diag(yI))
        else: Y = np.zeros((n, n), dtype=y.dtype)
        lin = np.zeros(n, dtype=y.dtype)
        for W, lst in self.linmap[Z].items():
            for r, c in lst: lin[W] += c * y[r]
        return y[self.rows['one', Z]], lin, Y

    def extra_scores(self, y, Z, M):
        out = 0
        for r, U, Up in self.pairrows.get(Z, ()):
            out = out + y[r] * ((M[:, U] == 1) & (M[:, Up] == 1))
        return out

    def entries(self, Z, q):
        ent = [(self.rows['one', Z], 1)] + [(r, 1) for r, U, Up in self.pairrows.get(Z, ()) if q[U] == 1 and q[Up] == 1]
        for W in np.nonzero(q)[0]:
            for r, c in self.linmap[Z][int(W)]: ent.append((r, c * int(q[W])))
        if self.use_I == 'pair':
            phi = self.Pagg @ q.astype(np.int64)
            for x in range(21):
                for w in range(x, 21):
                    c = 2 * int(phi[x]) * int(phi[w])
                    if c: ent.append((self.IPidx[x, w], c))
        nz = np.nonzero(q)[0] if self.use_I is True else []
        for x in range(len(nz)):
            for w in range(x, len(nz)):
                a, b = int(nz[x]), int(nz[w])
                if (min(a, b), max(a, b)) in self.Ikept: ent.append((self.rows['I', min(a, b), max(a, b)], 2 * int(q[a]) * int(q[b])))
        d = defaultdict(int)
        for r, c in ent: d[r] += c
        return [(r, c) for r, c in d.items() if c]

    def master(self, cols):
        m = len(self.rname); nx = len(cols)
        le = [r for r in range(m) if self.rkind[r] == 'le']; eq = [r for r in range(m) if self.rkind[r] == 'eq']
        ri, ci, data = [], [], []
        for c, ent in enumerate(cols):
            for r, v in ent: ri.append(r); ci.append(c); data.append(float(v))
        for t, r in enumerate(le): ri.append(r); ci.append(nx + t); data.append(1.0)
        for t, r in enumerate(eq):
            ri += [r, r]; ci += [nx + len(le) + 2 * t, nx + len(le) + 2 * t + 1]; data += [1.0, -1.0]
        nc = nx + len(le) + 2 * len(eq)
        A = csc_matrix((data, (ri, ci)), shape=(m, nc))
        lp = highspy.HighsLp(); lp.num_col_ = nc; lp.num_row_ = m
        lp.col_cost_ = np.concatenate([np.zeros(nx + len(le)), np.ones(2 * len(eq))])
        lp.col_lower_ = np.zeros(nc)
        lp.col_upper_ = np.concatenate([np.full(nx, highspy.kHighsInf), np.array([self.rrhs[r] for r in le], dtype=float), np.full(2 * len(eq), highspy.kHighsInf)])
        b = np.array(self.rrhs, dtype=float); lp.row_lower_ = b; lp.row_upper_ = b
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("solver", "ipm"); h.setOptionValue("run_crossover", "off")
        h.passModel(lp); h.run()
        return h

    def master_ws(self, cols, start):
        """warm-started simplex master: build once (slacks + artificials + columns), then addCols for cols[start:]"""
        m = len(self.rname)
        def csc(block):
            starts = [0]; idx = []; val = []
            for ent in block:
                for r, v in ent: idx.append(r); val.append(float(v))
                starts.append(len(idx))
            return np.array(starts[:-1], dtype=np.int32), np.array(idx, dtype=np.int32), np.array(val, dtype=np.float64)
        if start == 0 or not hasattr(self, '_h'):
            le = [r for r in range(m) if self.rkind[r] == 'le']; eq = [r for r in range(m) if self.rkind[r] == 'eq']
            h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("solver", "simplex")
            if __import__('os').environ.get('WSPRIMAL'): h.setOptionValue("simplex_strategy", 4)   # primal: resumes from the old basis
            inf = highspy.kHighsInf
            b = np.array(self.rrhs, dtype=float)
            h.addRows(m, b, b, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            base = [[(r, 1)] for r in le] + [[(r, 1)] for r in eq] + [[(r, -1)] for r in eq]
            st, ix, vl = csc(base)
            cost = np.concatenate([np.zeros(len(le)), np.ones(2 * len(eq))])
            lo = np.zeros(len(base)); up = np.concatenate([np.array([self.rrhs[r] for r in le], dtype=float), np.full(2 * len(eq), inf)])
            h.addCols(len(base), cost, lo, up, len(ix), st, ix, vl)
            self._h = h; start = 0
        new = cols[start:]
        if new:
            st, ix, vl = csc(new)
            self._h.addCols(len(new), np.zeros(len(new)), np.zeros(len(new)), np.full(len(new), highspy.kHighsInf), len(ix), st, ix, vl)
        self._h.run()
        return self._h

    def solve(self, tlimit=280, per=3, verbose=True, init=None):
        """init: optional list of (Z, q-bytes) from a related model (column reuse); kept if valid here.
        env PER overrides per (columns added per orbit and iteration: the master's cost is dominated by its rows, so more columns
        per iteration means fewer master solves)"""
        per = int(__import__('os').environ.get('PER', per))
        t0 = time.time(); cols = []; keys = set(); self.colinfo = []; ckeys = []
        if init:
            look = {}
            for Z, qb in init:
                if Z not in look: look[Z] = {self.cols[Z][t].tobytes() for t in range(len(self.cols[Z]))}
                if qb in look[Z] and (Z, qb) not in keys:
                    q = np.frombuffer(qb, dtype=np.int8)
                    keys.add((Z, qb)); cols.append(self.entries(Z, q)); self.colinfo.append((Z, int(np.argmax(q == 2)))); ckeys.append((Z, qb))
        # start: 3 random columns per orbit
        rng = np.random.default_rng(0)
        for Z in range(self.n):
            M = self.cols[Z]
            for t in rng.choice(len(M), size=min(3, len(M)), replace=False):
                keys.add((Z, M[t].tobytes())); cols.append(self.entries(Z, M[t])); self.colinfo.append((Z, int(np.argmax(M[t] == 2)))); ckeys.append((Z, M[t].tobytes()))
        it = 0
        while True:
            it += 1
            tm = time.time()
            if __import__('os').environ.get('WS'): h = self.master_ws(cols, getattr(self, '_ncols_done', 0) if it > 1 else 0); self._ncols_done = len(cols)
            else: h = self.master(cols)
            obj = h.getInfo().objective_function_value; tm = time.time() - tm
            y = np.array(h.getSolution().row_dual)
            added = 0; best = 0.0; tp = time.time(); sumbest = 0.0
            lowmem = __import__('os').environ.get('LOWMEM') is not None
            if not lowmem and not hasattr(self, '_colsf'): self._colsf = [M.astype(np.float32) for M in self.cols]

            def price(yv):
                """scores of every column at the duals yv: (sum over orbits of the max score, candidates (Z, t, score))"""
                summax = 0.0; sumpos = 0.0; cand = []
                for Z in range(self.n):
                    const, lin, Y = self.score_parts(yv, Z)
                    if lowmem:                                   # no float32 cache: convert and score in chunks (RAM)
                        lf, Yf = lin.astype(np.float32), Y.astype(np.float32); parts = []
                        for c0 in range(0, len(self.cols[Z]), CHUNK):
                            M = self.cols[Z][c0:c0 + CHUNK].astype(np.float32)
                            parts.append(const + M @ lf + np.einsum('ij,ij->i', M @ Yf, M))
                        sc = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                    else:
                        M = self._colsf[Z]
                        sc = const + M @ lin.astype(np.float32) + np.einsum('ij,ij->i', M @ Y.astype(np.float32), M)
                    sc = sc.astype(np.float64) + self.extra_scores(yv, Z, self.cols[Z])
                    if not len(sc): continue
                    summax += float(sc.max()); sumpos += max(0.0, float(sc.max()))
                    for t in np.argsort(-sc)[:per]:
                        if sc[t] > 1e-5: cand.append((Z, int(t), float(sc[t])))
                return summax, cand, sumpos

            if not hasattr(self, '_rhsv'):
                self._rhsv = np.array(self.rrhs, dtype=float); self._lemask = np.array([k == 'le' for k in self.rkind])
            margin = lambda yv, summax: float(yv @ self._rhsv - np.maximum(yv[self._lemask] * self._rhsv[self._lemask], 0).sum() - summax)
            alpha = float(__import__('os').environ.get('SMOOTH', 0))
            if alpha > 0:                                        # Wentges smoothing: price at a blend of the best point and the master duals
                if it == 1 or getattr(self, '_yc', None) is None or len(self._yc) != len(y): self._yc, self._Mc = y.copy(), -np.inf
                ys = alpha * self._yc + (1 - alpha) * y
                smx, cand, _ = price(ys); Ms = margin(ys, smx)
                if Ms > self._Mc: self._yc, self._Mc = ys, Ms
                if not cand:                                     # mispricing: fall back to the master duals
                    smx, cand, _ = price(y); Mm = margin(y, smx)
                    if Mm > self._Mc: self._yc, self._Mc = y.copy(), Mm
                sumbest = obj - self._Mc if np.isfinite(self._Mc) else obj
            else:                                                # default (unchanged): price at the master duals
                smx, cand, sumbest = price(y)
            for Z, t, sct in cand:
                if obj > 1e-7:
                    key = (Z, self.cols[Z][t].tobytes())
                    if key in keys: continue
                    keys.add(key); cols.append(self.entries(Z, self.cols[Z][t])); added += 1; best = max(best, sct)
                    self.colinfo.append((Z, int(np.argmax(self.cols[Z][t] == 2)))); ckeys.append(key)
            if verbose: print("  iter %d obj %.5f cols %d added %d best %.3g lagr %.3f %.0fs (master %.1fs, pricing %.1fs)" % (it, obj, len(cols), added, best, obj - sumbest, time.time() - t0, tm, time.time() - tp)); sys.stdout.flush()
            ycert = self._yc if alpha > 0 else y
            if obj > 1e-7 and obj - sumbest > 0.05:          # Lagrangian (Farkas) bound positive: exact check now, stop if valid
                ok, _b = self.verify(ycert)
                if ok:
                    self.done = True; self.final_keys = list(ckeys)
                    self.last_x = np.array(h.getSolution().col_value)[:len(cols)] if not __import__('os').environ.get('WS') else None
                    return obj, ycert
            if added == 0 or time.time() - t0 > tlimit:
                self.done = added == 0
                self.last_x = np.array(h.getSolution().col_value)[:len(cols)] if not __import__('os').environ.get('WS') else None
                self.final_keys = list(ckeys)
                return obj, y

    def add_cut_row(self, c, S):
        r = len(self.rname); name = ('cut', c, S); self.rows[name] = r
        self.rname.append(name); self.rkind.append('le'); self.rrhs.append(2 * (len(S) - 1)); self.cuts.append((c, S))
        for Z in S:
            for W in S:
                if W != Z: self.linmap[Z][W].append((r, 1))

    def solve_lazy(self, tlimit=280, per=3, verbose=True, init=None):
        """column generation + separation of the pooled subtour cuts (sizes > 3) whenever the master is feasible"""
        t0 = time.time(); total_added_cuts = 0
        while True:
            obj, y = self.solve(tlimit=max(30, tlimit - (time.time() - t0)), per=per, verbose=verbose, init=init)
            if not self.done or obj > 1e-7 or not self.lazy_pool: return obj, y
            # feasible master: separate the pooled cuts with the primal solution
            x = self.last_x; keys = self.final_keys
            if x is None or len(keys) != len(x): return obj, y
            E = defaultdict(float)
            for (Z, qb), xv in zip(keys, x):
                if xv <= 1e-9: continue
                q = np.frombuffer(qb, dtype=np.int8)
                for F in np.nonzero(q == 1)[0]: E[(Z, int(F))] += xv
            viol = []
            for (c, S) in self.lazy_pool:
                val = sum(E.get((Z, F), 0.0) for Z in S for F in S if F != Z)
                if val > 2 * (len(S) - 1) + 1e-6: viol.append((val - 2 * (len(S) - 1), c, S))
            if not viol: return obj, y
            viol.sort(reverse=True)
            for _, c, S in viol[:200]:
                self.add_cut_row(c, S); self.lazy_pool.remove((c, S))
            total_added_cuts += min(200, len(viol))
            if verbose: print("  lazy cuts: %d violated, %d added (total %d)" % (len(viol), min(200, len(viol)), total_added_cuts)); sys.stdout.flush()
            init = keys                                   # restart CG from the current columns (entries rebuilt)

    def verify(self, y, den=10 ** 6):
        yi = np.array([int(round(v * den)) for v in y], dtype=np.int64)
        yb = int(sum(int(yi[r]) * int(self.rrhs[r]) for r in range(len(yi))))
        UB = 0; LB = 0
        for r in range(len(yi)):
            if self.rkind[r] == 'le':
                s = int(yi[r]) * int(self.rrhs[r]); UB += max(s, 0); LB += min(s, 0)
        for Z in range(self.n):
            const, lin, Y = self.score_parts(yi, Z)
            if len(self.cols[Z]) == 0: continue
            hi_, lo_ = None, None
            for c0 in range(0, len(self.cols[Z]), CHUNK):             # chunks: bounded RAM on large column sets
                Mi = self.cols[Z][c0:c0 + CHUNK]; M = Mi.astype(np.int64)
                sc = int(const) + M @ lin + np.einsum('ij,ij->i', M @ Y, M) + np.asarray(self.extra_scores(yi, Z, Mi), dtype=np.int64)
                hi_ = int(sc.max()) if hi_ is None else max(hi_, int(sc.max())); lo_ = int(sc.min()) if lo_ is None else min(lo_, int(sc.min()))
            UB += hi_; LB += lo_
        return (yb > UB or yb < LB), (yb / den, LB / den, UB / den)

CHUNK = 400000                                        # rows per chunk in pricing (LOWMEM) and verify

def save_cert(g, y, path, model, spec, bounds, den=10 ** 6):
    """certificate: model type + spec (enough to rebuild the orbit labelling), integer multipliers with row labels"""
    import gzip, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yi = [int(round(v * den)) for v in y]
    json.dump({"model": model, "spec": spec, "den": den, "bounds": bounds,
               "rows": [[repr(nm), g.rkind[r], g.rrhs[r], yi[r]] for r, nm in enumerate(g.rname) if yi[r]]}, gzip.open(path, "wt"))

if __name__ == "__main__":
    S = json.load(open(sys.argv[1]))
    idx = [int(a) for a in sys.argv[2:]] or list(range(len(S)))
    for t in idx:
        s = S[t]; A = s["A"]
        trip = [tuple(z) for z in s["leaf"]] + [(A, x, y) for (x, y) in s["Adg"]] + [(A, y, x) for (x, y) in s["Adg"]]
        t0 = time.time(); g = OrbitModel(trip, use_I={'none': False, 'pair': 'pair'}.get(__import__('os').environ.get('MODE'), True))
        print("survivor %d (%s job %d): columns per orbit min %d max %d total %d, rows %d, built %.0fs" % (t, s["tag"], s["job"],
              min(len(M) for M in g.cols), max(len(M) for M in g.cols), sum(len(M) for M in g.cols), len(g.rname), time.time() - t0)); sys.stdout.flush()
        obj, y = g.solve(tlimit=float(__import__('os').environ.get('TL', 280)))
        if g.done and obj > 1e-6:
            ok, b = g.verify(y); print("survivor %d: INFEASIBLE, slack %.4f, exact check %s %s" % (t, obj, "VALID" if ok else "FAILED", b))
            if ok:
                import gzip, os; os.makedirs("logs/orbit_certs", exist_ok=True)
                yi = [int(round(v * 10 ** 6)) for v in y]
                json.dump({"survivor": s, "triples": [list(z) for z in trip], "den": 10 ** 6,
                           "rows": [[list(map(lambda x: sorted(x) if isinstance(x, frozenset) else x, nm)) if False else repr(nm), g.rkind[r], g.rrhs[r], yi[r]] for r, nm in enumerate(g.rname) if yi[r]],
                           "bounds": b}, gzip.open("logs/orbit_certs/surv_%s_%d_%d.json.gz" % (s["tag"], s["job"], t), "wt"))
        else: print("survivor %d: %s obj %.5f" % (t, "FEASIBLE at orbit level" if g.done else "TIME CAP", obj))
        sys.stdout.flush()
