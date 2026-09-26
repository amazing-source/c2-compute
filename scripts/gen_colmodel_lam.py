"""[LAMBDA VARIANT (note 35): the D-matchings are modelled by per-block variables lambda_{c,g} over all valid D-graphs g at c
(exact convex hull of the block structures), linked at PAIR level: sum over the two orbits of R = {c,j} of the weight of columns
with pivot c and third block k  =  sum_g lambda_{c,g} * mult_g(j,k).]
General column model for Sigma = empty configurations with one-block squares (note 33) -- for a fixed ORIENTATION (each pair is
twisted or anchored at one of its blocks), everything else is variable: pivots of twisted orbits, the D-matchings at every pivot,
the free graphs at every block, and the far rows.  Solved by column generation; an infeasibility certificate y is checked EXACTLY
(integers, after scaling the rounded rational multipliers) over ALL columns, which are enumerated in the check.

Columns: (orbit O, pivot i, third block k, n-vector, far solution s).  Values (all proved; notes 27-31):
  val(P_O)=0, val({i,r})=2[r=k] (F1), val({j,r})=n_r (free neighbours at j), far values s with block sums 4 (L), all in {0,1,2}.
  n_r = 0 if {j,r} is anchored at j; n_r <= 1 if {j,r} twisted; n_k <= 1 (typed rule).
Rows: one column per orbit; twisted pairs: sisters on different pivots; D-link rows with matching variables m (orbits sharing exactly
the pivot block); free-link rows with edge variables e (orbits sharing exactly the free block); triangle and 5-cycle cuts on e;
sisters <= 1 common neighbour; S, PPE, coupling C, and (P2) -- linear in columns.
Optional extra constraints: forbid linked 2-cycles at given (c, {a,b}) (lemmas 28.5 / 30.1 when they apply)."""
import sys, itertools, time
from fractions import Fraction
from collections import defaultdict
import numpy as np
import highspy
from scipy.sparse import csc_matrix
N = 7
P = lambda a, b: frozenset((a, b))
PAIRS = [P(a, b) for a in range(N) for b in range(a + 1, N)]
PI = {R: t for t, R in enumerate(PAIRS)}

def far_solutions(blocks5, deg):
    prs = [P(a, b) for a, b in itertools.combinations(blocks5, 2)]
    out = []
    def rec(t, cur, rem):
        if t == len(prs):
            if all(v == 0 for v in rem.values()): out.append(tuple(cur))
            return
        a, b = tuple(prs[t])
        for v in (0, 1, 2):
            if rem[a] < v or rem[b] < v: break
            rem[a] -= v; rem[b] -= v; cur.append(v); rec(t + 1, cur, rem); cur.pop(); rem[a] += v; rem[b] += v
    rec(0, [], dict(deg))
    return prs, out

class GenModel:
    def __init__(self, orient, useP2=True, forbid_linked=(), cut5=True, allowed_ik=None):
        """orient: dict pair -> 'tw' or anchor block (int)."""
        self.orient = orient
        self.orbits = [(R, s) for R in PAIRS for s in (0, 1)]
        self.pivots = {}
        for (R, s) in self.orbits:
            o = orient[R]
            self.pivots[R, s] = sorted(R) if o in ('tw', 'free', 'anch') else [o]
        # column "types": (O, i, k, nvec) ; far solutions cached by (i, j, k, nvec)
        self.types = []
        for O in self.orbits:
            R = O[0]
            for i in self.pivots[O]:
                j = next(iter(R - {i}))
                others = [r for r in range(N) if r not in R]
                caps = {}
                for r in others:
                    o = orient[P(j, r)]
                    caps[r] = 2 if o in ('free', 'anch') else (1 if o == 'tw' else (0 if o == j else 2))
                for k in others:
                    if allowed_ik is not None and (i, k) not in allowed_ik.get(O, ()): continue
                    for a, b in itertools.combinations_with_replacement(others, 2):
                        nv = defaultdict(int); nv[a] += 1; nv[b] += 1
                        if any(nv[r] > caps[r] for r in nv) or nv[k] > 1: continue
                        self.types.append((O, i, j, k, tuple(sorted(nv.items()))))
        self.farcache = {}
        # variables m (pivot sharing) and e (free sharing): unordered orbit pairs sharing exactly one block
        self.share = []
        for O1, O2 in itertools.combinations(self.orbits, 2):
            I = O1[0] & O2[0]
            if len(I) == 1: self.share.append((O1, O2, next(iter(I))))
        self.useP2 = useP2; self.forbid_linked = forbid_linked; self.cut5 = cut5; self._keys = set()
        import mix_enum as ME
        self.dgs = {}
        for c in range(N):
            anch = [x for x in range(N) if x != c and orient[P(c, x)] == c]
            tws = [x for x in range(N) if x != c and orient[P(c, x)] == 'tw']
            self.dgs[c] = ME.dgraphs(anch, tws) if (anch or tws) else []
        self.build_rows()

    def far(self, i, j, k, nv):
        key = (i, j, k, nv)
        if key not in self.farcache:
            blocks = [t for t in range(N) if t not in (i, j)]
            nd = dict(nv)
            deg = {t: 4 - (2 if t == k else 0) - nd.get(t, 0) for t in blocks}
            prs, sols = far_solutions(blocks, deg)
            self.farcache[key] = (prs, np.array(sols, dtype=np.int64).reshape(-1, len(prs)))
        return self.farcache[key]

    def near_vector(self, O, i, j, k, nv):
        v = np.zeros(len(PAIRS), dtype=np.int64)
        nd = dict(nv)
        for r in range(N):
            if r in (i, j): continue
            v[PI[P(i, r)]] = 2 if r == k else 0
            v[PI[P(j, r)]] = nd.get(r, 0)
        return v

    def build_rows(self):
        rows = {}          # name -> index ; kinds
        self.rname = []; self.rkind = []; self.rrhs = []
        def add(name, kind, rhs):
            rows[name] = len(self.rname); self.rname.append(name); self.rkind.append(kind); self.rrhs.append(rhs)
        for O in self.orbits: add(('one', O), 'eq', 1)
        for R in PAIRS:
            if self.orient[R] == 'tw': add(('twist', R), 'eq', 1)
            if self.orient[R] == 'anch': add(('same', R), 'eq', 0)
        for R in PAIRS:
            for i in sorted(R):
                if i not in self.pivots[(R, 0)]: continue
                for k in range(N):
                    if k in R: continue
                    add(('DP', R, i, k), 'eq', 0)
        for c in range(N):
            if self.dgs[c]: add(('lam', c), 'eq', 1)
        for O in self.orbits:
            for j in O[0]:
                for r in range(N):
                    if r in O[0]: continue
                    add(('F', O, j, r), 'eq', 0)
        for j in range(N):
            orbs = [O for O in self.orbits if j in O[0]]
            for S in itertools.combinations(orbs, 3):
                if len({O[0] for O in S}) < 3: continue
                add(('cut3', j, S), 'le', 2)
            if self.cut5:
                for S in itertools.combinations(orbs, 5):
                    add(('cut5', j, S), 'le', 4)
            for r in range(N):
                if r != j and self.orient[P(j, r)] in (r, 'free', 'anch'): add(('sis', j, r), 'le', 1)
        for R in PAIRS:
            add(('S', R), 'eq', 24); add(('PPE', R), 'eq', 32)
        for a in range(len(PAIRS)):
            for b in range(a + 1, len(PAIRS)):
                Ra, Rb = PAIRS[a], PAIRS[b]
                add(('C', a, b), 'eq', 0)
                if self.useP2: add(('P2', a, b), 'eq', 16 - (4 if Ra & Rb else 0))
        for (c, pr) in self.forbid_linked:
            add(('nolink', c, pr), 'le', 1)
        self.rows = rows
        # fixed (non-column) variables m, e with their row entries
        self.fixed_cols = []      # list of (name, [(row, coef)])
        for c in range(N):
            for gi, g in enumerate(self.dgs[c]):
                ent = [(rows['lam', c], 1)]
                mult = defaultdict(int)
                for (x, y) in g: mult[(x, y)] += 1; mult[(y, x)] += 1
                for (x, y), mu in mult.items():
                    ent.append((rows['DP', P(c, x), c, y], -mu))
                self.fixed_cols.append((('lam', c, gi), ent))
        for (O1, O2, t) in self.share:
            k1 = next(iter(O2[0] - {t})); k2 = next(iter(O1[0] - {t}))
            # e: free block t for both
            ent = [(rows['F', O1, t, k1], -1), (rows['F', O2, t, k2], -1)]
            for name, idx in rows.items():
                if name[0] in ('cut3', 'cut5') and name[1] == t and O1 in name[2] and O2 in name[2]: ent.append((idx, 1))
            self.fixed_cols.append((('e', O1, O2), ent))

    def type_rows(self, O, i, j, k, nv):
        """(row, coef) entries common to all columns of this type, plus the orbit's pair index"""
        rows = self.rows; ent = [(rows['one', O], 1)]
        R = O[0]
        if self.orient[R] == 'tw' and i == min(R): ent.append((rows['twist', R], 1))
        if self.orient[R] == 'anch' and i == min(R): ent.append((rows['same', R], 1 if O[1] == 0 else -1))
        ent.append((rows['DP', R, i, k], 1))
        nd = dict(nv)
        for r, v in nd.items():
            ent.append((rows['F', O, j, r], v))
            if v == 2 and ('sis', j, r) in rows: ent.append((rows['sis', j, r], 1))
        return ent

    def prep(self, y):
        Q = np.diag([y[self.rows['PPE', R]] for R in PAIRS]).astype(float)
        linS = np.array([y[self.rows['S', R]] for R in PAIRS], dtype=float)
        linpa = [linS.copy() for _ in PAIRS]
        for a in range(len(PAIRS)):
            for b in range(a + 1, len(PAIRS)):
                yc = y[self.rows['C', a, b]]
                linpa[a][b] += yc; linpa[b][a] -= yc
                if self.useP2:
                    yp = y[self.rows['P2', a, b]]
                    Q[a, b] += yp / 2; Q[b, a] += yp / 2; linpa[a][b] += yp
        self._Q, self._linpa = Q, linpa

    def vals_of(self, O, i, j, k, nv):
        key = (O, i, k, nv)
        if not hasattr(self, '_vc'): self._vc = {}
        if key not in self._vc:
            prs, sols = self.far(i, j, k, nv)
            base = self.near_vector(O, i, j, k, nv)
            vals = np.repeat(base[None, :], len(sols), axis=0).astype(np.int8)
            for t, R in enumerate(prs): vals[:, PI[R]] = sols[:, t]
            self._vc[key] = vals
        return self._vc[key]

    def column_matrix(self, O, i, j, k, nv, y):
        vals = self.vals_of(O, i, j, k, nv)
        if len(vals) == 0: return np.zeros(0), vals
        v = vals.astype(float)
        const = sum(y[r] * c for r, c in self.type_rows(O, i, j, k, nv))
        lin = self._linpa[PI[O[0]]]
        scores = const + v @ lin + np.einsum('ij,jk,ik->i', v, self._Q, v)
        return scores, vals

    def column_matrix_old(self, O, i, j, k, nv, y):
        """score y^T A_c for all far solutions of this type (vectorised); returns (scores, vals)"""
        prs, sols = self.far(i, j, k, nv)
        base = self.near_vector(O, i, j, k, nv)
        vals = np.repeat(base[None, :], len(sols), axis=0)
        for t, R in enumerate(prs): vals[:, PI[R]] = sols[:, t]
        const = sum(y[r] * c for r, c in self.type_rows(O, i, j, k, nv))
        lin = np.array([y[self.rows['S', R]] for R in PAIRS], dtype=float)
        Q = np.diag([y[self.rows['PPE', R]] for R in PAIRS]).astype(float)
        pa = PI[O[0]]
        for a in range(len(PAIRS)):
            for b in range(a + 1, len(PAIRS)):
                yc = y[self.rows['C', a, b]]
                if pa == a: lin[b] += yc
                if pa == b: lin[a] -= yc
                if self.useP2:
                    yp = y[self.rows['P2', a, b]]
                    Q[a, b] += yp / 2; Q[b, a] += yp / 2
                    if pa == a: lin[b] += yp
        scores = const + vals @ lin + np.einsum('ij,jk,ik->i', vals, Q, vals)
        return scores, vals

    def entries_of(self, O, i, j, k, nv, val):
        ent = list(self.type_rows(O, i, j, k, nv))
        rows = self.rows; pa = PI[O[0]]
        for t, R in enumerate(PAIRS):
            if val[t]:
                ent.append((rows['S', R], int(val[t]))); ent.append((rows['PPE', R], int(val[t]) ** 2))
        for a in range(len(PAIRS)):
            for b in range(a + 1, len(PAIRS)):
                cc = (int(val[b]) if pa == a else 0) - (int(val[a]) if pa == b else 0)
                if cc: ent.append((rows['C', a, b], cc))
                if self.useP2:
                    pp = int(val[a]) * int(val[b]) + (int(val[b]) if pa == a else 0)
                    if pp: ent.append((rows['P2', a, b], pp))
        return ent

    def solve_cg(self, max_iter=200, per_type=2, tlimit=280, verbose=True):
        t0 = time.time(); m = len(self.rname)
        if not hasattr(self, 'pool'): self.pool = []
        cols = self.pool     # persistent list of entries lists
        colkey = []
        # start: slack columns only (+ fixed m/e columns)
        it = 0
        while True:
            it += 1
            h, ncols_total = self._master(cols)
            obj = h.getInfo().objective_function_value
            y = np.array(h.getSolution().row_dual)
            self.prep(y)
            added = 0; best = 0
            for (O, i, j, k, nv) in self.types:
                sc, vals = self.column_matrix(O, i, j, k, nv, y)
                if len(sc) == 0: continue
                order = np.argsort(-sc)[:per_type]
                for t in order:
                    if sc[t] > 1e-7:
                        key = (O, i, k, nv, tuple(vals[t]))
                        if key in self._keys: continue
                        self._keys.add(key); cols.append(self.entries_of(O, i, j, k, nv, vals[t])); added += 1
                        best = max(best, sc[t])
            if verbose: print("  iter %d obj %.4f cols %d added %d best %.3g %.0fs" % (it, obj, len(cols), added, best, time.time() - t0)); sys.stdout.flush()
            if added == 0 or it >= max_iter or time.time() - t0 > tlimit:
                self.last_y = y; self.last_obj = obj; self.done = (added == 0)
                self.last_x = np.array(h.getSolution().col_value)[:len(self.fixed_cols)]
                return obj, y

    def set_branch_rows(self, brs):
        """brs: list of (list of fixed-col indices, lo, hi): sum of those m's must lie in [lo, hi].  Implemented as rows
        sum m - s = 0 with a slack column s in [lo, hi] (so certificates stay box-checkable)."""
        self.nbase_rows = getattr(self, 'nbase_rows', len(self.rname))
        self.nbase_fixed = getattr(self, 'nbase_fixed', len(self.fixed_cols))
        del self.rname[self.nbase_rows:]; del self.rkind[self.nbase_rows:]; del self.rrhs[self.nbase_rows:]
        del self.fixed_cols[self.nbase_fixed:]
        self.bounds = {t: b for t, b in getattr(self, 'bounds', {}).items() if t < self.nbase_fixed}
        for q, (idxs, lo, hi) in enumerate(brs):
            r = len(self.rname); self.rname.append(('br', q)); self.rkind.append('eq'); self.rrhs.append(0)
            for t in idxs:
                nm, ent = self.fixed_cols[t]; self.fixed_cols[t] = (nm, [e for e in ent if e[0] < self.nbase_rows or e[0] >= r] + [(r, 1)])
            t_s = len(self.fixed_cols); self.fixed_cols.append((('brs', q), [(r, -1)])); self.bounds[t_s] = (lo, hi)
        # drop stale branch-row entries of previous calls from m columns
        for t in range(self.nbase_fixed):
            nm, ent = self.fixed_cols[t]
            self.fixed_cols[t] = (nm, [e for e in ent if e[0] < len(self.rname)])

    def _master(self, cols):
        m = len(self.rname)
        allcols = [ent for (_, ent) in self.fixed_cols] + cols
        nx = len(allcols)
        le = [r for r in range(m) if self.rkind[r] == 'le']; nl = len(le)
        data = []; ri = []; ci = []
        for c, ent in enumerate(allcols):
            for r, v in ent: ri.append(r); ci.append(c); data.append(float(v))
        for t, r in enumerate(le): ri.append(r); ci.append(nx + t); data.append(1.0)
        eq = [r for r in range(m) if self.rkind[r] == 'eq']
        for t, r in enumerate(eq):
            ri.append(r); ci.append(nx + nl + 2 * t); data.append(1.0)
            ri.append(r); ci.append(nx + nl + 2 * t + 1); data.append(-1.0)
        nc = nx + nl + 2 * len(eq)
        A = csc_matrix((data, (ri, ci)), shape=(m, nc))
        lp = highspy.HighsLp(); lp.num_col_ = nc; lp.num_row_ = m
        lp.col_cost_ = np.concatenate([np.zeros(nx + nl), np.ones(2 * len(eq))])
        lo = np.zeros(nx); up = np.ones(nx)
        for t, (l_, u_) in getattr(self, 'bounds', {}).items(): lo[t] = l_; up[t] = u_
        lp.col_upper_ = np.concatenate([up, np.array([self.rrhs[r] for r in le], dtype=float), np.full(2 * len(eq), highspy.kHighsInf)])
        lp.col_lower_ = np.concatenate([lo, np.zeros(nc - nx)])
        b = np.array(self.rrhs, dtype=float); lp.row_lower_ = b; lp.row_upper_ = b
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("solver", "ipm"); h.setOptionValue("run_crossover", "off"); h.passModel(lp); h.run()
        return h, nc

    def verify(self, y, den=10 ** 6):
        """Exact check over ALL columns: integer arithmetic with y scaled by den (y rounded to multiples of 1/den)."""
        yi = np.array([int(round(v * den)) for v in y], dtype=np.int64)
        yb = sum(int(yi[r]) * int(self.rrhs[r]) for r in range(len(yi)))
        UB = 0; LB = 0
        bnd = getattr(self, 'bounds', {})
        for t, (name, ent) in enumerate(self.fixed_cols):
            s = sum(int(yi[r]) * int(v) for r, v in ent); l_, u_ = bnd.get(t, (0, 1))
            UB += max(s * l_, s * u_); LB += min(s * l_, s * u_)
        for r in range(len(yi)):
            if self.rkind[r] == 'le':
                s = int(yi[r]) * int(self.rrhs[r]); UB += max(s, 0); LB += min(s, 0)
        # all columns, per type: exact integer scores
        for (O, i, j, k, nv) in self.types:
            prs, sols = self.far(i, j, k, nv)
            if len(sols) == 0: continue
            base = self.near_vector(O, i, j, k, nv)
            vals = np.repeat(base[None, :], len(sols), axis=0)
            for t, R in enumerate(prs): vals[:, PI[R]] = sols[:, t]
            const = sum(int(yi[r]) * int(c) for r, c in self.type_rows(O, i, j, k, nv))
            lin = np.array([yi[self.rows['S', R]] for R in PAIRS], dtype=np.int64)
            Q2 = 2 * np.diag([yi[self.rows['PPE', R]] for R in PAIRS]).astype(np.int64)   # 2*Q to stay integral
            pa = PI[O[0]]
            for a in range(len(PAIRS)):
                for b in range(a + 1, len(PAIRS)):
                    yc = yi[self.rows['C', a, b]]
                    if pa == a: lin[b] += yc
                    if pa == b: lin[a] -= yc
                    if self.useP2:
                        yp = yi[self.rows['P2', a, b]]
                        Q2[a, b] += yp; Q2[b, a] += yp
                        if pa == a: lin[b] += yp
            sc2 = 2 * const + 2 * (vals @ lin) + np.einsum('ij,jk,ik->i', vals, Q2, vals)   # = 2 * score, exact int64
            mx = int(sc2.max()); mn = int(sc2.min())
            # per orbit the 'one' row forces a convex combination; bounding over the box: sum of positive parts (valid, cruder)
            UB += sum(int(v) for v in sc2[sc2 > 0]) // 2 + (1 if (sc2[sc2 > 0].sum() % 2) else 0)
            LB += -((-sum(int(v) for v in sc2[sc2 < 0])) // 2) - 1
        return (yb > UB or yb < LB), (yb / den, LB / den, UB / den)

def branch_and_bound(g, tlimit, node_limit=500, verbose=True):
    """DFS on the m (D-matching) variables; every pruned node is certified exactly (verify with node bounds).
    Returns ('ALL_PRUNED', stats) if the whole tree is closed, else ('OPEN', stats)."""
    t0 = time.time(); stack = [dict()]; stats = {'nodes': 0, 'certified': 0, 'cert_fail': 0, 'int_feasible_like': 0}
    mids = [t for t, (nm, ent) in enumerate(g.fixed_cols) if nm[0] == 'm']
    while stack:
        if time.time() - t0 > tlimit or stats['nodes'] >= node_limit: return 'OPEN', stats
        b = stack.pop(); g.bounds = b; stats['nodes'] += 1
        obj, y = g.solve_cg(tlimit=max(10, tlimit - (time.time() - t0)), verbose=False)
        if not g.done: return 'OPEN', stats
        if obj > 1e-6:
            ok, info = g.verify(y)
            if ok: stats['certified'] += 1; continue
            stats['cert_fail'] += 1; return 'OPEN', stats
        x = g.last_x
        frac = [(abs(x[t] - 0.5), t) for t in mids if 1e-6 < x[t] < 1 - 1e-6 and t not in b]
        if not frac:
            stats['int_feasible_like'] += 1; return 'OPEN', stats     # m integral: would need branching on e / columns
        _, t = min(frac)
        if verbose: print("   node %d obj 0 -> branch on %s (x=%.3f) depth %d %.0fs" % (stats['nodes'], g.fixed_cols[t][0][1:], x[t], len(b), time.time() - t0)); sys.stdout.flush()
        b0 = dict(b); b0[t] = (0, 0); b1 = dict(b); b1[t] = (1, 1)
        stack.append(b0); stack.append(b1)
    return 'ALL_PRUNED', stats

def pair_level_groups(g):
    """aggregated D-adjacency between pairs {c,x} and {c,y} at pivot c: list of ((c, x, y), [fixed-col indices of m's])"""
    groups = defaultdict(list)
    for t, (nm, ent) in enumerate(g.fixed_cols):
        if nm[0] != 'm': continue
        O1, O2 = nm[1], nm[2]; c = next(iter(O1[0] & O2[0]))
        x = next(iter(O1[0] - {c})); y = next(iter(O2[0] - {c}))
        groups[(c, min(x, y), max(x, y))].append(t)
    return list(groups.items())

def bb_pairs(g, tlimit, node_limit=400, verbose=True):
    """DFS branching on aggregated pair-level D-adjacency M_{c;x,y} in {0} vs {>=1}; pruned nodes certified exactly."""
    t0 = time.time(); groups = pair_level_groups(g); stack = [[]]
    stats = {'nodes': 0, 'certified': 0, 'cert_fail': 0}
    while stack:
        if time.time() - t0 > tlimit or stats['nodes'] >= node_limit: return 'OPEN', stats
        brs = stack.pop(); g.set_branch_rows(brs); stats['nodes'] += 1
        obj, y = g.solve_cg(tlimit=max(10, tlimit - (time.time() - t0)), verbose=False)
        if not g.done: return 'OPEN', stats
        if obj > 1e-6:
            ok, info = g.verify(y)
            if ok: stats['certified'] += 1; continue
            stats['cert_fail'] += 1; return 'OPEN', stats
        x = g.last_x
        used = {tuple(b[0]) for b in brs}
        cand = []
        for key, idxs in groups:
            if tuple(idxs) in used: continue
            v = float(sum(x[t] for t in idxs))
            if 1e-6 < v < 1 - 1e-6 or (1 + 1e-6 < v < 2 - 1e-6): cand.append((abs(v - round(v) - 0.5) if v < 1 else abs(v - 1.5), key, idxs, v))
        if not cand: return 'OPEN(integral pair-level)', stats
        _, key, idxs, v = min(cand)
        if verbose: print("   node %d: branch on M%s = %.3f (depth %d) %.0fs" % (stats['nodes'], key, v, len(brs), time.time() - t0)); sys.stdout.flush()
        if v < 1: stack.append(brs + [(idxs, 0, 0)]); stack.append(brs + [(idxs, 1, 2)])
        else: stack.append(brs + [(idxs, 0, 1)]); stack.append(brs + [(idxs, 2, 2)])
    return 'ALL_PRUNED', stats

def orientation_H(In):
    """H family: T = Z7 twisted, chords anchored at their head: In[c] = blocks x with {c,x} anchored at c."""
    o = {}
    for R in PAIRS:
        a, b = sorted(R)
        if (b - a) % N in (1, N - 1): o[R] = 'tw'
    for c in range(N):
        for x in In[c]: o[P(c, x)] = c
    return o

if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "twset":        # twisted set given as list of pairs 'ab,cd,..'; all other pairs anchored at an unknown end
        tw = {P(int(x[0]), int(x[1])) for x in sys.argv[3].split(",")}
        orient = {R: ('tw' if R in tw else 'anch') for R in PAIRS}
        sys.argv = sys.argv[:3] + sys.argv[4:]
    elif kind == "alltw":
        orient = {R: 'tw' for R in PAIRS}
    elif kind == "universal":
        orient = {R: 'free' for R in PAIRS}
    elif kind == "Hcirc":
        In = {c: [(c - 2) % N, (c - 3) % N] for c in range(N)}
        orient = orientation_H(In)
    t0 = time.time()
    g = GenModel(orient, useP2="noP2" not in sys.argv, cut5="nocut5" not in sys.argv)
    print("rows", len(g.rname), "column types", len(g.types), "fixed cols", len(g.fixed_cols), "%.0fs" % (time.time() - t0)); sys.stdout.flush()
    obj, y = g.solve_cg(tlimit=float(sys.argv[2]) if len(sys.argv) > 2 else 250)
    import json; json.dump({"argv": sys.argv, "obj": obj, "done": g.done, "y": [float(v) for v in y]}, open("logs/gen_last.json", "w"))
    print("final obj %.4f" % obj, "converged" if g.done else "NOT converged")
    if g.done and obj > 1e-6:
        ok, info = g.verify(y); print("exact certificate:", ok, info)
