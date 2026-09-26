"""UNIFIED orbit-level column model (note 40): one-block AND disjoint squares, types fixed per pair, every D-partner variable.
Pair types: ('anch', c) both orbits one-block via c; ('tw',) orbit 0 via min, orbit 1 via max; ('dd',) both orbits dis;
('mix', c) orbit 0 one-block via c, orbit 1 dis.  (Sigma = empty here.)
Columns (all proved; notes 27-31, 39, 40):
  one-block Z (pivot i, free j): as orbit_colmodel_gen (partner = one-block orbit of {i,k} with pivot i, typed rule).
  dis Z of P = {i,j}: partner W = a dis orbit of a pair R = {k,l} disjoint from P, q[W] = 2, q[W'] = x in {0,1} ((2,0) or (2,1));
     loop (n_s = 1, only if the sister is dis, 39.3): q[sister] = 1, counting as one neighbour at i AND one at j;
     near: 2 - loop free neighbours at i (pairs {i,r}, r != j) and at j, never both in {i,k} (or {i,l}, {j,k}, {j,l}):
       two such roots would carry the same sign at k and, with w or sigma w, give (k,.) three neighbours (mu = 2);
     far: pairs disjoint from P, block sums 4 - n^i_t - n^j_t, v(R) = 2 + x, v <= 2 elsewhere; v = 2 -> (1,1), v = 1 -> one orbit.
Rows: those of orbit_colmodel (one, S, C, I, cuts on the free graphs; a loop is an edge of R_i and of R_j).
Usage: see __main__ (tests)."""
import sys, itertools, time, os, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, "scripts")
import orbit_colmodel as OM
from gen_colmodel_lam import far_solutions
N = 7

def far_sols_dis(blocks5, deg, R):
    """far rows with v(R) in {2,3} and v <= 2 elsewhere, block sums deg"""
    out = []
    for vR in (2, 3):
        d2 = dict(deg); a, b = tuple(R)
        d2[a] -= vR; d2[b] -= vR
        if min(d2.values()) < 0: continue
        prs, sols = far_solutions(blocks5, d2)
        iR = prs.index(frozenset(R))
        for s in sols:
            if s[iR] != 0: continue                    # the R-entry is carried by vR
            s2 = list(s); s2[iR] = vR; out.append(tuple(s2))
    prs = [frozenset(p) for p in itertools.combinations(blocks5, 2)]
    return prs, out

class OrbitModelU(OM.OrbitModel):
    def __init__(self, ptype, use_I=True, forbid=None):
        self.use_I = use_I; self.T = []; self.otype = []; self.forbid = forbid or set()     # forbid: (orbit, partner) pairs (block fixing)
        for a in range(N):
            for b in range(a + 1, N):
                t = ptype[(a, b)]
                for s in (0, 1):
                    if t[0] == 'anch': ty = ('one', t[1])
                    elif t[0] == 'tw': ty = ('one', (a, b)[s])
                    elif t[0] == 'dd': ty = ('dis',)
                    elif t[0] == 'mix': ty = ('one', t[1]) if s == 0 else ('dis',)
                    elif t[0] == 'sib': ty = ('sib',)
                    else: raise ValueError(t)
                    self.T.append((a, b, None)); self.otype.append(ty)
        n = self.n = 42
        self.pair = [frozenset(t[:2]) for t in self.T]
        self.orbs_of = defaultdict(list)
        for s in range(n): self.orbs_of[self.pair[s]].append(s)
        self.sis = [next(u for u in self.orbs_of[self.pair[s]] if u != s) for s in range(n)]
        self.free = {c: [s for s in range(n) if c in self.pair[s] and (self.otype[s][0] == 'dis' or (self.otype[s][0] == 'one' and self.otype[s][1] != c))] for c in range(N)}
        self.Sh = np.array([[2 if self.pair[a] == self.pair[b] else len(self.pair[a] & self.pair[b]) for b in range(n)] for a in range(n)])
        self.use_sis5 = False
        self.sibpairs = [frozenset(k) for k, v in ptype.items() if v[0] == 'sib']
        self.build_columns()
        self.empty_orbits = [Z for Z in range(n) if len(self.cols[Z]) == 0]
        if self.empty_orbits: return                     # combinatorially impossible: some orbit has no admissible column
        self.build_rows()

    def _expand(self, base, prs, sols, Dpair=None, Dz=None):
        mats = []
        for s in sols:
            v = base.copy(); opts = []
            for R, val in zip(prs, s):
                o1, o2 = self.orbs_of[R]
                if R == Dpair:
                    Wp = o2 if Dz == o1 else o1
                    v[Dz] += 2; v[Wp] += val - 2
                elif val == 2: v[o1] += 1; v[o2] += 1
                elif val == 1: opts.append((o1, o2))
            m = len(opts)
            if m == 0: mats.append(v[None, :]); continue
            ch = ((np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1).astype(bool)
            blk = np.repeat(v[None, :], 1 << m, axis=0)
            for t, (o1, o2) in enumerate(opts):
                blk[~ch[:, t], o1] += 1; blk[ch[:, t], o2] += 1
            mats.append(blk)
        return mats

    def build_columns(self):
        self.cols = []
        for Z in range(self.n):
            a, b, _ = self.T[Z]; ty = self.otype[Z]; mats = []
            other = lambda F, blk: next(iter(self.pair[F] - {blk}))
            if ty[0] == 'one':
                i = ty[1]; j = a + b - i
                cand = [F for F in self.free[j] if F != Z and i not in self.pair[F]]
                fb = [t for t in range(N) if t not in (i, j)]
                for k in fb:
                    for Dz in self.orbs_of[frozenset((i, k))]:
                        if self.otype[Dz] != ('one', i) or (Z, Dz) in self.forbid: continue
                        for F1, F2 in itertools.combinations(cand, 2):
                            r1, r2 = other(F1, j), other(F2, j)
                            if r1 == r2 == k: continue
                            nv = defaultdict(int); nv[r1] += 1; nv[r2] += 1
                            deg = {t: 4 - 2 * (t == k) - nv[t] for t in fb}
                            if min(deg.values()) < 0: continue
                            prs, sols = far_solutions(fb, deg)
                            base = np.zeros(42, dtype=np.int8); base[Dz] = 2; base[F1] += 1; base[F2] += 1
                            mats += self._expand(base, [frozenset(p) for p in prs], sols)
            elif ty[0] == 'sib':
                i, j = a, b; Zs = self.sis[Z]; fb = [t for t in range(N) if t not in (i, j)]
                prs, sols = far_solutions(fb, {t: 4 for t in fb})
                base = np.zeros(42, dtype=np.int8); base[Zs] = 2
                mats += self._expand(base, [frozenset(p) for p in prs], sols)
            else:
                i, j = a, b; Zs = self.sis[Z]; fb = [t for t in range(N) if t not in (i, j)]
                loops = (0, 1) if self.otype[Zs] == ('dis',) else (0,)
                for Dz in range(self.n):
                    if self.otype[Dz] != ('dis',) or (self.pair[Dz] & self.pair[Z]) or (Z, Dz) in self.forbid: continue
                    R = self.pair[Dz]; k, l = tuple(R)
                    ci = [F for F in self.free[i] if F not in (Z, Zs) and j not in self.pair[F]]
                    cj = [F for F in self.free[j] if F not in (Z, Zs) and i not in self.pair[F]]
                    for loop in loops:
                        ni = 2 - loop
                        for Fi in itertools.combinations(ci, ni):
                            ri = [other(F, i) for F in Fi]
                            if ni == 2 and (ri[0] == ri[1] and ri[0] in (k, l)): continue
                            for Fj in itertools.combinations(cj, ni):
                                rj = [other(F, j) for F in Fj]
                                if ni == 2 and (rj[0] == rj[1] and rj[0] in (k, l)): continue
                                nv = defaultdict(int)
                                for r in ri + rj: nv[r] += 1
                                deg = {t: 4 - nv[t] for t in fb}
                                if min(deg.values()) < 0: continue
                                prs, sols = far_sols_dis(fb, deg, R)
                                if not sols: continue
                                base = np.zeros(42, dtype=np.int8)
                                for F in Fi + Fj: base[F] += 1
                                if loop: base[Zs] += 1
                                mats += self._expand(base, prs, sols, Dpair=R, Dz=Dz)
            M = np.concatenate(mats) if mats else np.zeros((0, 42), dtype=np.int8)
            if len(M): assert (M.sum(axis=1) == 12).all(), (Z, ty)
            for R in self.sibpairs:                           # grid lemma 8.1: exactly one neighbour in every disjoint sib pair
                if not (R & self.pair[Z]) and len(M):
                    o1, o2 = self.orbs_of[R]; M = M[(M[:, o1] + M[:, o2]) == 1]
            self.cols.append(M)

def orient_types(In):
    """pair types for a one-block orientation dict (block -> anchored-at list); other pairs twisted"""
    pt = {}
    for a in range(N):
        for b in range(a + 1, N): pt[(a, b)] = ('tw',)
    for c, xs in In.items():
        for x in xs: pt[(min(c, x), max(c, x))] = ('anch', c)
    return pt

if __name__ == "__main__":
    import glob
    mode = sys.argv[1]
    if mode == "regress":                     # must reproduce orbit_colmodel_gen on a pure one-block orientation
        jobs = []
        for f in sorted(glob.glob("logs/mix6_cls_E01_12_23_34_45_05_S_*.json")):
            d = json.load(open(f))
            for po in d["per_orient"]:
                if po["leaves"]: jobs.append(po)
        In = {int(x): v for x, v in jobs[int(sys.argv[2])]["In"].items()}
        t0 = time.time(); g = OrbitModelU(orient_types(In))
        print("U-model (one-block only): columns %d rows %d %.0fs" % (sum(len(M) for M in g.cols), len(g.rname), time.time() - t0)); sys.stdout.flush()
        obj, y = g.solve(tlimit=float(os.environ.get('TL', 3000)), verbose=os.environ.get('V') is not None)
        if g.done and obj > 1e-6:
            ok, b = g.verify(y); print("INFEASIBLE slack %.4f exact %s %s" % (obj, "VALID" if ok else "FAILED", b))
        else: print("%s obj %.5f" % ("FEASIBLE" if g.done else "TIME CAP", obj))

def ad2_types(rep):
    pt = {(0, 1): ('dd',), (2, 3): ('dd',)}
    for k, h in rep.items():
        a, b = map(int, k.split(",")); pt[(a, b)] = ('anch', h)
    return pt

if __name__ == "__main__" and sys.argv[1] == "ad2":
    reps = json.load(open("logs/ad2_reps.json"))
    for t in [int(a) for a in sys.argv[2:]]:
        t0 = time.time(); g = OrbitModelU(ad2_types(reps[t]))
        print("AD2 rep %d: columns %d rows %d %.0fs" % (t, sum(len(M) for M in g.cols), len(g.rname), time.time() - t0)); sys.stdout.flush()
        obj, y = g.solve(tlimit=float(os.environ.get('TL', 5000)), verbose=os.environ.get('V') is not None)
        if g.done and obj > 1e-6:
            ok, b = g.verify(y)
            if ok: OM.save_cert(g, y, "logs/orbit_certs/ad2_%d.json.gz" % t, "U", {"ptype": {"%d,%d" % k: list(v) for k, v in ad2_types(reps[t]).items()}}, b)
            print("AD2 rep %d: INFEASIBLE slack %.4f exact %s %s %.0fs" % (t, obj, "VALID" if ok else "FAILED", b, time.time() - t0))
        else: print("AD2 rep %d: %s obj %.5f %.0fs" % (t, "FEASIBLE" if g.done else "TIME CAP", obj, time.time() - t0))
        sys.stdout.flush()

def closed_structures(ptype):
    """blocks c whose anchored-at-c pairs form a forced closed D-structure with exact counts (no twisted/mixed pair at c):
    in-degree 2 -> digon (16.2), in-degree 3 -> triangle (block-triangle lemma, note 41); returns [(c, arms)]"""
    out = []
    for c in range(N):
        arms = []; bad = False
        for x in range(N):
            if x == c: continue
            t = ptype[(min(c, x), max(c, x))]
            if t[0] == 'anch' and t[1] == c: arms.append(x)
            elif t[0] in ('tw', 'mix'): bad = True
        if not bad and len(arms) in (2, 3): out.append((c, tuple(arms)))
    return out

def apply_exact_filters(g, structs):
    """drop every column violating the exact counts N_Z = |B| - |B|[c in P_Z] - |P_Z & B| on the orbits of the pairs {c,b}, b in B"""
    kept = 0; before = sum(len(M) for M in g.cols)
    for (c, B) in structs:
        orbs = [o for b in B for o in g.orbs_of[frozenset((c, b))]]
        K = len(B)
        for Z in range(g.n):
            P = g.pair[Z]
            if c in P and (P - {c}) <= set(B): continue            # an orbit of the structure itself
            need = K - K * (c in P) - len(P & set(B))
            M = g.cols[Z]
            g.cols[Z] = M[M[:, orbs].sum(axis=1) == need]
    return before, sum(len(M) for M in g.cols)
