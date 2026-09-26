"""Orbit-level column model with the ORIENTATION fixed and every D-structure VARIABLE (note 40) -- orbit-level analogue of
gen_colmodel (pair level, too weak there).  Sigma = empty, one-block squares.
Orbits: (pair R, s), s in {0,1}.  Anchored at c: both orbits have pivot c.  Twisted {a,b} (a < b): orbit 0 has pivot a, orbit 1
pivot b (orbit labels are arbitrary, WLOG).  A column of Z (pivot i, free block j) chooses a third block k and a partner orbit D
among the orbits of {i,k} with pivot i, then two free neighbours (orbits free at j, i.e. with pivot != j, in pairs {j,r}, r != i;
not both in {j,k}), then a far row (block sums 4 - 2[t = k] - n_t).  Coupling rows q_Z(W) = q_W(Z) make the q = 2 entries a
perfect matching of the orbits consistent with the pivots, i.e. valid D-structures (integer level).  Rows / pricing / exact check
from orbit_colmodel.
Usage: python orbit_colmodel_gen.py <mix6 class json glob> <job index>     (orientation of that job)"""
import sys, json, glob, itertools, time, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, "scripts")
import orbit_colmodel as OM
from gen_colmodel_lam import far_solutions
N = 7

class OrbitModelGen(OM.OrbitModel):
    def __init__(self, In, use_I=True, forbid=None):
        """In: dict block -> list of x with {block,x} anchored at block; all other pairs twisted."""
        self.use_I = use_I
        anch = {}
        for c, xs in In.items():
            for x in xs: anch[frozenset((c, x))] = c
        self.T = []; self.pivot = []
        for a in range(N):
            for b in range(a + 1, N):
                R = frozenset((a, b))
                piv = [anch[R], anch[R]] if R in anch else [a, b]
                for s in (0, 1):
                    i = piv[s]; j = a + b - i; self.T.append((i, j, None)); self.pivot.append(i)
        n = self.n = 42
        self.pair = [frozenset(t[:2]) for t in self.T]
        self.orbs_of = defaultdict(list)
        for s in range(n): self.orbs_of[self.pair[s]].append(s)
        self.sis = [next(u for u in self.orbs_of[self.pair[s]] if u != s) for s in range(n)]
        self.free = {c: [s for s in range(n) if self.T[s][1] == c] for c in range(N)}
        self.Sh = np.array([[2 if self.pair[a] == self.pair[b] else len(self.pair[a] & self.pair[b]) for b in range(n)] for a in range(n)])
        self.forbid = forbid or set()
        self.use_sis5 = os.environ.get('SIS5') is not None
        self.build_columns(); self.build_rows()

    def build_columns(self):
        self.cols = []
        for Z in range(self.n):
            i, j, _ = self.T[Z]; mats = []
            for k in range(N):
                if k in (i, j): continue
                for Dz in self.orbs_of[frozenset((i, k))]:
                    if self.pivot[Dz] != i or (Z, Dz) in self.forbid: continue
                    mats += self._cols_for(Z, i, j, k, Dz)
            M = np.concatenate(mats); assert (M.sum(axis=1) == 12).all()
            self.cols.append(M)

    def _cols_for(self, Z, i, j, k, Dz):
        cand = [F for F in self.free[j] if F != Z and F != self.sis[Z]]
        far_blocks = [t for t in range(N) if t not in (i, j)]
        blocks_of = lambda F: next(iter(self.pair[F] - {j}))
        mats = []
        for F1, F2 in itertools.combinations(cand, 2):
            r1, r2 = blocks_of(F1), blocks_of(F2)
            if r1 == r2 == k: continue
            nv = defaultdict(int); nv[r1] += 1; nv[r2] += 1
            deg = {t: 4 - 2 * (t == k) - nv[t] for t in far_blocks}
            if min(deg.values()) < 0: continue
            prs, sols = far_solutions(far_blocks, deg)
            base = np.zeros(42, dtype=np.int8); base[Dz] = 2; base[F1] += 1; base[F2] += 1
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
                for t, (o1, o2) in enumerate(opts):
                    blk[~ch[:, t], o1] += 1; blk[ch[:, t], o2] += 1
                mats.append(blk)
        return mats

if __name__ == "__main__":
    pat, job = sys.argv[1], int(sys.argv[2])
    jobs = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        for po in d["per_orient"]:
            if po["leaves"]: jobs.append(po)
    po = jobs[job]; In = {int(x): v for x, v in po["In"].items()}
    t0 = time.time(); g = OrbitModelGen(In)
    print("orientation job %d In %s: columns %d rows %d built %.0fs" % (job, po["In"], sum(len(M) for M in g.cols), len(g.rname), time.time() - t0)); sys.stdout.flush()
    obj, y = g.solve(tlimit=float(os.environ.get('TL', 280)), verbose=os.environ.get('V') is not None)
    if g.done and obj > 1e-6:
        ok, b = g.verify(y)
        if ok: OM.save_cert(g, y, "logs/orbit_certs/gen_%s_%d.json.gz" % (os.environ.get('TAG', 'x'), job), "gen", {"In": po["In"]}, b)
        print("orientation job %d: INFEASIBLE (all D variable), slack %.4f, exact %s %s %.0fs" % (job, obj, "VALID" if ok else "FAILED", b, time.time() - t0))
    else: print("orientation job %d: %s obj %.5f %.0fs" % (job, "FEASIBLE" if g.done else "TIME CAP", obj, time.time() - t0))
