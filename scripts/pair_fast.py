"""FAST per-leaf solver for the pair-level model of 'structures' (external_structures/ad/general_model.build, reviewed sound in
note 41) on mixture leaves (Sigma = empty, no dis square, every orbit has a pivot and a partner; A-block: its orbits keep an
unknown partner only if the leaf does not fix it -- see pair_fast_A below).  Note 43.
Same variables (phi, then y, in general_model's order), same rows, same bounds; ONE HiGHS model per job, and from leaf to leaf only
  - the pivot bounds  phi_W(t y) = 2 (partner's pair) / 0 (other pairs through the pivot t),
  - the balance rows  (a superset over the job's leaves, switched on/off by their bounds),
  - the dichotomy coefficients  (the two D-partners of the orbits of P are left out of P's row)
change (warm start).  Phase 1 = min total violation, as in ad/certify.py; the row duals give the Farkas certificate, rationalised
(denominator 10^6) and CHECKED here in exact integer arithmetic; branch trees as in ad/certify.certify.  Certificates are exported
in general_model's row indexing (sparse), so ad/certify.check after general_model.build re-verifies them (scripts/pair_verify.py).
Usage: python pair_fast.py <class id> <job> [lo hi]      -> logs/pair_certs/leaf_<ci>_<job>[_lo_hi].jsonl.gz"""
import sys, os, json, time, gzip, itertools
from collections import Counter
import numpy as np
import highspy
sys.path.insert(0, os.environ.get('STRUCT_AD', (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad') if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad')) else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor_structures'))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general_model import closed_classes, PAIRS, pr
from pair_blockdfs import orbit_list, load_job
N = 7; PI = {P: i for i, P in enumerate(PAIRS)}; INF = highspy.kHighsInf; DEN = 10 ** 6


class FastPair:
    def __init__(self, In, leaves_assign):
        """In: orientation; leaves_assign: list of {block: triples} (every block except A fixed) -- used only to collect the
        balance classes that can occur."""
        self.In = In
        orbs = orbit_list(In, {})
        self.orbs0 = orbs; n = self.n = len(orbs)
        self.pair = [o['pair'] for o in orbs]; self.piv = [o['pivot'] for o in orbs]
        self.by_pair = {P: [i for i in range(n) if self.pair[i] == P] for P in PAIRS}
        # variables: phi (i, P) then y (i, P != pair_i) x 4, exactly as general_model.build (no ns, no free-partner x)
        self.phi = {(i, P): i * 21 + PI[P] for i in range(n) for P in PAIRS}
        self.y = {}; c = 21 * n
        for i in range(n):
            for P in PAIRS:
                if P != self.pair[i]: self.y[(i, P)] = c; c += 4
        self.nvar = c
        lb = np.zeros(c); ub = np.full(c, 3.0)
        rows = []                      # (name, coefs, lo, hi) ; names: ('F1', i, t), ('bal', i, k, S), ('sym', P, P2), ('oh', i, P), ('ln', i, P), ('dich', P)
        for i in range(n):
            P = self.pair[i]; ub[self.phi[(i, P)]] = 0
            for t in P:
                rows.append((('F1', i, t), [(self.phi[(i, pr(t, m))], 1) for m in range(N) if m != t], 2, 2))
                if self.piv[i] != t:
                    for m in range(N):
                        if m == t or pr(t, m) == P: continue
                        cap = sum(1 for j2 in self.by_pair[pr(t, m)] if self.piv[j2] != t)
                        ub[self.phi[(i, pr(t, m))]] = min(ub[self.phi[(i, pr(t, m))]], cap)
        # balance superset
        self.classes = {k: set() for k in range(N)}
        for asg in leaves_assign:
            o = orbit_list(In, asg)
            for k in range(N):
                cl, rest = closed_classes(o, k)
                for S in cl + ([rest] if rest else []): self.classes[k].add(frozenset(S))
        for i in range(n):
            P = self.pair[i]
            for k in range(N):
                if k in P: continue
                for S in sorted(self.classes[k], key=sorted):
                    rows.append((('bal', i, k, S), [(self.phi[(i, pr(k, m))], 1) for m in S], -INF, INF))
        for P, P2 in itertools.combinations(PAIRS, 2):
            rows.append((('sym', P, P2), [(self.phi[(i, P2)], 1) for i in self.by_pair[P]] + [(self.phi[(i, P)], -1) for i in self.by_pair[P2]], 0, 0))
        for (i, P), yy in self.y.items():
            for f in range(4): ub[yy + f] = 1
            ub[yy + 3] = 0                                   # no dis square in a mixture
            rows.append((('oh', i, P), [(yy + f, 1) for f in range(4)], 1, 1))
            rows.append((('ln', i, P), [(self.phi[(i, P)], 1)] + [(yy + f, -f) for f in range(4)], 0, 0))
        for P in PAIRS:
            co = []
            for i in range(n):
                if self.pair[i] == P: continue
                co += [(self.y[(i, P)] + 2, 1), (self.y[(i, P)] + 3, 2)]
            rows.append((('dich', P), co, 2, 2))
        self.rows = rows; self.rindex = {r[0]: t for t, r in enumerate(rows)}
        self.lb0, self.ub0 = lb, ub
        # HiGHS phase-1 model: x (nvar) + s+ (m) + s- (m), min sum s
        m = self.m = len(rows); h = self.h = highspy.Highs(); h.setOptionValue('output_flag', False)
        h.setOptionValue('threads', 1); h.setOptionValue('presolve', 'off')
        inf = INF
        h.addVars(c, lb, ub)
        h.addVars(2 * m, np.zeros(2 * m), np.full(2 * m, inf))
        h.changeColsCost(2 * m, np.arange(c, c + 2 * m, dtype=np.int32), np.ones(2 * m))
        for t, (name, co, lo, hi) in enumerate(rows):
            idx = [j for j, _ in co] + [c + t, c + m + t]; val = [v for _, v in co] + [1.0, -1.0]
            h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=float))
        self.cur_lb, self.cur_ub = lb.copy(), ub.copy()
        self.cur_rlo = np.array([r[2] for r in rows], dtype=float); self.cur_rhi = np.array([r[3] for r in rows], dtype=float)
        self.cur_dich_zero = set()     # (P, i): coefficient of y_{i,P,2} currently 0 in dich(P)

    def set_leaf(self, assign):
        """assign: {block: triples} (all blocks but A).  Returns the orbit list (general_model skeleton) of the leaf."""
        orbs = orbit_list(self.In, assign); self.orbs = orbs
        lb, ub = self.lb0.copy(), self.ub0.copy()
        for i, o in enumerate(orbs):
            t = o['pivot']
            if o['partner'] is None: continue               # (A-block orbits: unknown partner -- not supported here)
            q = orbs[o['partner']]['pair']
            for m in range(N):
                if m == t: continue
                j = self.phi[(i, pr(t, m))]
                if pr(t, m) == q: lb[j] = ub[j] = 2
                else: ub[j] = 0
        ch = np.nonzero((lb != self.cur_lb) | (ub != self.cur_ub))[0]
        if len(ch): self.h.changeColsBounds(len(ch), ch.astype(np.int32), lb[ch], ub[ch])
        self.cur_lb, self.cur_ub = lb, ub
        rlo, rhi = self.cur_rlo.copy(), self.cur_rhi.copy()
        comps = {k: closed_classes(orbs, k) for k in range(N)}
        self.their_bal = {}
        active = set()
        for k in range(N):
            cl, rest = comps[k]
            self.their_bal[k] = [frozenset(S) for S in cl + ([rest] if rest else [])]
            for S in self.their_bal[k]:
                assert S in self.classes[k], (k, S)
                active.add((k, S))
        for t, (name, co, lo, hi) in enumerate(self.rows):
            if name[0] != 'bal': continue
            _, i, k, S = name
            if (k, S) in active: rlo[t] = rhi[t] = len(S - set(self.pair[i]))
            else: rlo[t], rhi[t] = -INF, INF
        ch = np.nonzero((rlo != self.cur_rlo) | (rhi != self.cur_rhi))[0]
        if len(ch): self.h.changeRowsBounds(len(ch), ch.astype(np.int32), rlo[ch], rhi[ch])
        self.cur_rlo, self.cur_rhi = rlo, rhi
        zero = set()
        for P in PAIRS:
            for j in self.by_pair[P]:
                if orbs[j]['partner'] is not None: zero.add((P, orbs[j]['partner']))
        for (P, i) in self.cur_dich_zero - zero: self._dich_coef(P, i, 1)
        for (P, i) in zero - self.cur_dich_zero: self._dich_coef(P, i, 0)
        self.cur_dich_zero = zero
        return orbs

    def _dich_coef(self, P, i, v):
        r = self.rindex[('dich', P)]; col = self.y[(i, P)] + 2
        self.h.changeCoeff(r, col, float(v))

    def their_rows(self):
        """my row index for each row of general_model.build(self.orbs), in its order"""
        out = []
        for i in range(self.n):
            P = self.pair[i]
            for t in P: out.append(self.rindex[('F1', i, t)])
            for k in range(N):
                if k in P: continue
                for S in self.their_bal[k]: out.append(self.rindex[('bal', i, k, S)])
        for P, P2 in itertools.combinations(PAIRS, 2): out.append(self.rindex[('sym', P, P2)])
        for (i, P) in self.y:
            out.append(self.rindex[('oh', i, P)]); out.append(self.rindex[('ln', i, P)])
        for P in PAIRS: out.append(self.rindex[('dich', P)])
        return out

    def check_int(self, ynum, lb, ub):
        """exact test of a certificate (integer numerators over DEN) on the CURRENT leaf with column bounds lb, ub"""
        A = self.exact_matrix_current()
        rlo, rhi = self.cur_rlo, self.cur_rhi
        nz = np.nonzero(ynum)[0]
        if np.any(np.isinf(rhi[nz])) or np.any(np.isinf(rlo[nz])): return False
        U = int(sum(int(ynum[r]) * int(rhi[r] if ynum[r] > 0 else rlo[r]) for r in nz))
        cvec = A.T.dot(ynum)                               # int64
        lbi, ubi = lb.astype(np.int64), ub.astype(np.int64)
        L = int(np.minimum(cvec * lbi, cvec * ubi).sum())
        return L > U

    def exact_matrix_current(self):
        import scipy.sparse as sp
        if getattr(self, '_Akey', None) != frozenset(self.cur_dich_zero):
            r_, c_, v_ = [], [], []
            for t, (name, co, lo, hi) in enumerate(self.rows):
                for j, v in co: r_.append(t); c_.append(j); v_.append(v)
            A = sp.csr_matrix((np.array(v_, dtype=np.int64), (r_, c_)), shape=(self.m, self.nvar)).tolil()
            for (P, i) in self.cur_dich_zero: A[self.rindex[('dich', P)], self.y[(i, P)] + 2] = 0
            self._A = A.tocsr(); self._Akey = frozenset(self.cur_dich_zero)
        return self._A

    def phase1(self, lb, ub, vertex=True):
        """vertex=False: interior point without crossover (fast first test; duals approximate, the exact check decides);
        vertex=True: simplex, warm start (branch and bound)"""
        ch = np.nonzero((lb != self.cur_lb) | (ub != self.cur_ub))[0]
        if len(ch): self.h.changeColsBounds(len(ch), ch.astype(np.int32), lb[ch], ub[ch])
        self.cur_lb, self.cur_ub = lb, ub
        if vertex: self.h.setOptionValue('solver', 'simplex'); self.h.setOptionValue('presolve', 'off')
        else: self.h.setOptionValue('solver', 'ipm'); self.h.setOptionValue('run_crossover', 'off'); self.h.setOptionValue('presolve', 'on')
        self.h.run()
        st = self.h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal: return None, None, None
        sol = self.h.getSolution()
        obj = self.h.getInfo().objective_function_value
        return obj, np.array(sol.row_dual), np.array(sol.col_value[:self.nvar])

    def certify(self, maxnodes=400):
        stats = {'nodes': 0}
        lb0, ub0 = self.cur_lb.copy(), self.cur_ub.copy()
        if not hasattr(self, 'cache'): self.cache = []
        for k_, yc in enumerate(self.cache):                          # certificates of earlier leaves (same job, same rows) often still hold
            if self.check_int(yc, lb0, ub0):
                self.cache.insert(0, self.cache.pop(k_)); nz = np.nonzero(yc)[0]
                return {'status': 'dead', 'level': 'LP', 'nodes': 0, 'tree': {'cert_sparse': [[int(r), int(yc[r])] for r in nz]}}
        vtx = os.environ.get('ROOTIPM') is None                       # default: simplex vertex duals (sparse, reusable)
        obj, dual, x = self.phase1(lb0, ub0, vertex=vtx)
        if maxnodes == 0 and not (obj is not None and obj > 1e-7):
            return {'status': 'open', 'why': 'lp-feasible' if obj is not None else 'lp-error', 'nodes': 1}
        if obj is not None and obj > 1e-7:
            for sgn in (1, -1):
                ynum = np.rint(sgn * dual * DEN).astype(np.int64)
                ynum[np.isinf(self.cur_rhi) & np.isinf(self.cur_rlo)] = 0
                if self.check_int(ynum, lb0, ub0):
                    nz = np.nonzero(ynum)[0]; self.cache.insert(0, ynum); del self.cache[int(os.environ.get('CACHE', 8)):]
                    return {'status': 'dead', 'level': 'LP', 'nodes': 1, 'tree': {'cert_sparse': [[int(r), int(ynum[r])] for r in nz]}}
            if maxnodes == 0: return {'status': 'open', 'why': 'rounding', 'nodes': 1}

        def rec(lb, ub):
            stats['nodes'] += 1
            if stats['nodes'] > maxnodes: return 'limit'
            obj, dual, x = self.phase1(lb, ub)
            if obj is None: return 'lp-error'
            if obj > 1e-7:
                for sgn in (1, -1):
                    ynum = np.rint(sgn * dual * DEN).astype(np.int64)
                    ynum[np.isinf(self.cur_rhi) & np.isinf(self.cur_rlo)] = 0
                    if self.check_int(ynum, lb, ub):
                        nz = np.nonzero(ynum)[0]
                        return {'cert_sparse': [[int(r), int(ynum[r])] for r in nz]}
                return 'rounding'
            frac = [(abs(x[j] - round(x[j])), j) for j in range(self.nvar) if lb[j] < ub[j] and abs(x[j] - round(x[j])) > 1e-6]
            if not frac: return ('integral', x)
            _, j = max(frac); k = int(np.floor(x[j]))
            out = {'var': int(j), 'le': k}
            ub1 = ub.copy(); ub1[j] = k; lb2 = lb.copy(); lb2[j] = k + 1
            for key, (l_, u_) in (('lo', (lb, ub1)), ('hi', (lb2, ub))):
                sub = rec(l_, u_)
                if not isinstance(sub, dict): return sub
                out[key] = sub
            return out
        t = rec(lb0.copy(), ub0.copy())
        # restore the leaf's bounds
        ch = np.nonzero((lb0 != self.cur_lb) | (ub0 != self.cur_ub))[0]
        if len(ch): self.h.changeColsBounds(len(ch), ch.astype(np.int32), lb0[ch], ub0[ch])
        self.cur_lb, self.cur_ub = lb0, ub0
        if isinstance(t, dict): return {'status': 'dead', 'level': 'LP' if 'cert_sparse' in t else 'branch', 'nodes': stats['nodes'], 'tree': t}
        if isinstance(t, tuple): return {'status': 'alive', 'nodes': stats['nodes']}
        return {'status': 'unknown', 'why': t, 'nodes': stats['nodes']}

    def export_tree(self, tree):
        """sparse certificates re-indexed to general_model.build's rows: [[their row, numerator], ...] over DEN"""
        pos = {r: t for t, r in enumerate(self.their_rows())}
        def ex(T):
            if 'cert_sparse' in T:
                return {'cert_sparse': [[pos[r], v] for r, v in T['cert_sparse']], 'den': DEN}
            return {'var': T['var'], 'le': T['le'], 'lo': ex(T['lo']), 'hi': ex(T['hi'])}
        return ex(tree)


if __name__ == "__main__":
    ci, job = sys.argv[1], int(sys.argv[2])
    po = load_job(ci, job); In = {int(x): v for x, v in po["In"].items()}; A = po["A"]
    assert A is None, "leaves with an A-block: not handled by pair_fast (use the orbit LP)"
    leaves = [[tuple(z) for z in L] for L in po["leaves"]]
    lo_, hi_ = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(leaves))
    hi_ = min(hi_, len(leaves))
    asg = [{c: tuple(z for z in L if z[0] == c) for c in range(N)} for L in leaves[lo_:hi_]]
    t0 = time.time(); F = FastPair(In, asg); tb = time.time() - t0
    os.makedirs("logs/pair_certs", exist_ok=True)
    tag = "" if (lo_, hi_) == (0, len(leaves)) else "_%d_%d" % (lo_, hi_)
    out = gzip.open("logs/pair_certs/leaf_%s_%d%s.jsonl.gz" % (ci, job, tag), "wt")
    stat = Counter(); t1 = time.time(); openl = []
    for t, a in zip(range(lo_, hi_), asg):
        F.set_leaf(a)
        r = F.certify(maxnodes=int(os.environ.get('MAXBB', 400)))
        stat[r['status'] + '/' + str(r.get('level', r.get('why', '')))] += 1
        rec = {'leaf': t, 'status': r['status'], 'nodes': r['nodes']}
        if r['status'] == 'dead': rec['level'] = r['level']; rec['tree'] = F.export_tree(r['tree'])
        else:
            rec['why'] = r.get('why'); openl.append(t)
            if r['status'] != 'open': print("NOT DEAD leaf", t, r); sys.stdout.flush()
        out.write(json.dumps(rec) + chr(10))
        if (t - lo_ + 1) % 2000 == 0: print("  %d leaves %s %.1f ms/leaf" % (t - lo_ + 1, dict(stat), 1000 * (time.time() - t1) / (t - lo_ + 1))); sys.stdout.flush()
    out.close()
    json.dump({"ci": ci, "job": job, "range": [lo_, hi_], "open": openl}, open("logs/pair_certs/open_%s_%d%s.json" % (ci, job, tag), "w"))
    print("PAIRFAST %s job %d [%d,%d): %s  model %.1fs, %.1f ms/leaf, total %.0fs %s" % (ci, job, lo_, hi_, dict(stat), tb,
          1000 * (time.time() - t1) / max(1, hi_ - lo_), time.time() - t0, "ALL DEAD" if stat.get('dead/LP', 0) + stat.get('dead/branch', 0) == hi_ - lo_ else "NOT ALL DEAD"))
