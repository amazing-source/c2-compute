"""g- = 18, torsion regime t = 1 (r0 = 1, rho = 1): far layer as CP-SAT (C1, 27/09).
T_FF = Z + lam k k^T (mod 7), lam in {1, 3} (square classes), k in F_7^30 with X k = 0 (and k^T k = 0, automatic from
tr Z = 0).  Hence k_f = 0 iff Z_ff = 0 (rigid row); otherwise -Z_ff/lam must be a nonzero square and k_f = sg_f * s_f with
s_f = sqrt(-Z_ff/lam) (fixed representative) and a sign sg_f.
Model (on top of the local model (i)+(ii)+(iii), options filtered by full balance of the torsion and Z_ff in -lam{0, squares}):
  pairwise: T_fg = Z(j,k) + lam * sg_f sg_g * s(j) s(k)  in {0, +-1}; t_fg = [T_fg != 0] channelled through the sign product
  p_fg = [sg_f != sg_g];  D-partner variables and per-block counts exactly as q7_rigid_cpsat2;
  X k = 0 (mod 7): 19 linear congruences on the signed k (y+/- = x AND sign);  global sign flip fixed (sg_0 = 0).
Z is the Schur complement with ALL torsion balanced (P^- generalised inverse, x_f in im P).
Usage: python q7_t1_cpsat.py IDX [TL] [WORKERS] [--maxopts K]   (IDX in the list of t = 1 classes)"""
import sys, time, itertools
import numpy as np
from ortools.sat.python import cp_model
from q7_g16_test import classes_t_ge, ball, ginv_sym, K2, K3, L, FAR, P7, orbit_options
from q7_g18_regimes import torsion_basis, z_vector
from q7_rigid_cpsat2 import shadow_info

SQRT = {}
for v in range(1, P7):
    SQRT.setdefault(v * v % P7, v)          # 1 -> 1, 4 -> 2, 2 -> 3


def build_t1(tau, eta, lam, maxopts=None):
    Pm = ball(tau, eta)
    Pp = ginv_sym(Pm)
    tb = torsion_basis(Pm)
    assert len(tb) == 1
    Tb = np.array(tb) % P7
    zv = z_vector()
    opts, X, S = [], [], []
    for (op, Xc) in orbit_options(tau, eta, K2(tau, eta)):
        keep = ((zv @ Xc) % P7 == 0) & np.all((Tb @ Xc) % P7 == 0, axis=0)
        zz = (3 + np.einsum('ij,ij->j', Xc, (Pp @ Xc) % P7)) % P7
        sval = []
        for j in range(len(op)):
            if not keep[j]:
                sval.append(None); continue
            q = (-int(zz[j]) * pow(lam, P7 - 2, P7)) % P7        # k_f^2
            if q == 0:
                sval.append(0)
            elif q in SQRT:
                sval.append(SQRT[q])
            else:
                sval.append(None)
        idx = [j for j in range(len(op)) if sval[j] is not None]
        if maxopts:
            idx = idx[:maxopts]
        opts.append([op[j] for j in idx]); X.append(Xc[:, idx]); S.append([sval[j] for j in idx])
    n = len(opts)
    Zm = {}
    for f in range(n):
        for g in range(n):
            if f != g:
                Zm[(f, g)] = (X[f].T @ ((Pp @ X[g]) % P7)) % P7
    return opts, X, S, Zm


def make_model(tau, eta, lam, opts, X, S, Zm, skip_local=False):
    n = len(opts)
    m = cp_model.CpModel()
    x = [[m.NewBoolVar('x%d_%d' % (f, j)) for j in range(len(opts[f]))] for f in range(n)]
    sg = [m.NewBoolVar('s%d' % f) for f in range(n)]
    m.Add(sg[0] == 0)
    for f in range(n):
        m.AddExactlyOne(x[f])
    if not skip_local:
        k2, k3 = K2(tau, eta), K3(tau, eta)
        nonmu = [e for e in itertools.combinations(L, 2) if tau[e[0]] != e[1]]
        use = {e: [] for e in nonmu}; acc2, acc3 = {}, {}
        for f in range(n):
            z, sz = FAR[f]
            for j, (e, e2) in enumerate(opts[f]):
                v = x[f][j]; use[e].append(v); use[e2].append(v)
                for a in z:
                    for b in e: acc2.setdefault((a, b), []).append(v)
                for a in sz:
                    for b in e2: acc2.setdefault((a, b), []).append(v)
                for b in e:
                    for d in e2: acc3.setdefault((b, d), []).append(v)
                for b in e2:
                    for d in e: acc3.setdefault((b, d), []).append(v)
        for e in nonmu:
            m.AddExactlyOne(use[e])
        for a in L:
            for b in L:
                m.Add(sum(acc2.get((a, b), [])) == int(k2[a, b]))
                m.Add(sum(acc3.get((a, b), [])) == int(k3[a, b]))
    # sign products and pairwise channelling
    pv, tv, dv = {}, {}, {}
    for f in range(n):
        for g in range(f + 1, n):
            p = m.NewBoolVar('p%d_%d' % (f, g))                 # p = sg_f xor sg_g
            m.AddBoolXOr([sg[f], sg[g], p.Not()])
            pv[(f, g)] = pv[(g, f)] = p
            tv[(f, g)] = tv[(g, f)] = m.NewBoolVar('t%d_%d' % (f, g))
            dv[(f, g)] = dv[(g, f)] = m.NewBoolVar('d%d_%d' % (f, g))
            m.AddImplication(dv[(f, g)], tv[(f, g)].Not())
    for f in range(n):
        for g in range(n):
            if g == f:
                continue
            Z = Zm[(f, g)]
            sf = np.array(S[f]); sgv = np.array(S[g])
            for j in range(len(opts[f])):
                for prod, plit in ((1, pv[(f, g)].Not()), (-1, pv[(f, g)])):
                    T = (Z[j] + lam * prod * sf[j] * sgv) % P7
                    nz = [x[g][k] for k in np.nonzero((T == 1) | (T == 6))[0]]
                    zr = [x[g][k] for k in np.nonzero(T == 0)[0]]
                    m.AddBoolOr(nz + [x[f][j].Not(), plit.Not(), tv[(f, g)].Not()])
                    m.AddBoolOr(zr + [x[f][j].Not(), plit.Not(), tv[(f, g)]])
    for f in range(n):
        noncanc = [x[f][j] for j in range(len(opts[f])) if shadow_info(*opts[f][j])[1] is None]
        m.Add(sum(dv[(f, g)] for g in range(n) if g != f) == sum(noncanc))
    supp = [set(a // 2 for a in FAR[f][0]) for f in range(n)]
    for f in range(n):
        for j, (e, e2) in enumerate(opts[f]):
            s, lc = shadow_info(e, e2)
            for zb in range(6):
                shz = sum(1 for l in s if l // 2 == zb)
                thr = [g for g in range(n) if g != f and zb in supp[g]]
                base = 2 if zb in supp[f] else 4
                canc_here = 2 if (lc is not None and lc // 2 == zb) else 0
                m.Add(sum(tv[(f, g)] for g in thr) + 2 * sum(dv[(f, g)] for g in thr) + shz + canc_here == base
                      ).OnlyEnforceIf(x[f][j])
    # X k = 0 (mod 7): k_f = (+1 if sg_f = 0 else -1) * s_f(j)
    for i in range(19):
        terms = []
        for f in range(n):
            for j in range(len(opts[f])):
                c = int(X[f][i, j]) * int(S[f][j]) % P7
                if c == 0:
                    continue
                yp = m.NewBoolVar(''); yn = m.NewBoolVar('')
                m.AddBoolAnd([x[f][j], sg[f].Not()]).OnlyEnforceIf(yp)
                m.AddBoolOr([x[f][j].Not(), sg[f], yp])
                m.AddImplication(yp, x[f][j]); m.AddImplication(yp, sg[f].Not())
                m.AddBoolAnd([x[f][j], sg[f]]).OnlyEnforceIf(yn)
                m.AddBoolOr([x[f][j].Not(), sg[f].Not(), yn])
                terms.append(c * yp - c * yn)
        if terms:
            q = m.NewIntVar(-60, 60, '')
            m.Add(sum(terms) == 7 * q)
    return m, x, sg


def solve(idx, tl=240, workers=1, maxopts=None):
    cls = [c for c in classes_t_ge(1) if c[2] == 1]
    tau, eta, t, osz = cls[idx]
    out = []
    for lam in (1, 3):
        t0 = time.time()
        opts, X, S, Zm = build_t1(tau, eta, lam, maxopts)
        if any(len(o) == 0 for o in opts):
            out.append((lam, 'dead-unary')); continue
        m, x, sg = make_model(tau, eta, lam, opts, X, S, Zm)
        s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = workers
        out.append((lam, s.StatusName(s.Solve(m)), round(time.time() - t0)))
    return out


if __name__ == '__main__':
    idx = int(sys.argv[1]); tl = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    maxopts = int(sys.argv[sys.argv.index('--maxopts') + 1]) if '--maxopts' in sys.argv else None
    print(solve(idx, tl, workers, maxopts), flush=True)
