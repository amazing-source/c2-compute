"""g- = 18, t = 0 split block c, ALL far rows rigid (T_FF = Z mod 7): strengthened CP-SAT model (C1, 27/09).
Local data at c: one option (Phi(z), Phi(sigma z)) per far orbit f among those with Z_ff = 0; (i) each non-mu pair once; (ii), (iii).
Far structure (T_ff' = lift(Z_ff') in {0, +-1}, Z = 3I + X^T P^- X):
  (P) pairwise: the chosen options of f and g have Z_fg in {0, +-1};
  t_fg = [T_fg != 0] (boolean, channelled by conditioned clauses);
  (D) D-partners: a far orbit with a non-cancelling shadow has its D-partner D(f) among the far orbits: boolean d_fg (symmetric),
      sum_g d_fg = [f non-cancelling], d_fg => T_fg = 0;  (a cancelling shadow {l} = Phi(z) cap Phi(sigma z) has D(f) = o_l in star(c))
  (a) per-block counts (F1 on the row of f, Q_fW = |T_fW| except Q = 2 on D(f)): for every block z != c,
      #{far g through z : t_fg} + #{labels of s_f in block z} + 2 [z in supp D(f)] = 2 if z in supp f, else 4;
      (this contains (b): the row of f inside star(z), z in supp f, has 2 nonzeros or 0 - the local structure at z)
  (c) BT = 0 is automatic mod 7 for Z and |sums| <= 6 under (a): redundant, not added.
  (d) final check of a complete solution: the full T (42 x 42, gauge u = rep) satisfies T^2 + T = 12I - B^T B exactly, and at every
      block d the star(d) block has the shape of P_tau - P_eta (2-regular or pivots) with holonomy (-1)^(l/2) on each cycle;
      on failure the assignment is excluded (no-good) and the search resumes.
Usage: python q7_rigid_cpsat2.py IDX [TIME_LIMIT] [WORKERS] [--maxopts K]  (IDX in the list of t = 0 split classes)"""
import sys, time, itertools
import numpy as np
from ortools.sat.python import cp_model
from q7_g16_test import classes_t_ge, ball_disc, split, unary_filter, K2, K3, L, FAR, P7

C_BLOCK = 6                     # ball index of block c; labels l in 0..11 live in blocks l // 2 = 0..5


def lift(v):
    v = int(v) % P7
    return 1 if v == 1 else (-1 if v == 6 else 0)


def far_B(f):
    """B column of far orbit f (rep z = FAR[f][0]) on the 6 other blocks: dict block -> sign."""
    z = FAR[f][0]
    return {a // 2: (1 if a % 2 == 0 else -1) for a in z}


def build(tau, eta, maxopts=None):
    t, res, Pp = unary_filter(tau, eta, regime_t2=True)          # balance (z_c) + Z_ff = 0
    opts = [list(r[0]) for r in res]
    X = [r[1] for r in res]
    if maxopts:
        opts = [o[:maxopts] for o in opts]
        X = [x[:, :maxopts] for x in X]
    n = len(opts)
    Zm = {}
    for f in range(n):
        for g in range(n):
            if f != g:
                Zm[(f, g)] = (X[f].T @ ((Pp @ X[g]) % P7)) % P7
    return opts, X, Zm, Pp


def shadow_info(e, e2):
    s = {}
    for l in e:
        s[l] = s.get(l, 0) + 1
    for l in e2:
        s[l] = s.get(l, 0) - 1
    s = {l: v for l, v in s.items() if v}
    canc = set(e) & set(e2)
    return s, (next(iter(canc)) if canc else None)


def make_model(tau, eta, opts, Zm):
    n = len(opts)
    m = cp_model.CpModel()
    x = [[m.NewBoolVar('x%d_%d' % (f, j)) for j in range(len(opts[f]))] for f in range(n)]
    for f in range(n):
        m.AddExactlyOne(x[f])
    k2, k3 = K2(tau, eta), K3(tau, eta)
    nonmu = [e for e in itertools.combinations(L, 2) if tau[e[0]] != e[1]]
    use = {e: [] for e in nonmu}
    acc2, acc3 = {}, {}
    for f in range(n):
        z, sz = FAR[f]
        for j, (e, e2) in enumerate(opts[f]):
            v = x[f][j]
            use[e].append(v); use[e2].append(v)
            for a in z:
                for b in e:
                    acc2.setdefault((a, b), []).append(v)
            for a in sz:
                for b in e2:
                    acc2.setdefault((a, b), []).append(v)
            for b in e:
                for d in e2:
                    acc3.setdefault((b, d), []).append(v)
            for b in e2:
                for d in e:
                    acc3.setdefault((b, d), []).append(v)
    for e in nonmu:
        m.AddExactlyOne(use[e])
    for a in L:
        for b in L:
            m.Add(sum(acc2.get((a, b), [])) == int(k2[a, b]))
            m.Add(sum(acc3.get((a, b), [])) == int(k3[a, b]))
    # t_fg and pairwise support / channelling
    tv = {}
    for f in range(n):
        for g in range(f + 1, n):
            tv[(f, g)] = tv[(g, f)] = m.NewBoolVar('t%d_%d' % (f, g))
    for f in range(n):
        for g in range(n):
            if g == f:
                continue
            Z = Zm[(f, g)]
            for j in range(len(opts[f])):
                nz = [x[g][k] for k in np.nonzero((Z[j] == 1) | (Z[j] == 6))[0]]
                zr = [x[g][k] for k in np.nonzero(Z[j] == 0)[0]]
                # x_fj and t_fg  => some nonzero option of g;  x_fj and not t_fg => some zero option of g
                m.AddBoolOr(nz + [x[f][j].Not(), tv[(f, g)].Not()])
                m.AddBoolOr(zr + [x[f][j].Not(), tv[(f, g)]])
    # D-partners among far orbits
    dv = {}
    for f in range(n):
        for g in range(f + 1, n):
            dv[(f, g)] = dv[(g, f)] = m.NewBoolVar('d%d_%d' % (f, g))
            m.AddImplication(dv[(f, g)], tv[(f, g)].Not())
    canc_of = [[shadow_info(e, e2)[1] for (e, e2) in opts[f]] for f in range(n)]
    for f in range(n):
        noncanc = [x[f][j] for j in range(len(opts[f])) if canc_of[f][j] is None]
        m.Add(sum(dv[(f, g)] for g in range(n) if g != f) == sum(noncanc))
    # (a) per-block counts
    supp = [set(a // 2 for a in FAR[f][0]) for f in range(n)]
    for f in range(n):
        for j, (e, e2) in enumerate(opts[f]):
            s, lc = shadow_info(e, e2)
            for zb in range(6):
                shz = sum(1 for l in s if l // 2 == zb)
                far_through = [g for g in range(n) if g != f and zb in supp[g]]
                base = 2 if zb in supp[f] else 4
                canc_here = 2 if (lc is not None and lc // 2 == zb) else 0
                expr = sum(tv[(f, g)] for g in far_through) + 2 * sum(dv[(f, g)] for g in far_through)
                m.Add(expr + shz + canc_here == base).OnlyEnforceIf(x[f][j])
    return m, x, tv, dv


def full_T(tau, eta, opts, Zm, choice):
    """42 x 42 T in the gauge: star orbit g_l (rep (c+, l)), far orbit f (rep FAR[f][0]); plus B (7 x 42)."""
    n = len(opts)
    idx_star = {l: l for l in L}                   # orbits 0..11 = star, 12..41 = far
    T = np.zeros((42, 42), dtype=np.int64)
    B = np.zeros((7, 42), dtype=np.int64)
    for l in L:
        B[C_BLOCK, l] = 1
        B[l // 2, l] = 1 if l % 2 == 0 else -1
        T[l, tau[l]] += 1
        T[l, eta[l]] -= 1
    for f in range(n):
        o = 12 + f
        for b, sg in far_B(f).items():
            B[b, o] = sg
        e, e2 = opts[f][choice[f]]
        s, _ = shadow_info(e, e2)
        for l, v in s.items():
            T[o, l] = T[l, o] = v
        for g in range(n):
            if g != f:
                T[o, 12 + g] = lift(Zm[(f, g)][choice[f], choice[g]])
    return T, B


def final_check(T, B):
    ok_id = np.array_equal(T @ T + T, 12 * np.eye(42, dtype=np.int64) - B.T @ B)
    ok_sym = np.array_equal(T, T.T)
    return ok_id and ok_sym


def solve(idx, tl=240, workers=1, maxopts=None, max_rounds=20):
    cls = [c for c in classes_t_ge(0) if c[2] == 0]
    cls = [c for c in cls if split(*ball_disc(c[0], c[1]))]
    tau, eta, t, osz = cls[idx]
    t0 = time.time()
    opts, X, Zm, Pp = build(tau, eta, maxopts)
    if any(len(o) == 0 for o in opts):
        return 'dead-unary', 0
    m, x, tv, dv = make_model(tau, eta, opts, Zm)
    print('model built (%.0fs): options per orbit %s' % (time.time() - t0, [len(o) for o in opts]), flush=True)
    for rnd in range(max_rounds):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = max(10, tl - (time.time() - t0))
        s.parameters.num_workers = workers
        st = s.Solve(m)
        name = s.StatusName(st)
        if name not in ('OPTIMAL', 'FEASIBLE'):
            return name, rnd
        choice = [next(j for j in range(len(opts[f])) if s.Value(x[f][j])) for f in range(len(opts))]
        T, B = full_T(tau, eta, opts, Zm, choice)
        if final_check(T, B):
            return 'CANDIDATE (passes T^2 + T = 12I - B^T B)', rnd
        m.AddBoolOr([x[f][choice[f]].Not() for f in range(len(opts))])      # no-good
    return 'round cap', max_rounds


if __name__ == '__main__':
    idx = int(sys.argv[1])
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    maxopts = int(sys.argv[sys.argv.index('--maxopts') + 1]) if '--maxopts' in sys.argv else None
    print(solve(idx, tl, workers, maxopts), flush=True)
