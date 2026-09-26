"""CP-SAT version of the full-block local model (constraints (i)-(vi) of fullblock_rigid.py), with optional joint odd
far-far block T_FF and the exact equations (E1), (E2), row counts of tff_check.py.
Usage: python fullblock_cpsat.py cfg.json time_limit workers [joint]"""
import itertools, json, sys, time
import numpy as np
from ortools.sat.python import cp_model
from fullblock_rigid import mates_from_D, K2K3, far_roots, lab, pi, L, zmatrix, rigid_report


def build(m, cycles, anchors, joint=False, equiv=None):
    K2, K3 = K2K3(m)
    far, reps = far_roots()
    nonm = [e for e in itertools.combinations(L, 2) if m[e[0]] != e[1]]
    md = cp_model.CpModel()
    roots = [z for Z in reps for z in Z]                      # 60, sigma z = roots[i ^ 1]
    x = {(i, e): md.NewBoolVar('x%d_%d_%d' % (i, e[0], e[1])) for i in range(60) for e in nonm}
    for i in range(60):
        md.AddExactlyOne(x[i, e] for e in nonm)
    for e in nonm:
        md.AddExactlyOne(x[i, e] for i in range(60))
    U = {}
    for i in range(60):
        for b in L:
            U[i, b] = md.NewBoolVar('u%d_%d' % (i, b))
            md.Add(U[i, b] == sum(x[i, e] for e in nonm if b in e))
    for a in L:                                               # (ii)
        for b in L:
            md.Add(sum(U[i, b] for i in range(60) if a in roots[i]) == int(K2[a, b]))
    W = {}
    for Zi in range(30):                                      # (iii)
        i, j = 2 * Zi, 2 * Zi + 1
        for b in L:
            for d in L:
                w = md.NewBoolVar('w%d_%d_%d' % (Zi, b, d))
                md.AddBoolAnd([U[i, b], U[j, d]]).OnlyEnforceIf(w)
                md.AddBoolOr([U[i, b].Not(), U[j, d].Not()]).OnlyEnforceIf(w.Not())
                W[Zi, b, d] = w
    for b in L:
        for d in L:
            md.Add(sum(W[Zi, b, d] + W[Zi, d, b] for Zi in range(30)) == int(K3[b, d]))
    for Zi in range(30):
        i, j = 2 * Zi, 2 * Zi + 1
        z = roots[i]
        for b in L:                                           # no cancellation
            md.AddBoolOr([U[i, b].Not(), U[j, b].Not()])
        for a in z:                                           # (iv)
            md.AddBoolOr([U[i, a].Not(), U[j, pi[a]].Not()])
            md.AddBoolOr([U[i, pi[a]].Not(), U[j, a].Not()])
        for k in (i, j):
            zk = roots[k]
            for C in cycles:                                  # (v)
                lc = sorted(lab(C))
                md.Add(len({a // 2 for a in zk} & set(C)) + sum(U[k, v] for v in lc) + sum(U[k ^ 1, v] for v in lc) == len(C))
        anc = anchors.get(frozenset((z[0] // 2, z[1] // 2)))
        if anc is not None:                                   # (vi)
            for v in lab([anc]):
                md.Add(U[i, v] == 0); md.Add(U[j, v] == 0)
    if equiv:                                                 # Phi(g z) = g Phi(z) for the label permutation equiv
        ridx = {tuple(sorted(z)): i for i, z in enumerate(roots)}
        for i in range(60):
            gz = tuple(sorted(equiv[a] for a in roots[i]))
            for e in nonm:
                ge = tuple(sorted(equiv[v] for v in e))
                md.Add(x[ridx[gz], ge] == x[i, e])
    T = None
    if joint:
        T = add_odd_far(md, U, roots, reps, anchors)
    return md, x, U, roots, reps, nonm, T


def add_odd_far(md, U, roots, reps, anchors):
    """T[f,g] in {-1,0,1} for far orbits f != g (f = sigma-pair index, representative roots[2f]); (E1), (E2), 6 per row."""
    n = 30
    tp, tm = {}, {}
    for f in range(n):
        for g in range(f + 1, n):
            tp[f, g] = tp[g, f] = md.NewBoolVar('tp%d_%d' % (f, g))
            tm[f, g] = tm[g, f] = md.NewBoolVar('tm%d_%d' % (f, g))
            md.AddBoolOr([tp[f, g].Not(), tm[f, g].Not()])
    for f in range(n):
        md.Add(sum(tp[f, g] + tm[f, g] for g in range(n) if g != f) == 6)
        z = reps[f][0]
        for g in range(n):
            if g != f and {reps[g][0][0] // 2, reps[g][0][1] // 2} == {z[0] // 2, z[1] // 2}:
                if anchors.get(frozenset((z[0] // 2, z[1] // 2))) is not None:
                    md.Add(tp[f, g] == 0); md.Add(tm[f, g] == 0)
    # shadow s_g(v) = U[2g, v] - U[2g+1, v]; products T_fg * s_g(v) via 4 bool products
    prod = {}

    def pv(t, u, key):
        if key in prod:
            return prod[key]
        p = md.NewBoolVar('p_' + '_'.join(map(str, key)))
        md.AddBoolAnd([t, u]).OnlyEnforceIf(p)
        md.AddBoolOr([t.Not(), u.Not()]).OnlyEnforceIf(p.Not())
        prod[key] = p
        return p
    for f in range(n):
        z = reps[f][0]
        dvec = [0] * 12
        for a in z:
            dvec[a] += 1; dvec[pi[a]] -= 1
        Bf = {}
        for a in z:
            Bf[a // 2] = 1 if a % 2 == 0 else -1
        for v in L:
            terms = []
            for g in range(n):
                if g == f:
                    continue
                terms += [pv(tp[f, g], U[2 * g, v], ('pp', f, g, v)), pv(tm[f, g], U[2 * g + 1, v], ('mm', f, g, v))]
                terms_neg = [pv(tp[f, g], U[2 * g + 1, v], ('pm', f, g, v)), pv(tm[f, g], U[2 * g, v], ('mp', f, g, v))]
                terms += [(-1, t) for t in terms_neg]
            lhs = sum(t if not isinstance(t, tuple) else -t[1] for t in terms)
            md.Add(lhs == -(U[2 * f, v] - U[2 * f + 1, v]) - dvec[v])       # (E1)
        for e in range(6):
            lhs = []
            for g in range(n):
                if g == f:
                    continue
                zg = reps[g][0]
                bge = next((1 if a % 2 == 0 else -1 for a in zg if a // 2 == e), 0)
                if bge:
                    lhs.append(bge * (tp[f, g] - tm[f, g]))
            md.Add(sum(lhs) == (U[2 * f, 2 * e + 1] - U[2 * f + 1, 2 * e + 1]) - (U[2 * f, 2 * e] - U[2 * f + 1, 2 * e]))  # (E2)
    return tp, tm


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    joint = len(sys.argv) > 4 and sys.argv[4] == 'joint'
    cycles = [tuple(c) for c in cfg['cycles']]
    anchors = {frozenset(map(int, k.split(','))): v for k, v in cfg.get('anchors', {}).items()}
    m = mates_from_D(cycles)
    equiv = None
    if 'rotate' in cfg:                                       # rotation of a single D-cycle by cfg['rotate'] steps
        cyc = cycles[0]; k = cfg['rotate']; nxt = {cyc[j]: cyc[(j + k) % len(cyc)] for j in range(len(cyc))}
        equiv = [2 * nxt[v // 2] + v % 2 for v in L]
        assert all(m[equiv[v]] == equiv[m[v]] for v in L)
    t0 = time.time()
    md, x, U, roots, reps, nonm, T = build(m, cycles, anchors, joint=joint, equiv=equiv)
    print('D-cycles', cycles, '| joint T_FF' if joint else '| Phi only', '| build %.0fs' % (time.time() - t0), flush=True)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tl
    solver.parameters.num_workers = workers
    st = solver.Solve(md)
    print('status', solver.StatusName(st), '%.0fs' % (time.time() - t0), flush=True)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # convert to the var/chosen format of fullblock_rigid for the rigid test
        var, chosen = [], []
        for Zi in range(30):
            e = next(e for e in nonm if solver.Value(x[2 * Zi, e]))
            e2 = next(e for e in nonm if solver.Value(x[2 * Zi + 1, e]))
            var.append((Zi, e, e2)); chosen.append(Zi)
        Z = zmatrix(var, reps, chosen)
        rig, bad = rigid_report(Z, reps, anchors)
        print('  rigid far orbits', len(rig), rig, '| violating rigid rows', len(bad), bad[:6], flush=True)
        json.dump({'cycles': cfg['cycles'], 'anchors': cfg.get('anchors', {}),
                   'Phi': {str(list(reps[Zi][0])): [list(e), list(e2)] for Zi, e, e2 in var}},
                  open(sys.argv[1].replace('.json', '_sol.json'), 'w'), indent=1)
