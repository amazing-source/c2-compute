"""COMPLETE odd sector at a FULL block c, as one CP-SAT model (for the cloud).
The odd matrix S (49 x 49: 7 blocks + 12 star orbits o_v + 30 far orbits f) is fully parametrised at a full block by
  - the shadows s_f(v) = [v in Phi(z_f)] - [v in Phi(sigma z_f)]  (Phi: far roots -> non-m pairs, local constraints (i)-(vi));
  - the far-far block T (symmetric, zero diagonal, entries -1/0/1);
  (block rows B are fixed by the representatives; the star-star block is 0 because tau_c = eta_c = m).
S^2 + S = 12I is then EQUIVALENT to the following equations (NOTE_S29 par. 3), all imposed here:
  (a)  sum_f B[e,f] s_f(v) = 0                              for every block e != c and label v     (row d_e, column o_v)
  (b)  sum_f s_f(v) s_f(w) = 12[v=w] - 1 - sgn(v)sgn(w)[blk v = blk w]                             (o_v, o_w)
  (E1) sum_g T[f,g] s_g(v) = -(s_f(v) + d_f(v))                                                     (f, o_v)
  (E2) sum_g T[f,g] B[e,g] = -(s_f(e+) - s_f(e-))                                                   (f, d_e)
  (E3) B_f.B_g + s_f.s_g + (T^2)[f,g] + T[f,g] = 12[f=g]                                            (f, g)
(rows d_c / o_v against d_e are automatic.)  Infeasible => no involution has a full block of this D-structure (with the
given anchoring if any).  Symmetry breaking: none (the gauge is fixed by the D-structure).
Usage: python fullblock_odd_cpsat.py cfg.json time_limit workers"""
import itertools, json, sys, time
from ortools.sat.python import cp_model
from fullblock_rigid import mates_from_D, far_roots, pi, L
from fullblock_cpsat import build as build_phi


def sgn(v):
    return 1 if v % 2 == 0 else -1


def build_odd(m, cycles, anchors, equiv=None):
    md, x, U, roots, reps, nonm, _ = build_phi(m, cycles, anchors, joint=False, equiv=equiv)
    n = 30
    B = [[0] * 6 for _ in range(n)]
    D = [[0] * 12 for _ in range(n)]
    for f in range(n):
        for a in reps[f][0]:
            B[f][a // 2] = sgn(a)
            D[f][a] += 1
            D[f][pi[a]] -= 1
    s = {}
    for f in range(n):
        for v in L:
            s[f, v] = md.NewIntVar(-1, 1, 's%d_%d' % (f, v))
            md.Add(s[f, v] == U[2 * f, v] - U[2 * f + 1, v])
    # (a)
    for e in range(6):
        for v in L:
            md.Add(sum(B[f][e] * s[f, v] for f in range(n) if B[f][e]) == 0)
    # (b): s_f(v) s_f(w) products
    ss = {}
    for f in range(n):
        for v in L:
            for w in L:
                if w < v:
                    continue
                if w == v:
                    p = md.NewIntVar(0, 1, 'q%d_%d' % (f, v))
                    md.Add(p == U[2 * f, v] + U[2 * f + 1, v])          # s^2 = |s| (no cancellation)
                else:
                    p = md.NewIntVar(-1, 1, 'q%d_%d_%d' % (f, v, w))
                    md.AddMultiplicationEquality(p, [s[f, v], s[f, w]])
                ss[f, v, w] = p
    for v in L:
        for w in L:
            if w < v:
                continue
            rhs = 12 * (v == w) - 1 - (sgn(v) * sgn(w) if v // 2 == w // 2 else 0)
            md.Add(sum(ss[f, v, w] for f in range(n)) == rhs)
    # T variables
    t = {}
    for f in range(n):
        for g in range(f + 1, n):
            t[f, g] = t[g, f] = md.NewIntVar(-1, 1, 't%d_%d' % (f, g))
    absn = {}
    for f in range(n):
        for g in range(f + 1, n):
            a = md.NewBoolVar('a%d_%d' % (f, g))
            md.AddAbsEquality(a, t[f, g])
            absn[f, g] = absn[g, f] = a
    for f in range(n):
        md.Add(sum(absn[f, g] for g in range(n) if g != f) == 6)                         # (E3) diagonal
        z = reps[f][0]
        anc = anchors.get(frozenset((z[0] // 2, z[1] // 2)))
        if anc is not None:                                                              # T = 0 between anchored sisters
            for g in range(n):
                if g != f and {reps[g][0][0] // 2, reps[g][0][1] // 2} == {z[0] // 2, z[1] // 2}:
                    md.Add(t[f, g] == 0)
    # (E1)
    for f in range(n):
        for v in L:
            prods = []
            for g in range(n):
                if g == f:
                    continue
                p = md.NewIntVar(-1, 1, 'e1_%d_%d_%d' % (f, g, v))
                md.AddMultiplicationEquality(p, [t[f, g], s[g, v]])
                prods.append(p)
            md.Add(sum(prods) == -s[f, v] - D[f][v])
    # (E2)
    for f in range(n):
        for e in range(6):
            md.Add(sum(B[g][e] * t[f, g] for g in range(n) if g != f and B[g][e]) == -(s[f, 2 * e] - s[f, 2 * e + 1]))
    # (E3) off-diagonal
    for f in range(n):
        for g in range(f + 1, n):
            prods = []
            for h in range(n):
                if h in (f, g):
                    continue
                p = md.NewIntVar(-1, 1, 'e3_%d_%d_%d' % (f, g, h))
                md.AddMultiplicationEquality(p, [t[f, h], t[h, g]])
                prods.append(p)
            bb = sum(B[f][e] * B[g][e] for e in range(6))
            sdot = []
            for v in L:
                p = md.NewIntVar(-1, 1, 'sd_%d_%d_%d' % (f, g, v))
                md.AddMultiplicationEquality(p, [s[f, v], s[g, v]])
                sdot.append(p)
            md.Add(bb + sum(sdot) + sum(prods) + t[f, g] == 0)
    return md, x, s, t, reps, nonm


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    cycles = [tuple(c) for c in cfg['cycles']]
    anchors = {frozenset(map(int, k.split(','))): v for k, v in cfg.get('anchors', {}).items()}
    m = mates_from_D(cycles)
    equiv = None
    if 'rotate' in cfg:
        cyc = cycles[0]; k = cfg['rotate']; nxt = {cyc[j]: cyc[(j + k) % len(cyc)] for j in range(len(cyc))}
        equiv = [2 * nxt[v // 2] + v % 2 for v in L]
    t0 = time.time()
    md, x, s, t, reps, nonm = build_odd(m, cycles, anchors, equiv)
    print('complete odd model, D-cycles', cycles, '| anchored', sum(1 for v in anchors.values() if v is not None),
          '| equiv', equiv is not None, '| build %.0fs' % (time.time() - t0), flush=True)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tl
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = False
    st = solver.Solve(md)
    print('status', solver.StatusName(st), '%.0fs' % (time.time() - t0), flush=True)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = {'Phi': {str(list(reps[f][0])): [[v for v in L if solver.Value(s[f, v]) == 1],
                                               [v for v in L if solver.Value(s[f, v]) == -1]] for f in range(30)},
               'T': [[(solver.Value(t[f, g]) if g != f else 0) for g in range(30)] for f in range(30)]}
        json.dump(sol, open(sys.argv[1].replace('.json', '_oddsol.json'), 'w'))
        print('solution saved', flush=True)
