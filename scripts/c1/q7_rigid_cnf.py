"""CNF export of the strengthened all-rigid model (q7_rigid_cpsat2.py), for DRAT certification (kissat + drat-trim).
Same variables and constraints:
  x_fj (option j of far orbit f), t_fg = [T_fg != 0], d_fg (D-partner), f < g;
  exactly one x per orbit; (i) exactly one option per non-mu pair; (ii), (iii) cardinality equalities;
  channelling: x_fj & t_fg => OR{x_gk : Z(j,k) = +-1};  x_fj & -t_fg => OR{x_gk : Z(j,k) = 0};
  d_fg => -t_fg; cancelling option => no d_f.; non-cancelling option => at least one d_f.; at most one d_f. overall;
  per-block counts: totaliser over [t_fg (g through z)] + 2 x [d_fg (g through z)], and x_fj => (count == target_j).
The preprocessing (options with Z_ff = 0, Z mod 7 by exact linear algebra) is part of the reduction, not of the certificate.
Usage: python q7_rigid_cnf.py IDX OUT.cnf [--maxopts K] [--solve]   (--solve: check with pysat's CaDiCaL, small instances)"""
import sys, json, itertools
import numpy as np
_UID = itertools.count()
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from q7_g16_test import classes_t_ge, ball_disc, split, K2, K3, L, FAR
from q7_rigid_cpsat2 import build, shadow_info


def exact_counter(lits, pool, cnf):
    """bidirectional sequential counter: returns o[1..n] (o[j] <=> sum(lits) >= j), all literals distinct."""
    n = len(lits)
    prev = {}                                              # prev[j] <=> first i literals sum >= j (j = 1..i)
    for i, l in enumerate(lits, start=1):
        cur = {}
        for j in range(1, i + 1):
            v = pool.id(('cnt', next(_UID)))
            cur[j] = v
            a = prev.get(j)                                # sum of first i-1 >= j
            b = prev.get(j - 1) if j > 1 else None         # sum of first i-1 >= j-1 (true if j = 1)
            # v <= a
            if a is not None:
                cnf.append([-a, v])
            # v <= b & l
            if j == 1:
                cnf.append([-l, v])
            elif b is not None:
                cnf.append([-b, -l, v])
            # v => a or (b and l):  (-v, a, b) and (-v, a, l)   [b = true when j = 1]
            if j == 1:
                cnf.append([-v] + ([a] if a is not None else []) + [l])
            else:
                cnf.append([-v] + ([a] if a is not None else []) + [b])
                cnf.append([-v] + ([a] if a is not None else []) + [l])
        prev = cur
    return prev


def export(tau, eta, opts, Zm, skip_local=False):
    n = len(opts)
    pool = IDPool()
    cnf = CNF()
    X = lambda f, j: pool.id(('x', f, j))
    T = lambda f, g: pool.id(('t', min(f, g), max(f, g)))
    Dv = lambda f, g: pool.id(('d', min(f, g), max(f, g)))

    def eq(lits, k):
        if k > len(lits) or k < 0:
            cnf.append([])                             # empty clause: unsatisfiable
            return
        if k == 0:
            cnf.extend([[-v] for v in lits])
            return
        if k == len(lits):
            cnf.extend([[v] for v in lits])
            return
        enc = CardEnc.equals(lits=lits, bound=k, vpool=pool, encoding=EncType.seqcounter)
        cnf.extend(enc.clauses)
    for f in range(n):
        eq([X(f, j) for j in range(len(opts[f]))], 1)
    k2, k3 = K2(tau, eta), K3(tau, eta)
    nonmu = [e for e in itertools.combinations(L, 2) if tau[e[0]] != e[1]]
    use = {e: [] for e in nonmu}
    acc2, acc3 = {}, {}
    for f in range(n):
        z, sz = FAR[f]
        for j, (e, e2) in enumerate(opts[f]):
            v = X(f, j)
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
    if not skip_local:
        for e in nonmu:
            eq(use[e], 1)
        for a in L:
            for b in L:
                eq(acc2.get((a, b), []), int(k2[a, b]))
                eq(acc3.get((a, b), []), int(k3[a, b]))
    for f in range(n):
        for g in range(n):
            if g == f:
                continue
            Z = Zm[(f, g)]
            for j in range(len(opts[f])):
                nz = [X(g, k) for k in np.nonzero((Z[j] == 1) | (Z[j] == 6))[0]]
                zr = [X(g, k) for k in np.nonzero(Z[j] == 0)[0]]
                cnf.append([-X(f, j), -T(f, g)] + nz)
                cnf.append([-X(f, j), T(f, g)] + zr)
    D2 = lambda f, g: pool.id(('d2', min(f, g), max(f, g)))
    for f in range(n):
        for g in range(f + 1, n):
            cnf.append([-Dv(f, g), -T(f, g)])
            cnf.append([-Dv(f, g), D2(f, g)])
            cnf.append([Dv(f, g), -D2(f, g)])
    for f in range(n):
        dl = [Dv(f, g) for g in range(n) if g != f]
        cnf.extend(CardEnc.atmost(lits=dl, bound=1, vpool=pool, encoding=EncType.seqcounter).clauses)
        for j, (e, e2) in enumerate(opts[f]):
            if shadow_info(e, e2)[1] is None:
                cnf.append([-X(f, j)] + dl)
            else:
                for dv in dl:
                    cnf.append([-X(f, j), -dv])
    supp = [set(a // 2 for a in FAR[f][0]) for f in range(n)]
    for f in range(n):
        for zb in range(6):
            thr = [g for g in range(n) if g != f and zb in supp[g]]
            lits = [T(f, g) for g in thr] + [Dv(f, g) for g in thr] + [D2(f, g) for g in thr]
            out = exact_counter(lits, pool, cnf)
            rhs = [out[k] for k in range(1, len(lits) + 1)]   # rhs[k-1] <=> sum >= k (both directions)
            base = 2 if zb in supp[f] else 4
            for j, (e, e2) in enumerate(opts[f]):
                s, lc = shadow_info(e, e2)
                target = base - sum(1 for l in s if l // 2 == zb) - (2 if (lc is not None and lc // 2 == zb) else 0)
                if target < 0 or target > len(lits):
                    cnf.append([-X(f, j)])
                    continue
                if target >= 1:
                    cnf.append([-X(f, j), rhs[target - 1]])
                if target < len(lits):
                    cnf.append([-X(f, j), -rhs[target]])
    return cnf, pool


if __name__ == '__main__':
    idx = int(sys.argv[1]); out = sys.argv[2]
    maxopts = int(sys.argv[sys.argv.index('--maxopts') + 1]) if '--maxopts' in sys.argv else None
    cls = [c for c in classes_t_ge(0) if c[2] == 0]
    cls = [c for c in cls if split(*ball_disc(c[0], c[1]))]
    tau, eta, t, osz = cls[idx]
    opts, Xm, Zm, Pp = build(tau, eta, maxopts)
    if any(len(o) == 0 for o in opts):
        print('dead-unary: no CNF needed'); sys.exit(0)
    cnf, pool = export(tau, eta, opts, Zm)
    cnf.to_file(out)
    json.dump({'class_index': idx, 'tau': list(tau), 'eta': list(eta), 'maxopts': maxopts,
               'x_vars': {'%d,%d' % (f, j): pool.id(('x', f, j)) for f in range(len(opts)) for j in range(len(opts[f]))},
               'options': [[list(map(list, o)) for o in os_] for os_ in opts]}, open(out + '.map.json', 'w'))
    print('CNF: %d vars, %d clauses -> %s' % (pool.top, len(cnf.clauses), out), flush=True)
    if '--solve' in sys.argv:
        from pysat.solvers import Solver
        with Solver(name='cd15', bootstrap_with=cnf.clauses) as s:
            print('pysat CaDiCaL:', 'SAT' if s.solve() else 'UNSAT', flush=True)
