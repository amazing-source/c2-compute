"""Second SAT encoding of the local system at a block (any type (mu, tau), C2 notation), built around the iota-pairing:
  x[z,e]  : Phi(z) = e (bijection far roots -> non-mu pairs)                        (i)
  U[z,b]  : b in Phi(z); (ii) as cardinality over U (10 literals each)               (ii)
  q[{e,f}]: iota pairs e with f (Phi sigma z = f when Phi z = e); perfect matching on the non-mu pairs;
            (iii) as cardinality over q (cross counts; diagonal = secant pairs)      (iii)
  linking : q[{e,f}] & x[z,e] -> x[sigma z, f]   and   x[z,e] & x[sigma z, f] -> q[{e,f}]   (ternary clauses)
  pointwise (C1): L1, S1, S2 on (x, U); full block extras (no cancellation, (iv), (v)) if cfg has "cycles".
Same logical content as sat_local.py (UNSAT here <=> UNSAT there); only the encoding of (iii) differs.
Usage: python sat_local2.py cfg.json seconds [--dimacs] [--solver=name]"""
import itertools, json, sys, time, threading
from pysat.formula import IDPool, CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
import iota_design as D

L = list(range(12)); PI = [v ^ 1 for v in L]


def encode(mu, tau, cycles=()):
    K2, K3 = D.K2K3(mu, tau)
    far = [z for z in itertools.combinations(L, 2) if z[0] // 2 != z[1] // 2]
    fidx = {z: i for i, z in enumerate(far)}
    sig = [fidx[tuple(sorted((PI[z[0]], PI[z[1]])))] for z in far]
    edges = [e for e in itertools.combinations(L, 2) if mu[e[0]] != e[1]]
    E = len(edges)
    pool = IDPool(); cnf = CNF()
    X = lambda z, e: pool.id(('x', z, e))
    U = lambda z, b: pool.id(('u', z, b))

    def eq(lits, k):
        if k < 0 or k > len(lits):
            cnf.append([]); return
        cnf.extend(CardEnc.equals(lits=lits, bound=k, vpool=pool, encoding=EncType.seqcounter).clauses)
    for z in range(60):
        eq([X(z, e) for e in range(E)], 1)
    for e in range(E):
        eq([X(z, e) for z in range(60)], 1)
    for z in range(60):
        for b in L:
            es = [e for e in range(E) if b in edges[e]]
            cnf.append([-U(z, b)] + [X(z, e) for e in es])
            for e in es:
                cnf.append([-X(z, e), U(z, b)])
    for a in L:                                                        # (ii)
        for b in L:
            eq([U(z, b) for z in range(60) if a in far[z]], int(K2[a, b]))
    # iota-pair variables: allowed iff disjoint, or secant at a label b with K3[b,b] = 2
    Q = {}
    for i in range(E):
        for j in range(i + 1, E):
            com = set(edges[i]) & set(edges[j])
            if any(K3[b, b] < 2 for b in com):
                continue
            Q[i, j] = pool.id(('q', i, j))
    qv = lambda i, j: Q.get((min(i, j), max(i, j)))
    for i in range(E):
        eq([qv(i, j) for j in range(E) if j != i and qv(i, j)], 1)
    acc = {}
    for (i, j), v in Q.items():
        for b in edges[i]:
            for d in edges[j]:
                acc.setdefault((min(b, d), max(b, d)), []).append(v)
    for b in L:                                                        # (iii) through iota
        for d in L:
            if d < b:
                continue
            eq(acc.get((b, d), []), int(K3[b, d]) if b != d else int(K3[b, b]) // 2)
    for z in range(60):                                                # linking x <-> q
        sz = sig[z]
        for e in range(E):
            for f in range(E):
                if f == e:
                    continue
                v = qv(e, f)
                if v is None:
                    cnf.append([-X(z, e), -X(sz, f)])                  # forbidden pairing
                else:
                    cnf.append([-v, -X(z, e), X(sz, f)])
                    cnf.append([-X(z, e), -X(sz, f), v])
    for z in range(60):                                                # pointwise (C1)
        sz = sig[z]
        for a in far[z]:
            for aa in (a, PI[a]):
                cnf.append([-U(z, aa), -U(sz, PI[aa])])
            cnf.append([-U(z, a), -U(sz, tau[a])])
            cnf.append([U(z, a), -U(z, mu[a]), -U(sz, tau[a])])
    if cycles:
        for z in range(60):
            sz = sig[z]
            for b in L:
                cnf.append([-U(z, b), -U(sz, b)])
            for C in cycles:
                lc = [2 * p + s for p in C for s in (0, 1)]
                need = len(C) - len({a // 2 for a in far[z]} & set(C))
                eq([U(z, v) for v in lc] + [U(sz, v) for v in lc], need)
    return cnf, pool, far, edges


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    sname = next((a.split('=')[1] for a in sys.argv if a.startswith('--solver=')), 'cadical195')
    if 'cycles' in cfg:
        from fullblock_rigid import mates_from_D
        cycles = [tuple(c) for c in cfg['cycles']]
        m = mates_from_D(cycles); mu = tau = m
    else:
        cycles = []; mu, tau = cfg['mu'], cfg['tau']
    t0 = time.time()
    cnf, pool, far, edges = encode(mu, tau, cycles)
    print(sys.argv[1], '[encoding 2: iota-pairs]', 'vars', pool.top, 'clauses', len(cnf.clauses), 'encode %.1fs' % (time.time() - t0), flush=True)
    if '--dimacs' in sys.argv:
        out = sys.argv[1].replace('.json', '_enc2.cnf')
        cnf.to_file(out, comments=['c local system, encoding 2 (iota-pairs), ' + json.dumps(cfg)])
        print('DIMACS written to', out); sys.exit(0)
    with Solver(name=sname, bootstrap_with=cnf.clauses) as s:
        timer = threading.Timer(secs, lambda: s.interrupt()); timer.start()
        res = s.solve_limited(expect_interrupt=True); timer.cancel()
        print('status', {True: 'SAT', False: 'UNSAT', None: 'UNKNOWN (interrupted)'}[res], '%.1fs' % (time.time() - t0), flush=True)
        if res:
            model = set(l for l in s.get_model() if l > 0)
            phi = {str(list(far[z])): list(edges[e]) for z in range(60) for e in range(len(edges)) if pool.id(('x', z, e)) in model}
            json.dump({'cfg': cfg, 'Phi': phi}, open(sys.argv[1].replace('.json', '_satsol2.json'), 'w'), indent=1)
