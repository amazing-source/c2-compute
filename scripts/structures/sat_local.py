"""SAT encoding (CaDiCaL via pysat) of the local system at a block c:
  (i) Phi bijection far roots -> non-mu pairs;  (ii) K2 counts;  (iii) K3 counts;  full block extras (iv), (v), no cancellation.
Variables x[z,e]; u[z,b] <-> OR_{e containing b} x[z,e]; w[Z,b,d] <-> u[z,b] & u[sigma z,d].
Cardinality constraints: pysat.card (totalizer / seqcounter).  UNSAT => no block of this type in an involution.
Usage: python sat_local.py cfg.json seconds [solver]"""
import itertools, json, sys, time, threading
from pysat.formula import IDPool, CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
import iota_design as D

L = list(range(12))
PI = [v ^ 1 for v in L]


def encode(mu, tau, cycles=()):
    K2, K3 = D.K2K3(mu, tau)
    far = [z for z in itertools.combinations(L, 2) if z[0] // 2 != z[1] // 2]
    fidx = {z: i for i, z in enumerate(far)}
    sig = [fidx[tuple(sorted((PI[z[0]], PI[z[1]])))] for z in far]
    edges = [e for e in itertools.combinations(L, 2) if mu[e[0]] != e[1]]
    pool = IDPool()
    X = lambda z, e: pool.id(('x', z, e))
    U = lambda z, b: pool.id(('u', z, b))
    Wv = lambda z, b, d: pool.id(('w', z, b, d))
    cnf = CNF()

    def eq(lits, k):
        if k < 0 or k > len(lits):
            cnf.append([])            # impossible
            return
        enc = CardEnc.equals(lits=lits, bound=k, vpool=pool, encoding=EncType.seqcounter)
        cnf.extend(enc.clauses)
    for z in range(60):
        eq([X(z, e) for e in range(60)], 1)
    for e in range(60):
        eq([X(z, e) for z in range(60)], 1)
    for z in range(60):
        for b in L:
            es = [e for e in range(60) if b in edges[e]]
            cnf.append([-U(z, b)] + [X(z, e) for e in es])
            for e in es:
                cnf.append([-X(z, e), U(z, b)])
    for a in L:                                               # (ii)
        for b in L:
            eq([U(z, b) for z in range(60) if a in far[z]], int(K2[a, b]))
    reps = [z for z in range(60) if z < sig[z]]
    for z in reps:                                            # w = u(z,b) & u(sz,d)
        for b in L:
            for d in L:
                w = Wv(z, b, d)
                cnf.append([-w, U(z, b)]); cnf.append([-w, U(sig[z], d)])
                cnf.append([w, -U(z, b), -U(sig[z], d)])
    for b in L:                                               # (iii)
        for d in L:
            eq([Wv(z, b, d) for z in reps] + [Wv(z, d, b) for z in reps], int(K3[b, d])) if b != d else \
                eq([Wv(z, b, b) for z in reps], int(K3[b, b]) // 2)
    # pointwise conditions (C1, 27/09; re-derived from lambda = 1, mu = 2; valid at every block type), for z and for sigma z:
    #  (L1) a in blocks(z): not (a in Phi z and pi a in Phi sigma z)
    #  (S1) l in z: l in Phi z  => eta l not in Phi sigma z          (eta = tau in C2 notation = pi o h)
    #  (S2) l in z: l not in Phi z => not (mu l in Phi z and eta l in Phi sigma z)
    for z in range(60):
        sz = sig[z]
        for a in far[z]:
            for aa in (a, PI[a]):
                cnf.append([-U(z, aa), -U(sz, PI[aa])])
            l = a
            cnf.append([-U(z, l), -U(sz, tau[l])])
            cnf.append([U(z, l), -U(z, mu[l]), -U(sz, tau[l])])
    if cycles:                                                # full block extras
        for z in range(60):
            sz = sig[z]
            for b in L:
                cnf.append([-U(z, b), -U(sz, b)])             # no cancellation
            for a in far[z]:
                cnf.append([-U(z, a), -U(sz, PI[a])])        # (iv)
                cnf.append([-U(z, PI[a]), -U(sz, a)])
            for C in cycles:                                  # (v)
                lc = [2 * p + s for p in C for s in (0, 1)]
                need = len(C) - len({a // 2 for a in far[z]} & set(C))
                eq([U(z, v) for v in lc] + [U(sz, v) for v in lc], need)
    return cnf, pool, far, edges


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    sname = sys.argv[3] if len(sys.argv) > 3 else 'cadical195'
    if 'cycles' in cfg:
        from fullblock_rigid import mates_from_D
        cycles = [tuple(c) for c in cfg['cycles']]
        m = mates_from_D(cycles); mu = tau = m
    else:
        cycles = []; mu, tau = cfg['mu'], cfg['tau']
    t0 = time.time()
    cnf, pool, far, edges = encode(mu, tau, cycles)
    print(sys.argv[1], 'vars', pool.top, 'clauses', len(cnf.clauses), 'encode %.1fs' % (time.time() - t0), flush=True)
    if '--dimacs' in sys.argv:                 # for an external solver with a DRAT proof: cadical in.cnf proof.drat; drat-trim
        out = sys.argv[1].replace('.json', '.cnf')
        cnf.to_file(out, comments=['c local system (i)-(iii) + full-block extras + C1 pointwise (L1,S1,S2)', 'c cfg ' + json.dumps(cfg)])
        print('DIMACS written to', out, flush=True)
        sys.exit(0)
    with Solver(name=sname, bootstrap_with=cnf.clauses) as s:
        timer = threading.Timer(secs, lambda: s.interrupt())
        timer.start()
        res = s.solve_limited(expect_interrupt=True)
        timer.cancel()
        st = {True: 'SAT', False: 'UNSAT', None: 'UNKNOWN (interrupted)'}[res]
        print('status', st, '%.1fs' % (time.time() - t0), flush=True)
        if res:
            model = set(l for l in s.get_model() if l > 0)
            phi = {str(list(far[z])): list(edges[e]) for z in range(60) for e in range(60) if pool.id(('x', z, e)) in model}
            json.dump({'cfg': cfg, 'Phi': phi}, open(sys.argv[1].replace('.json', '_satsol.json'), 'w'), indent=1)
