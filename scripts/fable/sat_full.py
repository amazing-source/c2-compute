"""Whole-graph SAT model for C1's family (B) = ALL-DIS (structures 27/09, requested by C1).
Vertices: x0 = 0; interior world vertex w (block w // 2, sign w & 1) -> 1 + w (14); roots = pairs of interior vertices of
different blocks (84) -> 15 + index.  sigma = (w -> w ^ 1).  Fixed: x0 ~ interior; w ~ w ^ 1; interior a ~ root r iff a in r.
Variables: adjacency of two roots (one per sigma-class; {u, sigma u} never adjacent).
Constraints (all exact): every root has degree 14 (12 roots); lambda/mu for every (root, root) and (interior, root) pair;
ALL-DIS: for every root u and every root z whose pair meets the pair of u (sister roots included), not(u ~ z and sigma u ~ z)
(the D-partner of every orbit lies on a disjoint pair).  Symmetry: the type tau0 of block 0 is FIXED (adjacency of the block-0
roots from (mu, h) of tau0, c-frame label l = 2q + e <-> world vertex 2(q+1) + e); the other blocks are free.
Optional: the block-0 local system of sat_local2 (--local0, redundant but may help), SEC-dis units at block 0 (implied).
Usage: python sat_whole.py types.json id [seconds] [--local0] [--dimacs]"""
import itertools, json, sys, time, threading
from pysat.formula import IDPool, CNF
from pysat.card import CardEnc, EncType

M = 7
ALLDIS = '--alldis' in sys.argv
SOLVER = 'cadical195'
INT = list(range(2 * M))
ROOTS = [r for r in itertools.combinations(INT, 2) if r[0] // 2 != r[1] // 2]
RID = {r: i for i, r in enumerate(ROOTS)}
sg = lambda r: tuple(sorted((r[0] ^ 1, r[1] ^ 1)))


class Model:
    def __init__(self, mu, tau, local0=False):
        self.pool = IDPool(); self.cnf = CNF(); self.memo = {}
        self.mu, self.tau = mu, tau
        self.h = [tau[l] ^ 1 for l in range(12)]
        self.fixed = {}                              # canonical root pair -> bool (block-0 type)
        W = lambda l: 2 * (l // 2 + 1) + (l & 1)     # c-frame label -> world vertex
        roots0 = [r for r in ROOTS if r[0] // 2 == 0]
        want = set()
        for l in range(12):
            for a, b in (((0, W(l)), (0, W(mu[l]))), ((0, W(l)), (1, W(self.h[l])))):
                X, Y = tuple(sorted(a)), tuple(sorted(b))
                want |= {self.key(X, Y), self.key(sg(X), sg(Y))}
        for X, Y in itertools.combinations(roots0, 2):
            self.fixed[self.key(X, Y)] = self.key(X, Y) in want
        self.local0 = local0

    @staticmethod
    def key(X, Y):
        return min(tuple(sorted((X, Y))), tuple(sorted((sg(X), sg(Y)))))

    def A(self, X, Y):
        """literal / constant for 'root X ~ root Y'"""
        if X == Y:
            return False
        if Y == sg(X):
            return False
        k = self.key(X, Y)
        if k in self.fixed:
            return self.fixed[k]
        return self.pool.id(('A',) + k)

    def AND(self, a, b):
        if a is False or b is False:
            return False
        if a is True:
            return b
        if b is True:
            return a
        k = ('and', min(a, b), max(a, b))
        if k not in self.memo:
            v = self.pool.id(k); self.memo[k] = v
            self.cnf.extend([[-v, a], [-v, b], [v, -a, -b]])
        return self.memo[k]

    def exactly(self, lits, k, tag):
        rest, t = [], 0
        for l in lits:
            if l is True:
                t += 1
            elif l is not False:
                rest.append(l)
        k -= t
        if k < 0 or k > len(rest):
            print('static contradiction', tag); self.cnf.append([]); return
        if not rest:
            return
        assert len(set(rest)) == len(rest), tag
        if k == 0:
            self.cnf.extend([[-l] for l in rest])
        elif k == len(rest):
            self.cnf.extend([[l] for l in rest])
        else:
            self.cnf.extend(CardEnc.equals(lits=rest, bound=k, vpool=self.pool, encoding=EncType.seqcounter).clauses)

    def build(self):
        seen = set(); n = {'degree': 0, 'rootroot': 0, 'introot': 0, 'alldis': 0}
        for u in ROOTS:                                  # degree: 12 root neighbours
            if sg(u) < u:
                continue
            self.exactly([self.A(u, z) for z in ROOTS if z != u], 12, ('deg', u)); n['degree'] += 1
        for u, z in itertools.combinations(ROOTS, 2):    # lambda/mu for root pairs, one per sigma-class
            k = self.key(u, z)
            if k in seen:
                continue
            seen.add(k)
            terms = [self.AND(self.A(u, w), self.A(z, w)) for w in ROOTS if w != u and w != z]
            if z == sg(u):
                # common neighbours of u and sigma u come in sigma-pairs {w, sigma w} carrying the SAME literal:
                # mu = 2 means exactly one such pair (Lemma 1: the D-square)
                self.exactly(sorted({t for t in terms if not isinstance(t, bool)}) + [t for t in terms if t is True][:1], 1, ('rr-sigma', u))
            else:
                self.exactly(terms + [self.A(u, z)], 2 - len(set(u) & set(z)), ('rr', u, z))
            n['rootroot'] += 1
        seen = set()
        for a in INT:                                    # lambda/mu for (interior a, root z)
            for z in ROOTS:
                k = min((a, z), (a ^ 1, sg(z)))
                if k in seen:
                    continue
                seen.add(k)
                terms = [self.A(w, z) for w in ROOTS if a in w and w != z]
                self.exactly(terms, 1 if a in z else 2 - ((a ^ 1) in z), ('ir', a, z)); n['introot'] += 1
        for u in (ROOTS if ALLDIS else []):             # ALL-DIS (optional): common neighbours of u and sigma u avoid the blocks of u
            su = sg(u); bu = {u[0] // 2, u[1] // 2}
            for z in ROOTS:
                if z in (u, su) or not ({z[0] // 2, z[1] // 2} & bu):
                    continue
                a, b = self.A(u, z), self.A(su, z)
                if a is False or b is False:
                    continue
                cl = [x for x in ((-a if not isinstance(a, bool) else None), (-b if not isinstance(b, bool) else None)) if x is not None]
                if a is True and b is True:
                    print('static all-dis contradiction', u, z); self.cnf.append([])
                    continue
                self.cnf.append(cl); n['alldis'] += 1
        return n


if __name__ == '__main__':
    T = json.load(open(sys.argv[1])); tid = int(sys.argv[2])
    rec = next(t for t in T if t.get('id', None) == tid) if isinstance(T[0], dict) and 'id' in T[0] else T[tid]
    secs = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 200
    t0 = time.time()
    Mo = Model(rec['mu'], rec['tau'])
    cnt = Mo.build()
    print('type %d (pivots %s): vars %d, clauses %d, families %s, build %.1fs' % (tid, rec.get('pivots'), Mo.pool.top,
          len(Mo.cnf.clauses), cnt, time.time() - t0), flush=True)
    if '--dimacs' in sys.argv:
        out = 'whole_%s_%d.cnf' % ('alldis' if ALLDIS else 'full', tid)
        Mo.cnf.to_file(out, comments=['c sat_whole.py (structures): whole graph srg(99,14,1,2) with sigma, ALL-DIS family (B), '
                                      'block-0 type fixed = %s' % json.dumps({'mu': rec['mu'], 'tau': rec['tau']})])
        print('DIMACS', out); sys.exit(0)
    from pysat.solvers import Solver
    print('solver', SOLVER, 'alldis', ALLDIS, flush=True)
    with Solver(name=SOLVER, bootstrap_with=Mo.cnf.clauses) as s:
        timer = threading.Timer(secs, lambda: s.interrupt()); timer.start()
        t1 = time.time(); res = s.solve_limited(expect_interrupt=True); timer.cancel()
    print('status', {True: 'SAT', False: 'UNSAT', None: 'UNKNOWN (interrupted)'}[res], '%.1fs' % (time.time() - t1), flush=True)
