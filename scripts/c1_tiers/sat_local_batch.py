"""Batch SAT campaign on block types (C1, 27/09): structures' sat_local.encode (local system (i)-(iii) + pointwise L1/S1/S2)
+ lex-leader symmetry breaking under Stab_B6(mu, tau), CaDiCaL with a time cap, or DIMACS export for DRAT.
Input: a JSON list of records {mu, tau, ...} (e.g. data/pivot_types.json filtered), selected by index.
Usage: python sat_local_batch.py TYPES.json I0 I1 SECONDS [--prefix k] [--dimacs OUTDIR]
Output lines: index, profile, |Stab|, vars, clauses, status (UNSAT / SAT / UNKNOWN)."""
import sys, json, time, threading, itertools, os
HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.environ.get('SAT_LOCAL_DIR', r'C:/home/researcher/Desktop/c99_structures_lab/involution')
sys.path.insert(0, SL); sys.path.insert(0, HERE)
from sat_local import encode
from fullblock_symmetry import lexleader_clauses
from pivot_types import stabiliser
from pysat.solvers import Solver


def run_one(rec, secs, prefix, dimacs_dir=None, idx=0):
    mu, tau = rec['mu'], rec['tau']
    t0 = time.time()
    cnf, pool, far, edges = encode(mu, tau, ())
    G = stabiliser(mu, tau)
    X = lambda z, e: pool.id(('x', z, e))
    cnt = itertools.count()
    if len(G) > 1:
        cnf.extend(lexleader_clauses(mu, X, lambda: pool.id(('lex', next(cnt))), prefix_roots=prefix, G=G))
    if dimacs_dir:
        os.makedirs(dimacs_dir, exist_ok=True)
        path = os.path.join(dimacs_dir, 'type_%04d.cnf' % idx)
        cnf.to_file(path, comments=['c C1 sat_local_batch type %d mu %s tau %s |Stab| %d prefix %d' % (idx, mu, tau, len(G), prefix)])
        return 'DIMACS ' + path, len(G), pool.top, len(cnf.clauses), time.time() - t0
    with Solver(name='cadical195', bootstrap_with=cnf.clauses) as s:
        timer = threading.Timer(secs, lambda: s.interrupt()); timer.start()
        res = s.solve_limited(expect_interrupt=True); timer.cancel()
    st = {True: 'SAT', False: 'UNSAT', None: 'UNKNOWN'}[res]
    return st, len(G), pool.top, len(cnf.clauses), time.time() - t0


if __name__ == '__main__':
    types = json.load(open(sys.argv[1])); i0, i1 = int(sys.argv[2]), int(sys.argv[3]); secs = float(sys.argv[4])
    prefix = int(sys.argv[sys.argv.index('--prefix') + 1]) if '--prefix' in sys.argv else 10
    dd = sys.argv[sys.argv.index('--dimacs') + 1] if '--dimacs' in sys.argv else None
    for i in range(i0, min(i1, len(types))):
        st, g, nv, nc, dt = run_one(types[i], secs, prefix, dd, i)
        print('%d %s |Stab|=%d vars=%d clauses=%d %s (%.0fs)' % (i, types[i].get('anch_tw'), g, nv, nc, st, dt), flush=True)
