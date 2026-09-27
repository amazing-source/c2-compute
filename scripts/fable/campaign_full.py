"""
Campaign runner: full exact model (sat_full.py, both stages, no all-dis) over a list of block types,
one solver process per type with a time budget; records SAT/UNSAT/UNKNOWN in a JSON ledger.
UNSAT types are re-run with proof logging (pysat cadical195 with_proof) and the DRAT proof is written next to the
DIMACS export for independent checking with drat-trim.
Usage: python campaign_full.py types_all.json out.json [seconds_per_type] [start] [end] [workers]
"""
import sys, json, os, time, subprocess, itertools
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(args):
    idx, rec, secs = args
    from pysat.solvers import Solver
    import importlib.util
    spec = importlib.util.spec_from_file_location('sat_full', os.path.join(HERE, 'sat_full.py'))
    sf = importlib.util.module_from_spec(spec)
    sys.argv = ['sat_full.py']            # ALLDIS False
    spec.loader.exec_module(sf)
    t0 = time.time()
    Mo = sf.Model(rec['mu'], rec['tau'])
    cnt = Mo.build()
    if any(len(c) == 0 for c in Mo.cnf.clauses):
        return dict(idx=idx, pivots=rec['pivots'], status='UNSAT-static', time=time.time() - t0)
    import threading
    with Solver(name='cadical195', bootstrap_with=Mo.cnf.clauses) as s:
        timer = threading.Timer(secs, lambda: s.interrupt()); timer.start()
        res = s.solve_limited(expect_interrupt=True); timer.cancel()
    dt = time.time() - t0
    status = {True: 'SAT', False: 'UNSAT', None: 'UNKNOWN'}[res]
    out = dict(idx=idx, pivots=rec['pivots'], status=status, time=dt, vars=Mo.pool.top, clauses=len(Mo.cnf.clauses))
    if res is False:
        # re-run with proof logging to produce a DRAT certificate
        cnf_path = os.path.join(HERE, 'certs', 'full_type%d.cnf' % idx)
        os.makedirs(os.path.dirname(cnf_path), exist_ok=True)
        Mo.cnf.to_file(cnf_path)
        try:
            with Solver(name='cadical195', bootstrap_with=Mo.cnf.clauses, with_proof=True) as s2:
                r2 = s2.solve()
                proof = s2.get_proof()
            with open(cnf_path.replace('.cnf', '.drat'), 'w') as f:
                f.write('\n'.join(proof) + '\n')
            out['proof'] = cnf_path.replace('.cnf', '.drat'); out['proof_lines'] = len(proof); out['recheck'] = (r2 is False)
        except Exception as e:
            out['proof_error'] = repr(e)[:200]
    if res is True:
        out['model'] = [l for l in s.get_model() if l > 0][:50] if False else 'not stored'
    return out


if __name__ == '__main__':
    types = json.load(open(sys.argv[1]))
    outp = sys.argv[2]
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 600
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = int(sys.argv[5]) if len(sys.argv) > 5 else len(types)
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    ledger = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [(i, types[i], secs) for i in range(start, end) if types[i].get('valid', True) and str(i) not in ledger]
    print('types to run', len(todo), 'workers', workers, 'budget', secs, flush=True)
    with Pool(workers) as pool:
        for res in pool.imap_unordered(run_one, todo):
            ledger[str(res['idx'])] = res
            json.dump(ledger, open(outp, 'w'))
            print(json.dumps(res), flush=True)
