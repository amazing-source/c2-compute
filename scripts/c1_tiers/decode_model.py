"""C2: decode a CaDiCaL model ('v' lines) of a sat_local_batch.py CNF into a local solution {cfg, Phi}, with the same map as
structures' sat_local.py (Phi[str(far z)] = edge e for every true x(z, e)), after checking that the model satisfies every clause
of the exported CNF.   Usage: python decode_model.py TYPES.json IDX CADICAL_OUT CNF OUT.json"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('SAT_LOCAL_DIR', os.path.join(HERE, '..', 'structures'))); sys.path.insert(0, HERE)
from sat_local import encode
types, idx, cout, cnfpath, outpath = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
val = {}
for line in open(cout):
    if line.startswith('v '):
        for x in line.split()[1:]:
            x = int(x)
            if x: val[abs(x)] = x > 0
bad = 0; ncl = 0
for line in open(cnfpath):
    if not line.strip() or line[0] in 'cp': continue
    lits = [int(x) for x in line.split()[:-1]]; ncl += 1
    if not any(val.get(abs(l), False) == (l > 0) for l in lits): bad += 1
rec = types[idx]; cnf, pool, far, edges = encode(rec['mu'], rec['tau'], ())
phi = {str(list(far[z])): list(edges[e]) for z in range(60) for e in range(60) if val.get(pool.id(('x', z, e)), False)}
json.dump({'cfg': {'mu': rec['mu'], 'tau': rec['tau'], 'types_file': os.path.basename(sys.argv[1]), 'index': idx},
           'Phi': phi, 'model_check': {'clauses': ncl, 'violated': bad}}, open(outpath, 'w'), indent=1)
print('decoded: %d clauses, %d violated, |Phi| = %d' % (ncl, bad, len(phi)))
