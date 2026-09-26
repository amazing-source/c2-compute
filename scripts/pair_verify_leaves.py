"""Re-check of the per-leaf pair-level certificates of pair_fast.py (note 43), with structures' OWN model and checker:
for every dead leaf of logs/pair_certs/leaf_<ci>_<job>*.jsonl.gz
  1. its skeleton is rebuilt here from the leaf's triples (canonical order, coded independently of pair_fast/pair_blockdfs):
     orbits in lexicographic pair order, two per pair; anchored pair {c,x} (x in In[c]): both pivot at c; twisted pair {a<b}: first
     orbit pivots at a, second at b; at every block c, the orbits of {c,x} pivoting at c take the thirds y of the triples (c,x,y)
     in increasing order, and the t-th orbit of {c,x} with third y is paired with the t-th orbit of {c,y} with third x (WLOG:
     orbits of a pair pivoting at the same block are interchangeable); the result is checked by valid_skeleton (pair_verify.py);
  2. general_model.build (external_structures/ad) rebuilds (A, lo, hi, lb, ub) and ad/certify.check verifies the tree (exact
     rationals), the sparse certificates being expanded to general_model's dense row list.
Also: every leaf of the job has a record; the leaves not dead are listed (they go to the orbit stage).
Usage: python pair_verify_leaves.py <ci> <job> [<ci> <job> ...]      env SAMPLE=k: check only every k-th dead leaf"""
import sys, os, json, gzip, glob, time
from fractions import Fraction
sys.path.insert(0, os.environ.get('STRUCT_AD', (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad') if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad')) else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor_structures'))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general_model import build
from certify import check
from pair_verify import valid_skeleton, leaves_of
N = 7


def canonical_skeleton(In, leaf):
    head = {}
    for c, xs in In.items():
        for x in xs: head[(min(c, x), max(c, x))] = c
    orbs = []
    for a in range(N):
        for b in range(a + 1, N):
            pv = (head[(a, b)],) * 2 if (a, b) in head else (a, b)
            orbs += [{'pair': (a, b), 'pivot': pv[0], 'partner': None}, {'pair': (a, b), 'pivot': pv[1], 'partner': None}]
    for c in range(N):
        trip = sorted(z for z in leaf if z[0] == c)
        if not trip: continue
        slot = {}
        for x in sorted({z[1] for z in trip}):
            mine = [i for i, o in enumerate(orbs) if o['pivot'] == c and c in o['pair'] and x in o['pair']]
            for i, y in zip(mine, sorted(z[2] for z in trip if z[1] == x)): slot.setdefault((x, y), []).append(i)
        for (x, y), lst in slot.items():
            for i, j in zip(lst, slot[(y, x)]): orbs[i]['partner'] = j
    return orbs


def dense_tree(T, m):
    if 'cert_sparse' in T:
        y = [Fraction(0)] * m
        for r, v in T['cert_sparse']: y[r] = Fraction(v, T['den'])
        return {'cert': [str(v) for v in y]}
    return {'var': T['var'], 'le': T['le'], 'lo': dense_tree(T['lo'], m), 'hi': dense_tree(T['hi'], m)}


if __name__ == "__main__":
    allok = True; step = int(os.environ.get('SAMPLE', 1))
    for ci, job in zip(sys.argv[1::2], map(int, sys.argv[2::2])):
        t0 = time.time(); po = leaves_of(ci, job); In = {int(c): v for c, v in po["In"].items()}; leaves = po["leaves"]
        recs = {}
        for fn in glob.glob("logs/pair_certs/leaf_%s_%d.jsonl.gz" % (ci, job)) + glob.glob("logs/pair_certs/leaf_%s_%d_*_*.jsonl.gz" % (ci, job)):
            for line in gzip.open(fn, "rt"):
                r = json.loads(line); recs[r['leaf']] = r
        dead = sorted(t for t, r in recs.items() if r['status'] == 'dead'); ok = bad = 0
        for t in dead[::step]:
            leaf = [tuple(z) for z in leaves[t]]
            orbs = canonical_skeleton(In, leaf)
            good = valid_skeleton(orbs, In, {c: [z for z in leaf if z[0] == c] for c in range(N) if any(z[0] == c for z in leaf)})
            if good:
                A_, lo, hi, lb, ub = build(orbs)[:5]
                good = check(A_, lo, hi, lb, ub, dense_tree(recs[t]['tree'], A_.shape[0]))
            ok += good; bad += not good
            if not good: print("  INVALID leaf", t)
        missing = [t for t in range(len(leaves)) if t not in recs]
        notdead = [t for t, r in recs.items() if r['status'] != 'dead']
        res = bad == 0; allok &= res
        print("%s job %d: leaves %d records %d dead %d (checked %d: valid %d INVALID %d) not dead %d missing %d  %s %.0fs" % (ci, job,
              len(leaves), len(recs), len(dead), ok + bad, ok, bad, len(notdead), len(missing), "OK" if res else "FAIL", time.time() - t0))
        sys.stdout.flush()
    print("ALL OK" if allok else "SOME FAIL")
