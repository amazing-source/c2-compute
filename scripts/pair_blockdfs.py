"""Block-fixing DFS at the PAIR level (note 43): the model is the pair-level model of 'structures' (conway99-structures-lab,
ad/general_model.py, reviewed sound in note 41), whose orbits may have an UNKNOWN D-partner ('partner': None: the partner is a
free matching variable among the orbits with the same pivot, balance only on certain classes).  A node = partial assignment
{block: option}; the D-structure is fixed at the assigned blocks and unknown at every other block (A included).  Since the model
at a node is a relaxation of every leaf extending it, a dead node (exact certificate: Farkas at LP level or a branch tree,
ad/certify.py) kills all of them.  Alive/unknown node => fix one more block (fewest options > 1 among the node's leaves), children
= its options (complete cover of the node's leaves).
Certificates: logs/pair_certs/<ci>_<job>.jsonl, one line per dead node: spec (In, twisted edges, A, assign) + tree; re-checked by
scripts/pair_verify.py (rebuilds the skeleton from the spec on its own, then ad/certify.check).
Usage: python pair_blockdfs.py <class id> <job> [max nodes]      (structures' lab cloned in external_structures/, gitignored; env STRUCT_AD overrides)"""
import sys, json, glob, time, os
from collections import Counter
sys.path.insert(0, os.environ.get('STRUCT_AD', (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad') if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad')) else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor_structures'))))
from general_model import build
from certify import certify
N = 7


def load_job(ci, job):
    jobs = []
    for f in sorted(glob.glob("logs/mix6_cls_%s_S_*.json" % ci)):
        d = json.load(open(f))
        for po in d["per_orient"]:
            if po["leaves"]: jobs.append(po)
    return jobs[job]


def orbit_list(In, assign):
    """42 orbits {pair, pivot, partner}: pivots from the orientation (anchored pair {c,x}: both orbits pivot at c; twisted pair:
    one orbit per end); partners from the triples (c,x,y) of the assigned blocks, None at the other blocks."""
    anch = {frozenset((c, x)): c for c, xs in In.items() for x in xs}
    orbs = []
    for a in range(N):
        for b in range(a + 1, N):
            c = anch.get(frozenset((a, b)))
            for piv in ((c, c) if c is not None else (a, b)):
                orbs.append({'pair': (a, b), 'pivot': piv, 'partner': None})
    for c, trip in assign.items():
        slots = {}                                  # (x, y) -> orbit indices of pair {c,x} pivoting at c, labelled by the thirds
        for x in sorted({t[1] for t in trip}):
            os_ = [i for i, o in enumerate(orbs) if o['pivot'] == c and set(o['pair']) == {c, x}]
            ys = sorted(t[2] for t in trip if t[1] == x)
            assert len(os_) == len(ys), (c, x, os_, ys)
            for i, y in zip(os_, ys): slots.setdefault((x, y), []).append(i)
        for (x, y), Zs in slots.items():
            for t_, i in enumerate(Zs): orbs[i]['partner'] = slots[(y, x)][t_]
        assert all(o['partner'] is not None for o in orbs if o['pivot'] == c), c
    return orbs


if __name__ == "__main__":
    ci, job = sys.argv[1], int(sys.argv[2]); maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    po = load_job(ci, job); In = {int(x): v for x, v in po["In"].items()}; A = po["A"]
    leaves = [tuple(sorted(tuple(z) for z in L)) for L in po["leaves"]]
    blockof = lambda L, c: tuple(z for z in L if z[0] == c)
    os.makedirs("logs/pair_certs", exist_ok=True)
    out = open("logs/pair_certs/%s_%d.jsonl" % (ci, job), "w")
    stack = [({}, list(range(len(leaves))))]; nodes = 0; t0 = time.time(); open_nodes = 0; dead_leaves = 0; stat = Counter()
    while stack:
        assign, idx = stack.pop()
        if nodes >= maxn: open_nodes += 1 + len(stack); break
        nodes += 1; t1 = time.time()
        orbs = orbit_list(In, assign)
        r = certify(*build(orbs)[:5], maxnodes=int(os.environ.get('MAXBB', 60)))
        stat[r['status'] + ('/' + r['level'] if r['status'] == 'dead' else '')] += 1
        if r['status'] == 'dead':
            dead_leaves += len(idx)
            out.write(json.dumps({"node": nodes, "spec": {"ci": ci, "job": job, "In": po["In"], "A": A,
                                  "assign": {str(c): [list(z) for z in v] for c, v in assign.items()}},
                                  "nleaves": len(idx), "level": r['level'], "orbs": orbs, "tree": r['tree']}) + chr(10)); out.flush()
        if os.environ.get('V'): print(nodes, len(assign), len(idx), r['status'], r.get('level', r.get('why', '')), r['nodes'], '%.1fs' % (time.time() - t1)); sys.stdout.flush()
        if r['status'] == 'dead': continue
        free_blocks = [c for c in range(N) if c != A and c not in assign]
        opts = {c: Counter(blockof(leaves[t], c) for t in idx) for c in free_blocks}
        cands = [c for c in free_blocks if len(opts[c]) > 1] or [c for c in free_blocks if opts[c]]
        if not cands:
            print("UNRESOLVED fully fixed node", {c: v for c, v in assign.items()}, r['status']); open_nodes += 1; continue
        c = min(cands, key=lambda b: len(opts[b]))
        for trip in opts[c]:
            a2 = dict(assign); a2[c] = trip
            stack.append((a2, [t for t in idx if blockof(leaves[t], c) == trip]))
        if nodes % 50 == 0:
            print("  nodes %d stack %d dead leaves %d/%d %s %.0fs" % (nodes, len(stack), dead_leaves, len(leaves), dict(stat), time.time() - t0)); sys.stdout.flush()
    print("PAIRDFS %s job %d: leaves %d nodes %d dead leaves %d open %d %s %s %.0fs" % (ci, job, len(leaves), nodes, dead_leaves, open_nodes,
          dict(stat), "ALL DEAD" if open_nodes == 0 and dead_leaves == len(leaves) else "INCOMPLETE", time.time() - t0))
