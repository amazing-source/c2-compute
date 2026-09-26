"""Re-check of the pair-level block-DFS certificates (note 43), written apart from pair_blockdfs.py (no code shared with it).
For every dead node of logs/pair_certs/<ci>_<job>.jsonl:
  1. the stored skeleton (42 orbits: pair, pivot, partner) is checked against the spec alone (valid_skeleton): pivots from the
     orientation, partners realising exactly the option's triples at the assigned blocks, no partner elsewhere (which orbit of a
     pair gets which third is WLOG: orbits of a pair pivoting at the same block are interchangeable in the model);
  2. the model is rebuilt with structures' general_model.build and the tree is checked with ad/certify.check (exact rationals).
Coverage: every leaf of the job agrees with the assignment of at least one dead node on all its assigned blocks.
Usage: python pair_verify.py <ci> <job> [...more ci job pairs]"""
import sys, json, glob, os, time
sys.path.insert(0, os.environ.get('STRUCT_AD', (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad') if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external_structures', 'ad')) else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor_structures'))))
from general_model import build
from certify import check
N = 7


def valid_skeleton(orbs, In, assign):
    """the stored skeleton represents the node: 42 orbits, two per pair; pivots from the orientation (anchored pair {c,x}: both at
    c; twisted pair: one at each end); at an assigned block c every orbit pivoting at c has a partner, and the multiset of triples
    (c, x = other block of the pair, y = other block of the partner's pair) equals the option; elsewhere no partner."""
    anchored_at = {}
    for c, xs in In.items():
        for x in xs: anchored_at[(min(c, x), max(c, x))] = c
    if len(orbs) != 42: return False
    for a in range(N):
        for b in range(a + 1, N):
            piv = sorted(o['pivot'] for o in orbs if tuple(o['pair']) == (a, b))
            want = [anchored_at[(a, b)]] * 2 if (a, b) in anchored_at else [a, b]
            if piv != want: return False
    for c in range(N):
        mine = [i for i, o in enumerate(orbs) if o['pivot'] == c]
        if c not in assign:
            if any(orbs[i]['partner'] is not None for i in mine): return False
            continue
        trip = []
        for i in mine:
            j = orbs[i]['partner']
            if j is None or orbs[j]['pivot'] != c or orbs[j]['partner'] != i or j == i: return False
            x = sum(orbs[i]['pair']) - c; y = sum(orbs[j]['pair']) - c
            if c not in orbs[i]['pair'] or c not in orbs[j]['pair'] or x == y: return False
            trip.append((c, x, y))
        if sorted(trip) != sorted(assign[c]): return False
    return True


def leaves_of(ci, job):
    jobs = []
    for f in sorted(glob.glob("logs/mix6_cls_%s_S_*.json" % ci)):
        for po in json.load(open(f))["per_orient"]:
            if po["leaves"]: jobs.append(po)
    return jobs[job]


if __name__ == "__main__":
    allok = True
    for ci, job in zip(sys.argv[1::2], map(int, sys.argv[2::2])):
        t0 = time.time(); po = leaves_of(ci, job)
        dead = [json.loads(l) for l in open("logs/pair_certs/%s_%d.jsonl" % (ci, job))]
        ok = bad = 0; covers = []
        for rec in dead:
            sp = rec["spec"]
            assert sp["In"] == po["In"] and sp["A"] == po["A"]
            In = {int(c): v for c, v in sp["In"].items()}
            assign = {int(c): [tuple(z) for z in v] for c, v in sp["assign"].items()}
            orbs = [{'pair': tuple(o['pair']), 'pivot': o['pivot'], 'partner': o['partner']} for o in rec["orbs"]]
            good = valid_skeleton(orbs, In, assign)
            if good:
                A_, lo, hi, lb, ub = build(orbs)[:5]
                good = check(A_, lo, hi, lb, ub, rec["tree"])
            ok += good; bad += not good
            if good: covers.append({c: tuple(sorted(v)) for c, v in assign.items()})
        unc = 0
        for L in po["leaves"]:
            L = [tuple(z) for z in L]
            blk = lambda c: tuple(sorted(z for z in L if z[0] == c))
            if not any(all(blk(c) == v for c, v in cv.items()) for cv in covers): unc += 1
        res = bad == 0 and unc == 0; allok &= res
        print("%s job %d: dead nodes %d valid %d INVALID %d | leaves %d uncovered %d  %s %.0fs" % (ci, job, len(dead), ok, bad,
              len(po["leaves"]), unc, "OK" if res else "FAIL", time.time() - t0)); sys.stdout.flush()
    print("ALL OK" if allok else "SOME FAIL")
