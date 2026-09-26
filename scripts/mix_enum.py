"""Mixtures (twisted subgraph = any even subgraph, note 35): orientation and D-structure enumeration with PROVED degree lemmas only.
Degree lemmas (Sigma = empty, one-block squares):
  t_c = 2  =>  out_c >= 2 (19.1: f = 2 out + 2; out = 0 gives a digon, out = 1 a 4-cycle with two sisters sharing 2 neighbours)
  t_c = 0  =>  d_c in {0, 2, 3, 6} (26.2: loop, K22 / digon, and the tail 26.3 for arms)
D-graph at a pivot c: multigraph on the pairs {c,x} with c-orbits: anchored-at-c pairs degree 2, twisted pairs degree 1, no loop."""
import itertools, sys
from collections import Counter
N = 7
P = lambda a, b: frozenset((a, b))

def dgraphs(anch, tw):
    """all D-graphs at a block: anch = list of anchored pair labels (degree 2), tw = twisted pair labels (degree 1).
    Returned as sorted tuples of edges (x, y) with x <= y (multi-edges allowed, no loops)."""
    half = [x for x in anch for _ in (0, 1)] + list(tw)
    res = set()
    def rec(rem, acc):
        if not rem: res.add(tuple(sorted(acc))); return
        a = rem[0]
        for t in range(1, len(rem)):
            b = rem[t]
            if a == b: continue
            rec(rem[1:t] + rem[t + 1:], acc + [tuple(sorted((a, b)))])
    rec(sorted(half), [])
    return sorted(res)

def orientations(twE):
    tw = {P(a, b) for a, b in twE}
    tdeg = Counter(v for e in tw for v in e)
    rest = [P(a, b) for a, b in itertools.combinations(range(N), 2) if P(a, b) not in tw]
    for bits in range(1 << len(rest)):
        In = {c: [] for c in range(N)}
        for t, R in enumerate(rest):
            a, b = sorted(R)
            head, tail = (a, b) if (bits >> t) & 1 else (b, a)
            In[head].append(tail)
        ok = True
        for c in range(N):
            d = len(In[c]); t_ = tdeg[c]; out = 6 - d - t_
            if t_ == 2 and out < 2: ok = False; break
            if t_ == 0 and d not in (0, 2, 3, 6): ok = False; break
        if ok: yield In, tw

if __name__ == "__main__":
    twE = [tuple(map(int, e)) for e in sys.argv[1].split(",")]
    no = 0; tot = 0
    for In, tw in orientations(twE):
        no += 1; prod = 1
        for c in range(N):
            tws = [x for x in range(N) if P(c, x) in tw]
            prod *= len(dgraphs(In[c], tws))
        tot += prod
    print("twisted", twE, "| admissible orientations", no, "| labelled skeletons (before any filter)", tot)
