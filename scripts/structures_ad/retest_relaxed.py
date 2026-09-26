"""Re-test the classes left ALIVE/UNKNOWN by run_relaxed: enumerate the D-structures of the blocks with d >= 4 (A-blocks,
degree-4 blocks) so that the balance holds on each of their D-cycles; dd partners stay as variables.
Usage: python retest_relaxed.py reps.json log1 [log2 ...]"""
import sys, json, itertools, re, time
from pair_level import N, pr, two_regular_graphs
from pair_level_relaxed import feasible_relaxed
reps = json.load(open(sys.argv[1]))
idx = []
for lg in sys.argv[2:]:
    for line in open(lg):
        m = re.match(r'(ALIVE|UNKNOWN) rep (\d+)', line)
        if m: idx.append(int(m.group(2)))
for t in sorted(set(idx)):
    rep = reps[t]; dd = [tuple(p) for p in rep['dd']]
    head = {pr(a, h): h for a, h in rep['arcs']}
    In = {c: [x for x in range(N) if x != c and head.get(pr(c, x)) == c] for c in range(N)}
    free = [k for k in range(N) if len(In[k]) >= 4]
    opts = [two_regular_graphs(In[k]) for k in free]
    t0 = time.time(); alive = []; unknown = 0; n = 0
    for choice in itertools.product(*opts):
        n += 1
        r = feasible_relaxed(In, dd, fixedDs=dict(zip(free, choice)), time_limit=60)
        if r is None: continue
        if isinstance(r, str): unknown += 1
        else: alive.append(choice)
    print('rep', t, 'n_dd', len(dd), 'free blocks', free, 'structures', n, 'alive', len(alive), 'unknown', unknown,
          '%.0fs' % (time.time() - t0), alive[:2], flush=True)
