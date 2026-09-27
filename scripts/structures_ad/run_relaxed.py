"""Relaxed pair-level test (free structures as partner variables) over a dump of AD orientation classes.
Usage: python run_relaxed.py reps.json [lo hi] [nodich]"""
import sys, json, time
from pair_level import N, pr
from pair_level_relaxed import feasible_relaxed

reps = json.load(open(sys.argv[1]))
lo_, hi_ = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, len(reps))
dich = not (len(sys.argv) > 4 and sys.argv[4] == 'nodich')
t0 = time.time()
bynd = {}
for t in range(lo_, min(hi_, len(reps))):
    rep = reps[t]
    dd = [tuple(p) for p in rep['dd']]
    head = {pr(a, h): h for a, h in rep['arcs']}
    In = {c: [x for x in range(N) if x != c and head.get(pr(c, x)) == c] for c in range(N)}
    r = feasible_relaxed(In, dd, dichotomy=dich)
    s = bynd.setdefault(len(dd), [0, 0, 0])
    s[0] += 1
    if r is not None:
        if isinstance(r, str):
            s[2] += 1
            print('UNKNOWN rep', t, r, rep)
        else:
            s[1] += 1
            print('ALIVE rep', t, 'n_dd', len(dd), rep)
        sys.stdout.flush()
    if t % 200 == 0:
        print('..', t, '%.0fs' % (time.time() - t0), bynd)
        sys.stdout.flush()
print('done', bynd, '%.0fs' % (time.time() - t0))
