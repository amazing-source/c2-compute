import sys, time
from phisolve import TypeInstance, load_type
from iota import IotaSearch
mu, tau = load_type('type_tier8_0.json')
inst = TypeInstance(7, mu, tau)
s = IotaSearch(inst, time_limit=60, verbose=0, max_solutions=1)
sols = s.run()
print('python: nodes', s.nodes, 'sols', len(sols))
if sols:
    io = sols[0]
    # verify K3 directly
    nl = inst.nl
    K3 = [[0]*nl for _ in range(nl)]
    for p, q in enumerate(io):
        for v in inst.pairs[p]:
            for w in inst.pairs[q]:
                K3[v][inst.pi[w]] += 1
    print('K3 matches definition:', K3 == inst.K3)
    print('iota fixed-point-free involution:', all(io[io[p]] == p and io[p] != p for p in range(60)))
# root statistics
s2 = IotaSearch(inst)
K3rem = s2.K3rem
counts = []
for p in range(60):
    n = 0
    for q in range(60):
        if q == p: continue
        a, b = min(p,q), max(p,q)
        if all(K3rem[v][w] >= 1 for (v,w) in s2._contrib(a,b)): n += 1
    counts.append(n)
print('python root feasible-partner counts:', counts)
print('K3 rows:', inst.K3)
