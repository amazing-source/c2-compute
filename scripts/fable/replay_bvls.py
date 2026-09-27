"""Positive control: replay the true block-local data of BvLS (m = 11) through the constraint machinery."""
import numpy as np, time
from inv import Config, bvls_in_labelling
from analyze import block_type, phi_map, Orbits, sign_matrix, srg_eigs
from phisolve import TypeInstance, PhiSearch, Residual

cfg, A = bvls_in_labelling()
m = 11
mu_d, tau_d = block_type(cfg, A)
Phi = phi_map(cfg, A)
lab_idx = {v: 2 * (v[0] - 1) + v[1] for v in cfg.labels}
mu = [0] * 20; tau = [0] * 20
for v in cfg.labels:
    mu[lab_idx[v]] = lab_idx[mu_d[v]]
    tau[lab_idx[v]] = lab_idx[tau_d[v]]
inst = TypeInstance(m, mu, tau)
print('BvLS block type: pivots', sum(inst.pivot), 'valid', inst.valid, 'dimK', inst.dimK, 'resdim', inst.resdim)
# true Phi in inst indexing
def root_of(z):
    a, b = cfg.ends(z)
    return inst.root_index[tuple(sorted((lab_idx[a], lab_idx[b])))]
true_phi = {}
for z, pair in Phi.items():
    p = tuple(sorted(lab_idx[v] for v in pair))
    true_phi[root_of(z)] = inst.pair_index[p]
assert len(true_phi) == inst.nroots
# check iota consistency and K3
iota = {}
for r, p in true_phi.items():
    iota[p] = true_phi[inst.troot[r]]
assert all(iota[iota[p]] == p and iota[p] != p for p in iota)
K3 = [[0] * inst.nl for _ in range(inst.nl)]
for p, q in iota.items():
    for v in inst.pairs[p]:
        for w in inst.pairs[q]:
            K3[v][inst.pi[w]] += 1
print('K3 from true iota matches formula:', K3 == inst.K3)
K2 = [[0] * inst.nl for _ in range(inst.nl)]
for r, p in true_phi.items():
    for a in inst.roots[r]:
        for v in inst.pairs[p]:
            K2[a][v] += 1
print('K2 from true Phi matches formula:', K2 == inst.K2)
# replay through PhiSearch: local bounds, pair bounds, counts, then residual with true T
srch = PhiSearch(inst, use_residual=False, verbose=0)
ok_local = all(inst.local_ok(k, inst.pairs[true_phi[k]], inst.pairs[true_phi[inst.troot[k]]]) for k in inst.orbits)
print('local bounds hold for all true orbits:', ok_local)
bad = 0
for k in srch.order:
    pz, ptz = inst.pairs[true_phi[k]], inst.pairs[true_phi[inst.troot[k]]]
    if not srch._pair_ok(k, pz, ptz):
        bad += 1
    srch._apply(k, true_phi[k], true_phi[inst.troot[k]], +1)
    if not srch._feasible_counts():
        bad += 1
print('replay: pair-bound / count violations:', bad, '; all pairs used:', all(srch.pair_used))
# true T at orbit level and residual check with true T as the only allowed value
orb = Orbits(cfg)
S = sign_matrix(cfg, A, orb)
r, s, d = srg_eigs(m)
def orbS(k1, k2):
    z1 = inst.roots[inst.orbits[k1]] if False else None
# map inst orbit (rep root index) -> Orbits index
def rep_vertex(rootk):
    a, b = inst.roots[rootk]
    la, sa = a // 2 + 1, a % 2
    lb, sb = b // 2 + 1, b % 2
    return cfg.D((la, sa), (lb, sb))
orbs = srch.order
ks, rhos = [], []
for k in orbs:
    sig = inst.shadow(inst.pairs[true_phi[k]], inst.pairs[true_phi[inst.troot[k]]])
    g = inst.gvec(k, sig)
    kv = inst.kvec(g)
    ks.append(kv); rhos.append(inst.r - float(kv @ kv))
print('rho range %.4f..%.4f, sum %.4f (expected d*resdim = %d)' % (min(rhos), max(rhos), sum(rhos), d * inst.resdim))
# true T between orbits: S[orbit i, orbit j] in Orbits indexing with the same reps? Orbits reps: far orbit rep has sign 0 on smaller line = inst rep (a even). good.
T_true = {}
n = len(orbs)
allowed = [[None] * n for _ in range(n)]
for i in range(n):
    for j in range(i):
        vi = rep_vertex(orbs[i]); vj = rep_vertex(orbs[j])
        t = int(S[orb.index[vi], orb.index[vj]])
        al = srch._allowed_T(orbs[i], orbs[j])
        assert t in al, (t, al)
        allowed[i][j] = [t]
res = Residual(inst)
t0 = time.time()
sols = res.complete(ks, rhos, allowed, max_solutions=1)
print('residual completion with the true T: solutions', len(sols), '(%.1fs)' % (time.time() - t0))
