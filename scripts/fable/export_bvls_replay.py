"""Export the BvLS block-0 data (type, Gpinv, true phi, true far-far T) for the C replay control."""
import numpy as np
from inv import bvls_in_labelling
from analyze import block_type, phi_map, Orbits, sign_matrix
from phisolve import TypeInstance

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
with open('data_bvls.txt', 'w') as f:
    f.write('%d\n' % m)
    f.write(' '.join(map(str, mu)) + '\n')
    f.write(' '.join(map(str, tau)) + '\n')
    for row in inst.Gpinv:
        f.write(' '.join(repr(float(x)) for x in row) + '\n')
def root_of(z):
    a, b = cfg.ends(z)
    return inst.root_index[tuple(sorted((lab_idx[a], lab_idx[b])))]
phi_true = [-1] * inst.nroots
for z, pair in Phi.items():
    p = tuple(sorted(lab_idx[v] for v in pair))
    phi_true[root_of(z)] = inst.pair_index[p]
orb = Orbits(cfg)
S = sign_matrix(cfg, A, orb)
def rep_vertex(rootk):
    a, b = inst.roots[rootk]
    return cfg.D((a // 2 + 1, a % 2), (b // 2 + 1, b % 2))
no = inst.norb
T = np.zeros((no, no), dtype=int)
for i, ki in enumerate(inst.orbits):
    for j, kj in enumerate(inst.orbits):
        if i != j:
            T[i, j] = S[orb.index[rep_vertex(ki)], orb.index[rep_vertex(kj)]]
with open('replay_bvls.txt', 'w') as f:
    f.write(' '.join(map(str, phi_true)) + '\n')
    for i in range(no):
        f.write(' '.join(map(str, T[i])) + '\n')
print('exported: nroots', inst.nroots, 'norb', no, 'nonzero T per row', sorted(set((T != 0).sum(axis=1))))
