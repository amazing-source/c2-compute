"""Stage 2 of the g- = 16 local test (regime t = 2): on the options surviving the unary filters (balance + Z_ff = 0),
(i) every non-mu pair used once, (ii), (iii) of C2's free-star lemma, plus the pairwise rigid-row conditions
T_ff' = Z_ff' in {0, +-1} (conflicts) and the row counts #{f' : Z_ff' != 0} = 10 - |supp s_f| (big-M), as one small ILP."""
import sys, json, time, itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
from q7_g16_test import unary_filter, K2, K3, FAR, L, P7


def stage2(tau, eta, tl=60, pairwise=True):
    built = build_model(tau, eta, pairwise)
    if built is None:
        return 'dead-unary', {}
    A, lo, hi, nv, info = built
    t0 = time.time()
    resm = milp(np.zeros(nv), constraints=[LinearConstraint(A, np.array(lo, float), np.array(hi, float))],
                integrality=np.ones(nv), bounds=Bounds(0, 1), options={'time_limit': tl})
    st = {0: 'FEASIBLE', 2: 'INFEASIBLE'}.get(resm.status, 'unknown(%d)' % resm.status)
    info['time'] = time.time() - t0
    return st, info


def build_model(tau, eta, pairwise=True):
    t, res, Pp = unary_filter(tau, eta, regime_t2=True)
    if any(len(r[0]) == 0 for r in res):
        return None
    k2, k3 = K2(tau, eta), K3(tau, eta)
    var = []                      # (orbit index, option index)
    for fi, (opts, Xc) in enumerate(res):
        for j in range(len(opts)):
            var.append((fi, j))
    nv = len(var)
    vid = {v: i for i, v in enumerate(var)}
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = [0]

    def add(entries, a, b):
        for j, c in entries:
            rows.append(r[0]); cols.append(j); vals.append(c)
        lo.append(a); hi.append(b); r[0] += 1
    # one option per orbit
    for fi, (opts, Xc) in enumerate(res):
        add([(vid[(fi, j)], 1) for j in range(len(opts))], 1, 1)
    # (i) each non-mu pair used exactly once (as Phi(z) or Phi(sigma z))
    nonmu = [e for e in itertools.combinations(L, 2) if tau[e[0]] != e[1]]
    use = {e: [] for e in nonmu}
    acc2, acc3 = {}, {}
    for fi, (opts, Xc) in enumerate(res):
        z, sz = FAR[fi]
        for j, (e, e2) in enumerate(opts):
            v = vid[(fi, j)]
            use[e].append(v); use[e2].append(v)
            for a in z:
                for b in e:
                    acc2.setdefault((a, b), []).append(v)
            for a in sz:
                for b in e2:
                    acc2.setdefault((a, b), []).append(v)
            for b in e:
                for d in e2:
                    acc3.setdefault((b, d), []).append(v)
            for b in e2:
                for d in e:
                    acc3.setdefault((b, d), []).append(v)
    for e in nonmu:
        add([(v, 1) for v in use[e]], 1, 1)
    for a in L:
        for b in L:
            add([(v, 1) for v in acc2.get((a, b), [])], int(k2[a, b]), int(k2[a, b]))
            add([(v, 1) for v in acc3.get((a, b), [])], int(k3[a, b]), int(k3[a, b]))
    ncons_local = r[0]
    if pairwise:
        # Z between options: Z_ff' = x_f^T P^+ x_f'
        Zbad, Znz = [], {}
        for fi in range(len(res)):
            Xf = res[fi][1]
            for gi in range(fi + 1, len(res)):
                Xg = res[gi][1]
                Zm = (Xf.T @ ((Pp @ Xg) % P7)) % P7
                for j in range(Zm.shape[0]):
                    for k in range(Zm.shape[1]):
                        val = int(Zm[j, k])
                        if val not in (0, 1, 6):
                            add([(vid[(fi, j)], 1), (vid[(gi, k)], 1)], 0, 1)
                        elif val:
                            Znz.setdefault((fi, j), []).append(vid[(gi, k)])
                            Znz.setdefault((gi, k), []).append(vid[(fi, j)])
        # row counts
        for fi, (opts, Xc) in enumerate(res):
            for j, (e, e2) in enumerate(opts):
                supp = len(set(e) ^ set(e2))
                need = 10 - supp
                v = vid[(fi, j)]
                ent = [(u, 1) for u in Znz.get((fi, j), [])]
                M = 40
                # sum ent - need <= M (1 - y)  and  sum ent - need >= -M (1 - y)
                add(ent + [(v, M)], -np.inf, need + M)
                add(ent + [(v, -M)], need - M, np.inf)
    A = sp.csr_matrix((np.array(vals, float), (np.array(rows), np.array(cols))), shape=(r[0], nv))
    return A, lo, hi, nv, {'nvar': nv, 'ncons': r[0], 'ncons_local': ncons_local,
                           'opts_per_orbit': [len(x[0]) for x in res], 'var': var, 'res_opts': [x[0] for x in res]}


if __name__ == '__main__':
    summ = json.load(open('g16_unary_summary.json'))
    alive = [s for s in summ if s['t'] == 2 and s['min_opts'] > 0]
    print('t = 2 classes alive after unary + Witt:', len(alive), flush=True)
    t0 = time.time()
    for s in alive:
        if time.time() - t0 > 250:
            print('time cap reached'); break
        st, info = stage2(tuple(s['tau']), tuple(s['eta']), tl=30)
        print(s['tau'], s['eta'], st, info, flush=True)
