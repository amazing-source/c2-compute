"""Even sector at q:7 (C1, 27/09): can g+ be 14 or 16?  (g+ := rank_7 of the even part of Q = A - 3I + 3J, so that
14 + k = g- + g+.)  Even ball of a block line (20 coordinates: x, the 7 blocks, the 12 star orbits of c; basis
e_x, e_{b+} + e_{b-}, e_v + e_{sigma v}; symmetric form D(A - 3I) + 3 w w^T, w = (1, 2, ..., 2)): rank 18 - t+, with a
2-dimensional trivial kernel that is automatically balanced (checked on options) and a t+-dimensional even torsion.
Regimes (notes/126 split by sectors): g+ = (18 - t+) + 2(t+ - r0+) + rho+, and Witt with beta+ NON-split (notes/129 sec. 4):
rho+ >= 2 if 18 - t+ is even and the reduced even ball is SPLIT, rho+ >= 1 if 18 - t+ is odd.
  g+ = 16: t+ = 2: r0 = 2, rho = 0, non-split ball;  t+ = 3: r0 = 3;  t+ = 4: (3, 0) non-split or (4, 2);  t+ = 5: (4, 1) or (5, 3).
  g+ = 14: t+ = 4: (4, 0) non-split;  t+ = 5: (5, 1).
  t+ <= 1 impossible for g+ <= 16 (18 - t+ <= g+).
Unary filters on the options (Phi(z), Phi(sigma z)) of each far orbit, via its even column
  x+ = (6 on x; 12 + 2[b in supp f] on block b; 12 + 2 q_f(l) on star orbit g_l),  q_f(l) = [l in Phi z] + [l in Phi sigma z]:
  balance: x+ orthogonal to (trivial kernel +) the required balanced torsion subspace;
  rigid diagonal (full balance and rho+ = 0): x+^T P^- x+ = N+_ff = 2 Q_ff + 1 + 5 = 6 (mod 7).
Then the local model (i)+(ii)+(iii) (+ pointwise filters with Q7_POINTWISE=1) on the surviving options."""
import sys, itertools, time, json, os
import numpy as np
ILP_TL = float(os.environ.get('Q7_ILP_TL', '60'))
from q7_g16_test import (classes_t_ge, congruence_diag, kernel_mod, rank_mod, ginv_sym, orbit_options, K2, FAR, P7, L)
from q7_g18_regimes import local_ilp


def even_ball(tau, eta):
    n = 20
    M = np.zeros((n, n), dtype=np.int64)
    w = np.array([1] + [2] * 7 + [2] * 12)
    M[0, 0] = -3
    for b in range(1, 8):
        M[0, b] = M[b, 0] = 2
        M[b, b] = -4
    for l in L:
        g = 8 + l
        for b in (1 + l // 2, 7):
            M[b, g] = M[g, b] = 2
        M[g, 8 + tau[l]] += 2
        M[g, 8 + eta[l]] += 2
        M[g, g] -= 6
    return (M + 3 * np.outer(w, w)) % P7


def even_col(z, e, e2):
    x = np.zeros(20, dtype=np.int64)
    x[0] = 6
    for b in range(1, 8):
        x[b] = 12
    for a in z:
        x[1 + a // 2] += 2
    for l in L:
        x[8 + l] = 2 * ((l in e) + (l in e2)) + 12
    return x % P7


def split_type(Mat):
    d = congruence_diag(Mat.tolist())
    r = len(d)
    det = 1
    for x in d:
        det = det * x % P7
    leg = 1 if pow(det, 3, P7) == 1 else -1
    return r, (None if r % 2 else leg == (1 if (r // 2) % 2 == 0 else -1))


def trivial_and_torsion(P):
    """kernel basis split: trivial part (2 dims, the kernel of the ball for generic types is automatically balanced) is
    not identifiable basis-free; we return the whole kernel and its dimension."""
    return kernel_mod(P.tolist())


def filtered(tau, eta, Kbal, rigid):
    P = even_ball(tau, eta)
    Pp = ginv_sym(P) if rigid else None
    k2 = K2(tau, eta)
    res = []
    for (opts, Xc), (z, sz) in zip(orbit_options(tau, eta, k2), FAR):
        keep = []
        for (e, e2) in opts:
            x = even_col(z, e, e2)
            if len(Kbal) and np.any((Kbal @ x) % P7):
                continue
            if rigid and int(x @ ((Pp @ x) % P7)) % P7 != 6:
                continue
            keep.append((e, e2))
        res.append((keep, Xc[:, :0]))            # local_ilp only uses the option lists
    return res


def proj_points(t):
    for v in itertools.product(range(P7), repeat=t):
        if any(v) and next(x for x in v if x) == 1:
            yield np.array(v, dtype=np.int64)


def run(gplus, tcap=270):
    t0 = time.time()
    cls = classes_t_ge(0)
    out = []
    for (tau, eta, t, osz) in cls:
        P = even_ball(tau, eta)
        r, sp = split_type(P)
        tp = 18 - r
        if tp < 18 - gplus:
            continue
        if time.time() - t0 > tcap:
            print('TIME CAP'); break
        K = kernel_mod(P.tolist())                  # 2 + tp vectors
        # regimes
        choices = []                                # (label, balanced subspace (as kernel combos) or 'full', rigid?)
        if gplus == 16:
            if tp == 2:
                if sp is True:
                    out.append((tau, eta, tp, 'dead-witt')); continue
                choices.append(('full', True))
            elif tp == 3:
                choices.append(('full', False))
            elif tp == 4:
                choices.append(('full', False))                      # (4, 2)
                if sp is False:
                    choices.append(('hyper', False))                 # (3, 0): balanced hyperplane of the torsion
            elif tp == 5:
                choices.append(('hyper', False))                     # (4, 1) (and (5,3) is contained in it)
        elif gplus == 14:
            if tp == 4:
                if sp is True:
                    out.append((tau, eta, tp, 'dead-witt')); continue
                choices.append(('full', True))
            elif tp == 5:
                choices.append(('full', False))
        verdicts = []
        for kind, rigid in choices:
            if kind == 'full':
                st = local_ilp(tau, eta, filtered(tau, eta, K, rigid), tl=ILP_TL)
                verdicts.append(st)
            else:
                # balanced subspace = a hyperplane of the kernel CONTAINING the trivial part (automatically balanced):
                # trivial coefficient vectors c (c.K orthogonal to every option column); hyperplanes a.c' = 0 with a in
                # trivial^perp (coefficient space), up to scalars: (7^tp - 1)/6 of them.
                k2_ = K2(tau, eta)
                cols = [even_col(z, e, e2) for (opts, Xc), (z, sz) in zip(orbit_options(tau, eta, k2_), FAR)
                        for (e, e2) in opts]
                Xall = (K @ np.array(cols).T) % P7                    # dimK x N
                triv = kernel_mod(Xall.T.tolist())                   # coefficient vectors c with c.K balanced for all
                assert len(triv) == 2, ('trivial part', len(triv))
                Aperp = kernel_mod(triv.tolist())                    # a with a.c = 0 for trivial c  (dim = tp)
                alive = False
                for coeff in proj_points(Aperp.shape[0]):
                    a = (coeff @ Aperp) % P7
                    # subspace {c : a.c = 0}: basis = rows of the nullspace of a
                    B = kernel_mod([list(a)])               # (n-1) x n coefficient basis
                    Ksub = (np.array(B) @ K) % P7
                    st = local_ilp(tau, eta, filtered(tau, eta, Ksub, False), tl=ILP_TL / 2)
                    if st not in ('dead-unary', 'dead-local'):
                        alive = True; break
                verdicts.append('some hyperplane alive' if alive else 'all hyperplanes dead')
        dead = all(v in ('dead-unary', 'dead-local', 'all hyperplanes dead') for v in verdicts)
        out.append((tau, eta, tp, 'dead' if dead else 'ALIVE ' + str(verdicts)))
        print('t+=%d %s %s -> %s (%.0fs)' % (tp, {True: 'split', False: 'nonsplit', None: 'odd'}[sp], tuple(tau), out[-1][3],
                                             time.time() - t0), flush=True)
    from collections import Counter
    print('g+ = %d: ' % gplus, Counter(o[3] if not o[3].startswith('ALIVE') else 'ALIVE' for o in out))
    json.dump([{'tau': list(o[0]), 'eta': list(o[1]), 't_plus': o[2], 'verdict': o[3]} for o in out],
              open('gplus_%d_results.json' % gplus, 'w'), indent=1)
    return out


if __name__ == '__main__':
    g = int(sys.argv[1]); tcap = float(sys.argv[2]) if len(sys.argv) > 2 else 270
    run(g, tcap)
