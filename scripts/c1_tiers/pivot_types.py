"""Block types (tau, eta) = C2's (mu, tau) with pivots (mu cap tau != empty), for the local-system SAT campaign (C1, 27/09).
A pivot label l has mu(l) = tau(l) (its star orbit's D-partner lies in star(c)).  Per block x (x = 0..5, the six other blocks):
#pivot labels in x = 2 (pair {c,x} anchored at c), 1 (twisted pair, one orbit pivots at c) or 0 (anchored at x) - this is
C2's (anch_c, tw_c) profile.  Writes data/pivot_types.json: one record per class (up to Aut(X, pi)) with
mu, tau, pivot labels, per-block pivot counts, (|anch|, |tw|), |Stab_B6(mu, tau)|, t (odd torsion), det class, t+ (even).
Usage: python pivot_types.py OUT.json"""
import sys, json, itertools
from collections import Counter
from q7_g16_test import classes_t_ge, ball_disc, split, L, PI
from q7_gplus_test import even_ball, split_type


def b6():
    for perm in itertools.permutations(range(6)):
        for flips in range(64):
            yield tuple(2 * perm[i // 2] + ((i % 2) ^ ((flips >> (i // 2)) & 1)) for i in L)


def stabiliser(mu, tau):
    return [g for g in b6() if all(g[mu[v]] == mu[g[v]] and g[tau[v]] == tau[g[v]] for v in L)]


def profile(mu, tau):
    piv = [l for l in L if mu[l] == tau[l]]
    per_block = [sum(1 for l in piv if l // 2 == x) for x in range(6)]
    return piv, per_block, (per_block.count(2), per_block.count(1))


if __name__ == '__main__':
    out = []
    cls = classes_t_ge(0)
    for (mu, tau, t, osz) in cls:
        piv, pb, at = profile(mu, tau)
        if not piv:
            continue
        r, sp = split_type(even_ball(mu, tau))
        rec = {'mu': list(mu), 'tau': list(tau), 'pivots': piv, 'per_block': pb, 'anch_tw': list(at),
               'n_pivot_labels': len(piv), 't': t, 'odd_split': split(*ball_disc(mu, tau)),
               't_plus': 18 - r, 'even_split': sp, 'orbit_size': osz}
        out.append(rec)
    G_orders = []
    for rec in out:
        rec['stab_order'] = len(stabiliser(rec['mu'], rec['tau']))
    json.dump(out, open(sys.argv[1], 'w'))
    print('pivot classes:', len(out))
    print('by (|anch|, |tw|):', sorted(Counter(tuple(r['anch_tw']) for r in out).items()))
    print('stabiliser orders:', sorted(Counter(r['stab_order'] for r in out).items()))
