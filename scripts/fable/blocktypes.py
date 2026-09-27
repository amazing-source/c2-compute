"""
Enumerate block types (mu, tau) for m = 7 up to the label group G = 2^6 x| S_6 (order 46080), and their validity
(K2, K3 >= 0).  Labels 0..11, line(v) = v // 2, pi(v) = v ^ 1.
Output: types_all.json = list of {mu, tau, stab, orbit_size, pivots, valid}.
"""
import itertools, json, sys
from collections import deque

NL = 12
LINES = 6


def perfect_matchings(elems):
    if not elems:
        yield ()
        return
    a = elems[0]
    for i in range(1, len(elems)):
        b = elems[i]
        rest = elems[1:i] + elems[i + 1:]
        for pm in perfect_matchings(rest):
            yield ((a, b),) + pm


def matching_to_inv(pm):
    inv = [0] * NL
    for a, b in pm:
        inv[a] = b
        inv[b] = a
    return tuple(inv)


def group_elements():
    """all label permutations g = (line permutation sigma, sign flips eps) as tuples g[v]."""
    els = []
    for sigma in itertools.permutations(range(LINES)):
        for eps in itertools.product((0, 1), repeat=LINES):
            g = tuple(2 * sigma[v // 2] + ((v % 2) ^ eps[v // 2]) for v in range(NL))
            els.append(g)
    return els


def act(g, inv):
    """conjugate an involution inv by g: (g inv g^-1)[g v] = g inv[v]."""
    out = [0] * NL
    for v in range(NL):
        out[g[v]] = g[inv[v]]
    return tuple(out)


def main():
    G = group_elements()
    assert len(G) == 46080
    gens = []
    # generators: line transpositions (0 1), 6-cycle, one sign flip
    for g in G:
        pass
    # BFS orbits of mu under G using all elements is expensive (10395 x 46080); use generators
    def line_perm(sigma, eps=(0,) * LINES):
        return tuple(2 * sigma[v // 2] + ((v % 2) ^ eps[v // 2]) for v in range(NL))
    gens = [line_perm((1, 0, 2, 3, 4, 5)), line_perm((1, 2, 3, 4, 5, 0)), line_perm((0, 1, 2, 3, 4, 5), (1, 0, 0, 0, 0, 0))]
    all_mu = [matching_to_inv(pm) for pm in perfect_matchings(list(range(NL)))]
    assert len(all_mu) == 10395
    seen = {}
    mu_reps = []
    for mu in all_mu:
        if mu in seen:
            continue
        orbit = {mu}
        dq = deque([mu])
        while dq:
            x = dq.popleft()
            for g in gens:
                y = act(g, x)
                if y not in orbit:
                    orbit.add(y)
                    dq.append(y)
        for y in orbit:
            seen[y] = mu
        mu_reps.append((mu, len(orbit)))
    print('mu orbits:', len(mu_reps), 'sizes sum', sum(s for _, s in mu_reps))
    types = []
    for mu, osz in mu_reps:
        stab = [g for g in G if act(g, mu) == mu]
        assert len(stab) * osz == 46080
        # orbits of tau under stab
        seen_t = set()
        for tau in all_mu:
            if tau in seen_t:
                continue
            orbit = set()
            dq = deque([tau])
            orbit.add(tau)
            while dq:
                x = dq.popleft()
                for g in stab:
                    y = act(g, x)
                    if y not in orbit:
                        orbit.add(y)
                        dq.append(y)
            seen_t |= orbit
            # validity
            pi = [v ^ 1 for v in range(NL)]
            h = [pi[tau[v]] for v in range(NL)]
            K2 = [[2 - (v == a) - (v == pi[a]) - (v == mu[a]) - (h[v] == a) for v in range(NL)] for a in range(NL)]
            mup = [pi[mu[pi[w]]] for w in range(NL)]
            K3 = [[(1 if h[v] == w else 2) - (v == w) - (h[mu[v]] == w) - (h[v] == mup[w]) for w in range(NL)] for v in range(NL)]
            valid = all(x >= 0 for row in K2 for x in row) and all(x >= 0 for row in K3 for x in row)
            pivots = sum(1 for v in range(NL) if mu[v] == tau[v])
            types.append(dict(mu=list(mu), tau=list(tau), stab=len(stab) // (len(orbit)) if False else None,
                              orbit_size=len(orbit) * osz, pivots=pivots, valid=valid))
    nvalid = sum(1 for t in types if t['valid'])
    print('type classes:', len(types), 'valid:', nvalid, 'total labelled:', sum(t['orbit_size'] for t in types), '(expected 10395^2 =', 10395 ** 2, ')')
    from collections import Counter
    print('valid classes by #pivots:', sorted(Counter(t['pivots'] for t in types if t['valid']).items()))
    json.dump(types, open('types_all.json', 'w'))


if __name__ == '__main__':
    main()
