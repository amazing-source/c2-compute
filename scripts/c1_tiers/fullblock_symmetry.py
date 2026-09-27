"""Symmetry breaking for the full-block local system (structures' involution/sat_local.py), C1 27/09.
Full block: mu = tau = m (from the D-cycles, gauge of fullblock_rigid.mates_from_D).  The model ((i)-(v), no cancellation,
pointwise L1/S1/S2) is defined from pi (block flip) and m only, so it is invariant under
    G = Stab_{B6}(m) = { label permutations g commuting with pi, with g m g^-1 = m },
acting by Phi -> g Phi g^-1, i.e. on the variables X(z, e) -> X(g z, g e).
(The swap c+ <-> c- acts as Phi -> pi Phi pi: it is the element pi of B6, nothing new.)
|G| = 12 (6-cycle), 32 (2+4), 72 (3+3), 384 (2+2+2).
Breaking chain (sound): fix a sequence of far roots z_0, z_1, ...; at level k, among the values Phi(z_k) allowed so far, keep one
representative per orbit of H_k = Stab_G(z_0..z_k; e_0..e_{k-1}) (pointwise stabiliser of the previous choices and of z_k).
Clauses: for every prefix (e_0..e_{k-1}) of representatives,  -X(z_0,e_0) v ... v -X(z_{k-1},e_{k-1}) v OR_{e rep} X(z_k, e).
Indexing exactly as sat_local.encode: far = combinations of labels in different blocks (lexicographic), edges = non-m pairs."""
import itertools, json, sys

L = list(range(12))
PI = [v ^ 1 for v in L]


def mates_from_D(cycles):
    m = [None] * 12
    for cyc in cycles:
        l = len(cyc)
        for j in range(l):
            p, q = cyc[j], cyc[(j + 1) % l]
            a, b = 2 * p, 2 * q + 1
            m[a], m[b] = b, a
    assert all(x is not None for x in m) and all(m[m[v]] == v for v in L) and all(m[v] != PI[v] for v in L)
    return m


def b6():
    for perm in itertools.permutations(range(6)):
        for flips in range(64):
            yield tuple(2 * perm[i // 2] + ((i % 2) ^ ((flips >> (i // 2)) & 1)) for i in L)


def group(m):
    return [g for g in b6() if all(g[m[v]] == m[g[v]] for v in L)]


def indexing(m):
    far = [z for z in itertools.combinations(L, 2) if z[0] // 2 != z[1] // 2]
    edges = [e for e in itertools.combinations(L, 2) if m[e[0]] != e[1]]
    return far, {z: i for i, z in enumerate(far)}, edges, {e: i for i, e in enumerate(edges)}


def actions(m, G):
    far, fidx, edges, eidx = indexing(m)
    act = []
    for g in G:
        fz = [fidx[tuple(sorted((g[a], g[b])))] for (a, b) in far]
        fe = [eidx[tuple(sorted((g[a], g[b])))] for (a, b) in edges]
        act.append((fz, fe))
    return act


def breaking_clauses(m, cycles, depth=3, X=None, roots=None):
    """list of clauses over X(z, e) (1-based literal ids given by X(z, e); default: tuples ('x', z, e) for inspection).
    roots: the sequence z_0, z_1, ... (default: 0, 1, 2, ...)."""
    G = group(m)
    act = actions(m, G)
    X = X or (lambda z, e: ('x', z, e))
    roots = roots or list(range(depth))
    clauses = []
    stats = []

    def rec(level, H, prefix):
        if level == depth:
            return
        z = roots[level]
        Hz = [h for h in H if act[h][0][z] == z]
        seen, reps = set(), []
        for e in range(60):
            if e in seen:
                continue
            orb = {act[h][1][e] for h in Hz}
            seen |= orb
            reps.append(e)
        stats.append((level, len(H), len(Hz), len(reps)))
        neg = [('neg', X(zz, ee)) for zz, ee in prefix]
        clauses.append(neg + [('pos', X(z, e)) for e in reps])
        for e in reps:
            He = [h for h in Hz if act[h][1][e] == e]
            if len(He) > 1:
                rec(level + 1, He, prefix + [(z, e)])
    rec(0, list(range(len(G))), [])
    return G, clauses, stats


if __name__ == '__main__':
    types = {'6cyc': [(0, 1, 2, 3, 4, 5)], '2+4': [(0, 1), (2, 3, 4, 5)], '3+3': [(0, 1, 2), (3, 4, 5)],
             '2+2+2': [(0, 1), (2, 3), (4, 5)]}
    out = {}
    for name, cyc in types.items():
        m = mates_from_D(cyc)
        G, clauses, stats = breaking_clauses(m, cyc, depth=4)
        far, fidx, edges, eidx = indexing(m)
        # sanity: the group preserves pi, m, the D-cycles (as block sets) and K2/K3 (hence (ii), (iii))
        cyc_sets = {frozenset(c) for c in cyc}
        for g in G:
            assert all(g[PI[v]] == PI[g[v]] for v in L)
            assert {frozenset(g[2 * p] // 2 for p in c) for c in cyc} == cyc_sets
        first = stats[0]
        print('%-6s |G| = %3d; level 0: |Stab(z0)| = %d, values of Phi(z0) kept %d / 60; breaking clauses %d (depth 4)'
              % (name, len(G), first[2], first[3], len(clauses)))
        out[name] = {'cycles': [list(c) for c in cyc], 'm': m, 'group': [list(g) for g in G],
                     'far': [list(z) for z in far], 'edges': [list(e) for e in edges],
                     'action_far': [a[0] for a in actions(m, G)], 'action_edges': [a[1] for a in actions(m, G)]}
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], 'w'))
        print('groups and actions written to', sys.argv[1])


def lexleader_clauses(m, X, newvar, prefix_roots=10, G=None):
    """Lex-leader symmetry breaking (sound): for every g in G, X <=_lex X o g, where (X o g)(z, e) = X(g z, g e) and the
    variable order is (z, e) lexicographic over the first prefix_roots far roots.  X(z, e) -> literal id (positive int);
    newvar() -> fresh id.  Encoding per g (e_0 = true, positions with the same variable on both sides skipped):
        (-e_{i-1} v -x_i v y_i),  (-e_{i-1} v -x_i v -y_i v e_i),  (-e_{i-1} v x_i v y_i v e_i).
    Any G-orbit of solutions contains its lex-minimum, which satisfies all clauses; so SAT is preserved."""
    G = G or group(m)
    act = actions(m, G)
    order = [(z, e) for z in range(prefix_roots) for e in range(60)]
    clauses = []
    for (fz, fe) in act:
        if all(fz[z] == z for z in range(60)) and all(fe[e] == e for e in range(60)):
            continue                                   # identity
        prev = None                                    # None = true
        for (z, e) in order:
            x, y = X(z, e), X(fz[z], fe[e])
            if x == y:
                continue
            cur = newvar()
            pre = [] if prev is None else [-prev]
            clauses.append(pre + [-x, y])
            clauses.append(pre + [-x, -y, cur])
            clauses.append(pre + [x, y, cur])
            prev = cur
    return clauses
