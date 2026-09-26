"""INDEPENDENT re-check of unified-model certificates (model 'U': one-block AND disjoint squares, types fixed per pair; note 41).
No LP, no code shared with orbit_colmodel_U.py: pair types -> orbit types, free sets, and ALL columns are rebuilt here; each row is
rebuilt from its label with its own right-hand side (stored rhs only compared).
Orbit types from pair types: ('anch',c): both one-block via c; ('tw',): orbit 0 via min block, orbit 1 via max; ('dd',): both dis;
('mix',c): orbit 0 one-block via c, orbit 1 dis; ('sib',): both sister orbits (q[sister] = 2, far block sums 4, never free).
Grid lemma 8.1: every column has exactly one neighbour in each sister pair disjoint from its own pair.  Free at c: dis orbits of pairs containing c, one-block orbits with pivot != c.
Columns:
  one-block Z (pivot i, free j): partner = one-block orbit via i of a pair {i,k}; two distinct free neighbours at j outside the pairs
    containing i, not both in {j,k}; far pairs (avoiding i, j) with block sums 4 - 2[t=k] - n_t, values <= 2; 2 -> both orbits,
    1 -> one orbit.
  dis Z of P = {i,j}: partner = dis orbit of a pair R = {k,l} disjoint from P; loop (sister gets 1, counted at i and at j) only if
    the sister is dis; 2 - loop free neighbours at each of i, j (outside P's pair), never both in {.,k} nor both in {.,l} at the same
    block; far pairs (avoiding i, j) with block sums 4 - n^i_t - n^j_t; R has value 2 + x (x in {0,1}: the partner gets 2, its
    sister x); other far pairs <= 2 as above.
Rows: one, S, C, I (doubled), cut (<= 2(|S|-1), |S| not a union of even cycles >= 4, S inside the free set of c).
Valid: y.b > sum_Z max_c (A^T y)_c + sum_cut max(0, y rhs).
Usage: python orbit_verify_U.py <cert files>"""
import sys, gzip, json, itertools, time
from collections import defaultdict
import numpy as np
N = 7

def rows_of_far(blocks, deg, special=None, spvals=(0, 1, 2)):
    """all {0,1,2}-weightings of the pairs of 'blocks' with block sums deg; the pair 'special' takes values in spvals"""
    prs = list(itertools.combinations(blocks, 2)); out = []
    def go(t, acc, rem):
        if t == len(prs):
            if all(v == 0 for v in rem.values()): out.append(tuple(acc))
            return
        a, b = prs[t]
        vals = spvals if special is not None and {a, b} == set(special) else (0, 1, 2)
        for v in vals:
            if v <= rem[a] and v <= rem[b]:
                rem[a] -= v; rem[b] -= v; go(t + 1, acc + [v], rem); rem[a] += v; rem[b] += v
    go(0, [], dict(deg)); return prs, out

def unions(f):
    res = {0}
    def parts(n, mn):
        if n == 0: yield []; return
        for p in range(max(4, mn), n + 1, 2):
            for r in parts(n - p, p): yield [p] + r
    for pr in parts(f, 4):
        for r in range(len(pr) + 1):
            for c in itertools.combinations(pr, r): res.add(sum(c))
    return res

def check(fn):
    d = json.load(gzip.open(fn, "rt"))
    assert d["model"] == "U"
    ptype = {tuple(int(c) for c in k.split(",")): tuple(v) for k, v in d["spec"]["ptype"].items()}
    forbid = {(int(a), int(b)) for a, b in d["spec"].get("forbid", [])}      # block fixing: forbidden (orbit, partner)
    pair = []; otype = []
    for a in range(N):
        for b in range(a + 1, N):
            t = ptype[(a, b)]
            for s in (0, 1):
                pair.append(frozenset((a, b)))
                if t[0] == 'anch': otype.append(('one', int(t[1])))
                elif t[0] == 'tw': otype.append(('one', (a, b)[s]))
                elif t[0] == 'dd': otype.append(('dis',))
                elif t[0] == 'mix': otype.append(('one', int(t[1])) if s == 0 else ('dis',))
                elif t[0] == 'sib': otype.append(('sib',))
                else: raise ValueError(t)
    n = 42
    orbs = defaultdict(list)
    for s in range(n): orbs[pair[s]].append(s)
    sis = {s: [u for u in orbs[pair[s]] if u != s][0] for s in range(n)}
    freeat = {c: {s for s in range(n) if c in pair[s] and (otype[s][0] == 'dis' or (otype[s][0] == 'one' and otype[s][1] != c))} for c in range(N)}
    sibp = [frozenset(pair[s]) for s in range(0, n, 2) if otype[s] == ('sib',)]      # sister pairs: never free
    Sh = lambda a, b: 2 if pair[a] == pair[b] else len(pair[a] & pair[b])
    one = {}; S = defaultdict(int); C = defaultdict(int); I = defaultdict(int); cuts = []
    yb = 0; mism = 0; extra = 0
    for lab, kind, rhs_s, y in d["rows"]:
        nm = eval(lab, {"frozenset": frozenset})
        if nm[0] == 'one': rhs = 1; one[nm[1]] = one.get(nm[1], 0) + y
        elif nm[0] == 'S': rhs = 12; S[nm[1]] += y
        elif nm[0] == 'C': rhs = 0; C[(nm[1], nm[2])] += y
        elif nm[0] == 'I':
            a, b = nm[1], nm[2]; assert a <= b; rhs = 2 * (12 * (a == b) + 4 - Sh(a, b)); I[(a, b)] += y
        elif nm[0] == 'cut':
            c, Sset = nm[1], tuple(nm[2])
            assert set(Sset) <= freeat[c] and len(Sset) not in unions(len(freeat[c])) and len(set(Sset)) == len(Sset) >= 3 and kind == 'le'
            rhs = 2 * (len(Sset) - 1); cuts.append((c, frozenset(Sset), y)); extra += max(0, y * rhs)
        else: raise ValueError(nm)
        mism += rhs != rhs_s; yb += y * rhs
    Y = np.zeros((n, n), dtype=np.int64)
    for (a, b), y in I.items():
        if a == b: Y[a, a] += 2 * y
        else: Y[a, b] += y; Y[b, a] += y
    high = extra; ncols = 0
    for Z in range(n):
        lin = np.zeros(n, dtype=np.int64)
        for W, y in S.items(): lin[W] += y
        for (a, b), y in C.items():
            if a == Z: lin[b] += y
            if b == Z: lin[a] -= y
        for (a, b), y in I.items():
            if a != b and a == Z: lin[b] += y
            if a != b and b == Z: lin[a] += y
        for c, Sset, y in cuts:
            if Z in Sset:
                for W in Sset:
                    if W != Z: lin[W] += y
        const = one.get(Z, 0); vecs = []
        other = lambda F, blk: [x for x in pair[F] if x != blk][0]
        a, b = sorted(pair[Z])
        if otype[Z][0] == 'one':
            i = otype[Z][1]; j = a + b - i
            cand = sorted(F for F in freeat[j] if F != Z and i not in pair[F])
            far_b = [t for t in range(N) if t not in (i, j)]
            for k in far_b:
                for Dz in orbs[frozenset((i, k))]:
                    if otype[Dz] != ('one', i) or (Z, Dz) in forbid: continue
                    for F1, F2 in itertools.combinations(cand, 2):
                        if other(F1, j) == k and other(F2, j) == k: continue
                        cnt = defaultdict(int); cnt[other(F1, j)] += 1; cnt[other(F2, j)] += 1
                        deg = {t: 4 - (2 if t == k else 0) - cnt[t] for t in far_b}
                        if min(deg.values()) < 0: continue
                        prs, rows = rows_of_far(far_b, deg)
                        base = np.zeros(n, dtype=np.int64); base[Dz] = 2; base[F1] += 1; base[F2] += 1
                        for r in rows:
                            v = base.copy(); ones = []
                            for (p, q), val in zip(prs, r):
                                o = orbs[frozenset((p, q))]
                                if val == 2: v[o[0]] += 1; v[o[1]] += 1
                                elif val == 1: ones.append(o)
                            for ch in itertools.product(*ones):
                                w = v.copy()
                                for o in ch: w[o] += 1
                                vecs.append(w)
        elif otype[Z] == ('sib',):
            # sister orbit: its sister carries its 2 neighbours at i and at j (n_s = 2); far pairs with block sums 4
            far_b = [t for t in range(N) if t not in (a, b)]
            prs, rows = rows_of_far(far_b, {t: 4 for t in far_b})
            base = np.zeros(n, dtype=np.int64); base[sis[Z]] = 2
            for r in rows:
                v = base.copy(); ones = []
                for (p, q), val in zip(prs, r):
                    o = orbs[frozenset((p, q))]
                    if val == 2: v[o[0]] += 1; v[o[1]] += 1
                    elif val == 1: ones.append(o)
                for ch in itertools.product(*ones):
                    w = v.copy()
                    for o in ch: w[o] += 1
                    vecs.append(w)
        else:
            i, j = a, b; Zs = sis[Z]; far_b = [t for t in range(N) if t not in (i, j)]
            ci = sorted(F for F in freeat[i] if F not in (Z, Zs) and j not in pair[F])
            cj = sorted(F for F in freeat[j] if F not in (Z, Zs) and i not in pair[F])
            loops = (0, 1) if otype[Zs] == ('dis',) else (0,)
            for Dz in range(n):
                if otype[Dz] != ('dis',) or (pair[Dz] & pair[Z]) or (Z, Dz) in forbid: continue
                k, l = sorted(pair[Dz]); Wp = sis[Dz]
                for loop in loops:
                    ni = 2 - loop
                    for Fi in itertools.combinations(ci, ni):
                        ri = [other(F, i) for F in Fi]
                        if ni == 2 and ri[0] == ri[1] and ri[0] in (k, l): continue
                        for Fj in itertools.combinations(cj, ni):
                            rj = [other(F, j) for F in Fj]
                            if ni == 2 and rj[0] == rj[1] and rj[0] in (k, l): continue
                            cnt = defaultdict(int)
                            for r_ in ri + rj: cnt[r_] += 1
                            deg = {t: 4 - cnt[t] for t in far_b}
                            if min(deg.values()) < 0: continue
                            prs, rows = rows_of_far(far_b, deg, special=(k, l), spvals=(2, 3))
                            base = np.zeros(n, dtype=np.int64)
                            for F in Fi + Fj: base[F] += 1
                            if loop: base[Zs] += 1
                            for r in rows:
                                v = base.copy(); ones = []
                                for (p, q), val in zip(prs, r):
                                    o = orbs[frozenset((p, q))]
                                    if {p, q} == {k, l}: v[Dz] += 2; v[Wp] += val - 2
                                    elif val == 2: v[o[0]] += 1; v[o[1]] += 1
                                    elif val == 1: ones.append(o)
                                for ch in itertools.product(*ones):
                                    w = v.copy()
                                    for o in ch: w[o] += 1
                                    vecs.append(w)
        M = np.array(vecs, dtype=np.int64); assert len(M) and (M.sum(axis=1) == 12).all()
        for R in sibp:                          # grid lemma 8.1: exactly one neighbour in every sister pair disjoint from P_Z
            if not (R & pair[Z]):
                o1, o2 = orbs[R]; M = M[(M[:, o1] + M[:, o2]) == 1]
        assert len(M), Z
        sc = const + M @ lin + np.einsum('ij,ij->i', M @ Y, M)
        ncols += len(M); high += int(sc.max())
    return yb, high, mism, ncols

if __name__ == "__main__":
    allok = True
    for fn in sys.argv[1:]:
        t0 = time.time(); yb, high, mism, ncols = check(fn); ok = yb > high and mism == 0; allok &= ok
        print("%s: y.b %d bound %d margin %.4f rhs mismatches %d columns %d %s %.0fs" % (fn.replace('\\', '/').split('/')[-1], yb, high, (yb - high) / 1e6, mism, ncols, "VALID" if ok else "INVALID", time.time() - t0)); sys.stdout.flush()
    print("ALL VALID" if allok else "SOME INVALID")
