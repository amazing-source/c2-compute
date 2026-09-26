"""INDEPENDENT re-check of the orbit-level certificates (note 40), no LP, no code shared with orbit_colmodel.py beyond the stored
triples and multipliers.  Every row is rebuilt from its LABEL with its own right-hand side (stored rhs only compared); columns are
re-enumerated with an own far-row enumerator and an own orbit expansion; (A^T y)_c is computed from the row SEMANTICS:
  one[Z]: 1 ;  S[W]: q(W) ;  C[a,b]: q(b) if Z = a, -q(a) if Z = b ;  I[a,b]: 2 q(a) q(b) + [Z = a] q(b) + [Z = b] q(a) ;
  cut[c,S]: sum_{W in S, W != Z} q(W) if Z in S.
Valid certificate:  y^T b  >  sum_Z max_c (A^T y)_c + sum_{cut rows} max(0, y_r rhs_r)   (cut slacks lie in [0, rhs]).
Usage: python orbit_verify_certs.py <cert .json.gz files>"""
import sys, gzip, json, itertools, time
from collections import defaultdict
import numpy as np
N = 7

def far_rows(blocks, deg):
    prs = list(itertools.combinations(blocks, 2)); out = []
    def go(t, acc, rem):
        if t == len(prs):
            if not any(rem.values()): out.append(tuple(acc))
            return
        a, b = prs[t]
        for v in range(min(2, rem[a], rem[b]) + 1):
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

def structure(d):
    """own reconstruction of the orbit labelling and, per orbit, the list of (pivot i, free j, third k, partner Dz) options and
    the free candidates; freeat[c] (None if the free sets are variable: no cut rows allowed)"""
    model = d.get("model", "fixed")
    if model == "fixed" or model == "var":
        if model == "fixed": T = [tuple(t) for t in d["triples"]]
        else:
            A = d["spec"]["A"]; T = [tuple(t) for t in d["spec"]["leaf"]] + [(A, x, None) for x in range(N) if x != A for s in (0, 1)]
        pair = [frozenset(t[:2]) for t in T]
        piv = [t[0] for t in T]
    else:
        pair = []; piv = []
        anch = {}
        if model == "gen":
            for c, xs in d["spec"]["In"].items():
                for x in xs: anch[frozenset((int(c), x))] = int(c)
        for a in range(N):
            for b in range(a + 1, N):
                R = frozenset((a, b))
                for s in (0, 1):
                    pair.append(R)
                    piv.append(None if model == "all" else (anch[R] if R in anch else (a, b)[s]))
    n = len(pair); assert n == 42
    orbs = defaultdict(list)
    for s in range(n): orbs[pair[s]].append(s)
    assert len(orbs) == 21 and all(len(v) == 2 for v in orbs.values())
    sis = {s: [u for u in orbs[pair[s]] if u != s][0] for s in range(n)}
    opts = {}
    if model in ("fixed", "var"):
        nfix = 42 if model == "fixed" else 30
        pos = defaultdict(list)
        for s in range(nfix): pos[T[s]].append(s)
        for s in range(nfix):
            i, j, k = T[s]; opts[s] = [(i, j, k, pos[(i, k, j)][pos[(i, j, k)].index(s)])]
        for s in range(nfix):
            assert opts[opts[s][0][3]][0][3] == s
        for s in range(nfix, n):
            A, x = piv[s], [c for c in pair[s] if c != piv[s]][0]
            opts[s] = [(A, x, k, Dz) for k in range(N) if k not in (A, x) for Dz in orbs[frozenset((A, k))]]
    else:
        for s in range(n):
            opts[s] = []
            for i in (sorted(pair[s]) if piv[s] is None else [piv[s]]):
                j = [c for c in pair[s] if c != i][0]
                for k in range(N):
                    if k in (i, j): continue
                    for Dz in orbs[frozenset((i, k))]:
                        if piv[Dz] is None or piv[Dz] == i: opts[s].append((i, j, k, Dz))
    alw = d.get("spec", {}).get("allowed")
    if alw is not None:                                   # union LPs: thirds restricted to the filter's leaves (+ A free)
        alw = {tuple(int(c) for c in key.split(",")): set(v) for key, v in alw.items()}
        for s in range(n): opts[s] = [o for o in opts[s] if o[2] in alw.get((o[0], o[1]), set())]
    fb = d.get("spec", {}).get("forbid")
    if fb:                                                # branch nodes: forbidden (orbit, partner) choices
        fb = {(int(a), int(b)) for a, b in fb}
        for s in range(n): opts[s] = [o for o in opts[s] if (s, o[3]) not in fb]
    if model == "all":
        freeat = None
        cand = lambda Z, i, j: sorted(F for F in range(n) if j in pair[F] and i not in pair[F] and F != Z and F != sis[Z])
    else:
        freeat = {c: {s for s in range(n) if c in pair[s] and piv[s] != c} for c in range(N)}
        cand = lambda Z, i, j: sorted(F for F in freeat[j] if F != Z and F != sis[Z])
    return pair, orbs, sis, opts, freeat, cand

def check(fn):
    d = json.load(gzip.open(fn, "rt"))
    pair, orbs, sis, opts, freeat, cand = structure(d); n = 42
    def Sh(a, b): return 2 if pair[a] == pair[b] else len(pair[a] & pair[b])
    one = {}; S = defaultdict(int); C = defaultdict(int); I = defaultdict(int); cuts = []; sis5 = []
    yb = 0; mism = 0; extra = 0
    for lab, kind, rhs_s, y in d["rows"]:
        nm = eval(lab, {"frozenset": frozenset})
        if nm[0] == 'one': rhs = 1; one[nm[1]] = one.get(nm[1], 0) + y
        elif nm[0] == 'S': rhs = 12; S[nm[1]] += y
        elif nm[0] == 'C': rhs = 0; C[(nm[1], nm[2])] += y
        elif nm[0] == 'I':
            a, b = nm[1], nm[2]; assert a <= b; rhs = 2 * (12 * (a == b) + 4 - Sh(a, b)); I[(a, b)] += y
        elif nm[0] == 'sis5':
            c, U, Up = nm[1], nm[2], nm[3]
            assert freeat is not None and U in freeat[c] and Up in freeat[c] and pair[U] == pair[Up] and U != Up and kind == 'le'
            rhs = 1; sis5.append((c, U, Up, y)); extra += max(0, y * rhs)
        elif nm[0] == 'cut':
            c, Sset = nm[1], tuple(nm[2]); assert freeat is not None and set(Sset) <= freeat[c] and len(Sset) not in unions(len(freeat[c])) and len(set(Sset)) == len(Sset) >= 3
            rhs = 2 * (len(Sset) - 1); cuts.append((c, frozenset(Sset), y))
            extra += max(0, y * rhs)
            assert kind == 'le'
        else: raise ValueError(nm)
        mism += rhs != rhs_s
        yb += y * rhs
    # quadratic form and linear parts per orbit
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
            if a == b: continue
            if a == Z: lin[b] += y
            if b == Z: lin[a] += y
        for c, Sset, y in cuts:
            if Z in Sset:
                for W in Sset:
                    if W != Z: lin[W] += y
        const = one.get(Z, 0)
        # own enumeration of the columns of Z, over all its (pivot, third, partner) options
        best = None
        for (i, j, k, Dz) in opts[Z]:
            cands = cand(Z, i, j)
            other = lambda s: [x for x in pair[s] if x != j][0]
            far_b = [t for t in range(N) if t not in (i, j)]
            for F1, F2 in itertools.combinations(cands, 2):
                if other(F1) == k and other(F2) == k: continue
                cnt = defaultdict(int); cnt[other(F1)] += 1; cnt[other(F2)] += 1
                deg = {t: 4 - (2 if t == k else 0) - cnt[t] for t in far_b}
                if min(deg.values()) < 0: continue
                prs, rows = far_rows(far_b, deg)
                base = np.zeros(n, dtype=np.int64); base[Dz] = 2; base[F1] = 1; base[F2] = 1
                vecs = []
                for r in rows:
                    v = base.copy(); ones = []
                    for (a, b), val in zip(prs, r):
                        o = orbs[frozenset((a, b))]
                        if val == 2: v[o[0]] += 1; v[o[1]] += 1
                        elif val == 1: ones.append(o)
                    for ch in itertools.product(*ones):
                        w = v.copy()
                        for o in ch: w[o] += 1
                        vecs.append(w)
                if not vecs: continue
                M = np.array(vecs, dtype=np.int64); assert (M.sum(axis=1) == 12).all()
                sc = const + M @ lin + np.einsum('ij,ij->i', M @ Y, M)
                for c, U, Up, y in sis5:                  # Lemma 11.5 rows: near doubles of a sister pair free at c
                    if j == c and Z not in (U, Up): sc = sc + y * ((M[:, U] == 1) & (M[:, Up] == 1))
                ncols += len(M); m = int(sc.max()); best = m if best is None else max(best, m)
        high += best
    return yb, high, mism, ncols

if __name__ == "__main__":
    allok = True
    for fn in sys.argv[1:]:
        t0 = time.time(); yb, high, mism, ncols = check(fn); ok = yb > high and mism == 0; allok &= ok
        print("%s: y.b %d  bound %d  margin %.4f (x1e6)  rhs mismatches %d  columns %d  %s  %.0fs" % (fn.split('/')[-1], yb, high, (yb - high) / 1e6, mism, ncols, "VALID" if ok else "INVALID", time.time() - t0)); sys.stdout.flush()
    print("ALL VALID" if allok else "SOME INVALID")

def check_comb(fn):
    """combinatorial kill (model 'comb'): with the D-structure fixed at some blocks, the balance lemma 24.0 on every closed D-class
    S at a fixed block c (sum over m in S of q_Z(orbits of {c,m}) = |S| - |P_Z & S| for Z avoiding c) leaves some orbit with NO
    admissible column.  Re-enumerates that orbit's columns independently (gen semantics + forbid) and re-checks the balance."""
    d = json.load(gzip.open(fn, "rt")); spec = d["spec"]
    dd = {"model": "gen", "spec": {"In": spec["In"], "forbid": spec["forbid"]}}
    pair, orbs, sis, opts, freeat, cand = structure(dd); n = 42
    comps = [(int(c), set(S)) for c, S in spec["comps"]]
    dead = []
    for Z in spec["empty_orbits"]:
        survivors = 0
        for (i, j, k, Dz) in opts[Z]:
            other = lambda s: [x for x in pair[s] if x != j][0]
            far_b = [t for t in range(N) if t not in (i, j)]
            for F1, F2 in itertools.combinations(cand(Z, i, j), 2):
                if other(F1) == k and other(F2) == k: continue
                cnt = defaultdict(int); cnt[other(F1)] += 1; cnt[other(F2)] += 1
                deg = {t: 4 - (2 if t == k else 0) - cnt[t] for t in far_b}
                if min(deg.values()) < 0: continue
                prs, rows = far_rows(far_b, deg)
                base = np.zeros(n, dtype=np.int64); base[Dz] = 2; base[F1] = 1; base[F2] = 1
                for r in rows:
                    v = base.copy(); ones = []
                    for (a, b), val in zip(prs, r):
                        o = orbs[frozenset((a, b))]
                        if val == 2: v[o[0]] += 1; v[o[1]] += 1
                        elif val == 1: ones.append(o)
                    for ch in itertools.product(*ones):
                        w = v.copy()
                        for o in ch: w[o] += 1
                        ok = True
                        for c, S in comps:
                            if c in pair[Z]: continue
                            got = sum(int(w[o]) for m in S for o in orbs[frozenset((c, m))])
                            if got != len(S) - len(pair[Z] & S): ok = False; break
                        if ok: survivors += 1
        dead.append((Z, survivors))
    return dead
