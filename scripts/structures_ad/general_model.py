"""General pair-level model for a FULL skeleton of the involution case with Sigma = empty (no sibling squares).

A skeleton lists the 42 orbits, each with its pair P (2 blocks), its pivot (the block through which it is paired, i.e. its D-square
is 'one block' via that block) or None (dis square), and the index of its D-partner.  Everything imposed below is proved:
  (F1)    for t in P_W: sum_{P' contains t} phi_W(P') = 2  (own pair counted with phi_W(P_W) = n_s(P_W));
  (piv)   if W pivots at t: its two through-t neighbours are the two roots of D(W): phi_W(t y) = 2 (D(W) in ty), 0 on the
          other pairs through t (own pair included, so n_s = 0);
  (free)  if W does not pivot at t: its through-t neighbours lie in two DIFFERENT orbits that do not pivot at t (digons of R_t are
          exactly the D-pairs via t), so phi_W(tm) <= #{orbits of tm not pivoting at t}  (own pair: n_s);
  (bal)   balance (note 23 par. 8) at every block k not in P_W, on every union of components of G_k that is CERTAIN: each closed
          D-class at k (connected component of the D-graph at k all of whose pairs have BOTH orbits pivoting at k), and the
          complement of their union:  #(interiors of w in S) + sum_{m in S} phi_W(km) = |S|;
  (dis)   dis orbit: phi_W(pair of D(W)) in {2, 3};  n_s(P) can be 1 only if both orbits of P are dis (39.3), else 0;
  (sym)   sum over the 2 orbits of P of phi(P') = same with P, P' exchanged (= e(P,P')/2);
  (dich)  pair dichotomy = mu-budget 16.1: for every pair P,
          #{Z not in P, Z not a D-partner of an orbit of P : phi_Z(P) = 2} + 2 #{Z : phi_Z(P) = 3} = 2 - n_s(P),
          phi_Z(P) = 3 only if Z is the D-partner of an orbit of P and that square is dis ('one block' partners are (2,0)).
Feasibility is a small integer program (scipy milp / HiGHS), used as a finite test; infeasible subsystems are then written by
hand (single_orbit_proof-style) when they involve one orbit."""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 7
PAIRS = [(a, b) for a in range(N) for b in range(a + 1, N)]


def pr(a, b):
    return (a, b) if a < b else (b, a)


def check_skeleton(orbs):
    """orbs: list of dicts {pair, pivot, partner}.  Validates the combinatorics of D."""
    assert len(orbs) == 42
    cnt = {}
    for i, o in enumerate(orbs):
        cnt[o['pair']] = cnt.get(o['pair'], 0) + 1
        j = o['partner']
        if j is None:
            assert o['pivot'] is not None
            continue
        assert orbs[j]['partner'] == i and j != i
        P, Q = set(o['pair']), set(orbs[j]['pair'])
        if o['pivot'] is None:
            assert orbs[j]['pivot'] is None and not (P & Q), (i, o, orbs[j])
        else:
            assert orbs[j]['pivot'] == o['pivot'] and P & Q == {o['pivot']}, (i, o, orbs[j])
    assert all(cnt.get(P) == 2 for P in PAIRS)


def closed_classes(orbs, k):
    """closed D-classes at block k (sets of blocks), and the complement."""
    piv = {}
    for i, o in enumerate(orbs):
        if k in o['pair']:
            m = o['pair'][0] + o['pair'][1] - k
            piv.setdefault(m, []).append(o['pivot'] == k)
    adj = {m: set() for m in range(N) if m != k}
    freeset = set()
    for i, o in enumerate(orbs):
        if o['pivot'] == k and o['partner'] is None:
            freeset.add(o['pair'][0] + o['pair'][1] - k)
    for i, o in enumerate(orbs):
        if o['pivot'] == k and o['partner'] is not None:
            m = o['pair'][0] + o['pair'][1] - k
            q = orbs[o['partner']]['pair']
            m2 = q[0] + q[1] - k
            adj[m].add(m2)
            adj[m2].add(m)
    fl = sorted(freeset)
    for a_, b_ in zip(fl, fl[1:]):          # free partners match among themselves: their blocks form a union of classes
        adj[a_].add(b_)
        adj[b_].add(a_)
    seen = set()
    closed = []
    for m in adj:
        if m in seen or not any(piv[m]):
            continue
        comp = {m}
        stack = [m]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp.add(y)
                    stack.append(y)
        seen |= comp
        if all(all(piv[x]) for x in comp):
            closed.append(comp)
    rest = set(range(N)) - {k} - set().union(*closed) if closed else set(range(N)) - {k}
    return closed, rest


def build(orbs, dichotomy=True, balance=True, near=None):
    """near (optional): {t: {orbit index i through t: (j1, j2)}}: the two R_t-neighbours (orbit indices) of every orbit through
    t that is FREE at t (not pivoting at t).  Then phi_i(tm) is exact for those orbits, and the components of G_t are known."""
    check_skeleton(orbs)
    n_orb = len(orbs)
    PI = {P: i for i, P in enumerate(PAIRS)}
    by_pair = {P: [i for i, o in enumerate(orbs) if o['pair'] == P] for P in PAIRS}
    both_dis = {P: all(orbs[i]['pivot'] is None for i in by_pair[P]) for P in PAIRS}
    cnt = [0]

    def new(n=1):
        a = cnt[0]
        cnt[0] += n
        return a
    phi = {(i, P): new() for i in range(n_orb) for P in PAIRS}
    ns = {P: new() for P in PAIRS if both_dis[P]}
    y = {}
    if dichotomy:
        for i in range(n_orb):
            for P in PAIRS:
                if P != orbs[i]['pair']:
                    y[(i, P)] = new(4)
    xv = {}                                   # free-partner matching variables x[i,j] (i<j), same pivot, different pairs
    freeorb = [i for i, o in enumerate(orbs) if o['partner'] is None]
    for i, j in itertools.combinations(freeorb, 2):
        if orbs[i]['pivot'] == orbs[j]['pivot'] and orbs[i]['pair'] != orbs[j]['pair']:
            xv[(i, j)] = new()
    nvar = cnt[0]
    lb = np.zeros(nvar)
    ub = np.full(nvar, 3.0)
    for key in xv.values():
        ub[key] = 1
    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        rows.append(coefs)
        lo.append(l)
        hi.append(h)
    comps = {k: closed_classes(orbs, k) for k in range(N)}
    if near is not None:
        for k, nbk in near.items():
            # union-find on blocks: an R_k edge between orbits of pairs km, km' joins m and m'; closed classes stay
            par = {m: m for m in range(N) if m != k}
            def f(a):
                while par[a] != a:
                    a = par[a]
                return a
            for i, nb in nbk.items():
                m = orbs[i]['pair'][0] + orbs[i]['pair'][1] - k
                for j2 in nb:
                    m2 = orbs[j2]['pair'][0] + orbs[j2]['pair'][1] - k
                    par[f(m)] = f(m2)
            for i, o in enumerate(orbs):          # D-edges at k (digons of R_k) also join blocks
                if o['pivot'] == k and o['partner'] is not None:
                    m = o['pair'][0] + o['pair'][1] - k
                    q = orbs[o['partner']]['pair']
                    par[f(m)] = f(q[0] + q[1] - k)
            closed, rest = comps[k]
            groups = {}
            for m in rest:
                groups.setdefault(f(m), set()).add(m)
            comps[k] = (closed + list(groups.values()), set())
    for i, o in enumerate(orbs):
        P = o['pair']
        a, b = P
        if P in ns:
            add([(phi[(i, P)], 1), (ns[P], -1)], 0, 0)
        else:
            ub[phi[(i, P)]] = 0
        for t in (a, b):
            add([(phi[(i, pr(t, m))], 1) for m in range(N) if m != t], 2, 2)
            if o['pivot'] == t and o['partner'] is None:
                xs = {j: xv[pr(i, j)] for j in freeorb if pr(i, j) in xv}
                add([(v, 1) for v in xs.values()], 1, 1)
                for m in range(N):
                    if m == t:
                        continue
                    Q = pr(t, m)
                    js = [v for j, v in xs.items() if orbs[j]['pair'] == Q]
                    if js:
                        add([(phi[(i, Q)], 1)] + [(v, -2) for v in js], 0, 0)
                    else:
                        ub[phi[(i, Q)]] = 0
            elif o['pivot'] == t:
                q = orbs[o['partner']]['pair']
                for m in range(N):
                    if m == t:
                        continue
                    j = phi[(i, pr(t, m))]
                    if pr(t, m) == q:
                        lb[j] = ub[j] = 2
                    else:
                        ub[j] = 0
            else:
                for m in range(N):
                    if m == t or pr(t, m) == P:
                        continue
                    cap = sum(1 for j2 in by_pair[pr(t, m)] if orbs[j2]['pivot'] != t)
                    ub[phi[(i, pr(t, m))]] = min(ub[phi[(i, pr(t, m))]], cap)
                if near is not None and t in near and i in near[t]:
                    nb = near[t][i]
                    for m in range(N):
                        if m == t:
                            continue
                        Q = pr(t, m)
                        val = sum(1 for j2 in nb if orbs[j2]['pair'] == Q)
                        j = phi[(i, Q)]
                        if Q == P:
                            add([(j, 1)], val, val)
                        else:
                            lb[j] = ub[j] = val
        if balance:
            for k in range(N):
                if k in P:
                    continue
                closed, rest = comps[k]
                for S in closed + ([rest] if rest else []):
                    add([(phi[(i, pr(k, m))], 1) for m in S], len(S - {a, b}), len(S - {a, b}))
        if o['pivot'] is None:
            j = phi[(i, orbs[o['partner']]['pair'])]
            lb[j] = max(lb[j], 2)
    for P, P2 in itertools.combinations(PAIRS, 2):
        add([(phi[(i, P2)], 1) for i in by_pair[P]] + [(phi[(i, P)], -1) for i in by_pair[P2]], 0, 0)
    for P in ns:
        ub[ns[P]] = 1
    if dichotomy:
        for (i, P), yy in y.items():
            for f in range(4):
                ub[yy + f] = 1
            add([(yy + f, 1) for f in range(4)], 1, 1)
            add([(phi[(i, P)], 1)] + [(yy + f, -f) for f in range(4)], 0, 0)
            if orbs[i]['partner'] is None:
                ub[yy + 3] = 0
                continue
            is_partner = orbs[orbs[i]['partner']]['pair'] == P
            if not (is_partner and orbs[i]['pivot'] is None):
                ub[yy + 3] = 0
        for P in PAIRS:
            coefs = []
            const = 0
            partners = {orbs[j]['partner'] for j in by_pair[P] if orbs[j]['partner'] is not None}
            for i in range(n_orb):
                if orbs[i]['pair'] == P:
                    continue
                yy = y[(i, P)]
                if orbs[i]['partner'] is None:
                    # one-block free partner: counts 1 if phi = 2 and D(i) not in P;  D(i) in P  <=>  sum of x[i, j in P] = 1
                    coefs.append((yy + 2, 1))
                    for j in by_pair[P]:
                        if pr(i, j) in xv:
                            coefs.append((xv[pr(i, j)], -1))
                    continue
                if i not in partners:
                    coefs.append((yy + 2, 1))
                coefs.append((yy + 3, 2))
            if P in ns:
                coefs.append((ns[P], 1))
            add(coefs, 2 + const, 2 + const)
    import scipy.sparse as sp
    ri, ci, vv = [], [], []
    for r, cs in enumerate(rows):
        for j, c in cs:
            ri.append(r)
            ci.append(j)
            vv.append(c)
    A = sp.csr_matrix((np.array(vv, float), (np.array(ri), np.array(ci))), shape=(len(rows), nvar))
    A.sum_duplicates()
    return A, np.array(lo, float), np.array(hi, float), lb, ub, nvar, phi, ns


def feasible(orbs, dichotomy=True, balance=True, time_limit=120, near=None):
    A, lo, hi, lb, ub, nvar, phi, ns = build(orbs, dichotomy, balance, near)
    if np.any(lb > ub):
        return None
    res = milp(c=np.zeros(nvar), constraints=[LinearConstraint(A, lo, hi)], integrality=np.ones(nvar),
               bounds=Bounds(lb, ub), options={'time_limit': time_limit})
    if res.status == 0:
        return res.x, phi, ns
    if res.status == 2:
        return None
    return 'unknown(%d)' % res.status


def from_c2_key(key):
    """C2 leaf key: list of 42 triples (pivot, free, third): the orbit of pair {pivot, free}, paired via pivot with an orbit of
    pair {pivot, third}.  Partner of (c,x,y) = a triple (c,y,x); duplicates (digons) are matched arbitrarily (WLOG)."""
    trip = [tuple(t) for t in key]
    orbs = [{'pair': pr(c, x), 'pivot': c, 'partner': None} for c, x, y in trip]
    used = set()
    for i, (c, x, y) in enumerate(trip):
        if i in used:
            continue
        for j, t2 in enumerate(trip):
            if j != i and j not in used and t2 == (c, y, x):
                orbs[i]['partner'] = j
                orbs[j]['partner'] = i
                used |= {i, j}
                break
        assert i in used, (i, trip[i])
    return orbs


def from_ad(In, dd, Ds, ddm):
    """AD skeleton -> orbit list (for cross-checking against pair_level.py)."""
    from pair_level import cycle_matching
    idx = {}
    orbs = []
    for P in PAIRS:
        for s in (0, 1):
            idx[(P, s)] = len(orbs)
            orbs.append({'pair': P, 'pivot': None, 'partner': None})
    anchor = {}
    for k in range(N):
        for m in In[k]:
            anchor[pr(k, m)] = k
    D = {}
    for k, cycles in Ds.items():
        for cyc in cycles:
            for (m, s), (m2, s2) in cycle_matching(cyc).items():
                D[(pr(k, m), s)] = (pr(k, m2), s2)
    D.update(ddm)
    for (P, s), j in idx.items():
        orbs[j]['pivot'] = anchor.get(P)
        orbs[j]['partner'] = idx[D[(P, s)]]
    return orbs
