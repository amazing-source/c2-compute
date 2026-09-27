"""
Control for the "core" statement (mu_c = tau_c = pi at every line through the fixed vertex) on the one known
member of the family: BvLS = srg(243,22,1,2) with the involution t = -1 (unique fixed vertex 0).

For every line c through 0 this prints the number of pivot labels (mu_c(v) = tau_c(v)), the number of labels with
mu_c(v) = pi(v), tau_c(v) = pi(v), and the cycle types of pi*tau and mu*tau.  Result: at all 11 lines all 20 labels
are pivots and mu_c = tau_c = pi, i.e. every t-invariant square of BvLS is of type "sib" (lies in a plane through 0).

Also prints the pivot <-> partner-line correspondence check: label v of star_c is a pivot iff the partner orbit of
{+-v} (the other diagonal of the t-invariant square through v, -v) lies on a plane containing line c.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inv import Config, bvls_in_labelling
from analyze import block_type


def cycles(perm, dom):
    seen, out = set(), []
    for v in dom:
        if v in seen:
            continue
        L, w = 0, v
        while w not in seen:
            seen.add(w); w = perm[w]; L += 1
        out.append(L)
    return sorted(out)


def partner_pairs(cfg, A):
    """For each D vertex v: the antipodal pair {a,-a} = N(v) cap N(-v) (unique, since t swaps the two common nbrs)."""
    P = {}
    for v in cfg.V:
        if v[0] != 'D':
            continue
        iv, itv = cfg.vid[v], cfg.vid[cfg.t(v)]
        common = [w for w in cfg.V if A[iv, cfg.vid[w]] and A[itv, cfg.vid[w]]]
        assert len(common) == 2 and common[1] == cfg.t(common[0]) or common[0] == cfg.t(common[1]), (v, common)
        P[v] = common[0]
    return P


def main():
    cfg0, A = bvls_in_labelling()
    m = cfg0.m
    P = partner_pairs(cfg0, A)
    # type of each orbit: sib / one / dis = number of lines shared by v and its partner
    def lines(v):
        return {v[1], v[3]}
    n = {2: 0, 1: 0, 0: 0}
    for v in cfg0.V:
        if v[0] == 'D':
            n[len(lines(v) & lines(P[v]))] += 1
    print("BvLS m=%d: D vertices by partner type  sib=%d one=%d dis=%d" % (m, n[2], n[1], n[0]))
    for c in range(m):
        cfg = Config(m, c)
        mu, tau = block_type(cfg, A)
        piv = [v for v in cfg.labels if mu[v] == tau[v]]
        mupi = sum(1 for v in cfg.labels if mu[v] == cfg.pi(v))
        taupi = sum(1 for v in cfg.labels if tau[v] == cfg.pi(v))
        h = {v: cfg.pi(tau[v]) for v in cfg.labels}
        g = {v: mu[tau[v]] for v in cfg.labels}
        # pivot <-> partner on line c
        ok = all((v in piv) == (c in lines(P[cfg.star_p(v)])) for v in cfg.labels)
        print("line %2d: pivots %2d/%d  mu=pi on %2d  tau=pi on %2d  cycles(pi*tau) %s  cycles(mu*tau) %s  pivot<->partner-on-line %s"
              % (c, len(piv), cfg.nlab, mupi, taupi, cycles(h, cfg.labels), cycles(g, cfg.labels), ok))


if __name__ == '__main__':
    main()
