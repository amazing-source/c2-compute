"""
Step 1: enumerate all iota = fixed-point-free involutions on the non-mu pairs with cross counts K3:
    K3[v][w] = #{ p non-mu pair : v in p and pi(w) in iota(p) }.
DFS over the pairs in a fixed order, matching the first unmatched pair with a later unmatched pair,
with exact budget accounting and a coverability bound.
"""
import sys, time
from phisolve import TypeInstance, load_type


class IotaSearch:
    def __init__(self, inst, time_limit=None, verbose=0, max_solutions=None):
        self.inst = inst
        nl = inst.nl
        self.npairs = inst.npairs
        self.pairs = inst.pairs
        self.pi = inst.pi
        self.K3rem = [row[:] for row in inst.K3]
        # pairs containing a given label
        self.pairs_with = [[i for i, p in enumerate(self.pairs) if v in p] for v in range(nl)]
        self.match = [-1] * self.npairs
        self.solutions = []
        self.nodes = 0
        self.time_limit = time_limit
        self.t0 = time.time()
        self.stop = False
        self.verbose = verbose
        self.max_solutions = max_solutions
        # contribution table: for matching p with q, the list of (v, w) decremented (8 entries with multiplicity)
        self.contrib = {}

    def _contrib(self, p, q):
        key = (p, q)
        c = self.contrib.get(key)
        if c is None:
            v1, v2 = self.pairs[p]
            w1, w2 = self.pairs[q]
            pi = self.pi
            c = [(v1, pi[w1]), (v1, pi[w2]), (v2, pi[w1]), (v2, pi[w2]),
                 (w1, pi[v1]), (w1, pi[v2]), (w2, pi[v1]), (w2, pi[v2])]
            self.contrib[key] = c
        return c

    def _apply(self, p, q, sign):
        for (v, w) in self._contrib(p, q):
            self.K3rem[v][w] -= sign
        if sign == 1:
            self.match[p] = q
            self.match[q] = p
        else:
            self.match[p] = -1
            self.match[q] = -1

    def _ok_after(self, p, q):
        for (v, w) in self._contrib(p, q):
            if self.K3rem[v][w] < 0:
                return False
        return True

    def _coverable(self):
        """for each (v,w) with remaining budget b: count unmatched pairs r containing v that can be matched to some
        unmatched pair containing pi(w) (other than r): must be >= b.  Cheap version: count candidates."""
        K3rem = self.K3rem
        pi = self.pi
        pw = self.pairs_with
        match = self.match
        nl = self.inst.nl
        unmatched_with = [[r for r in pw[v] if match[r] < 0] for v in range(nl)]
        for v in range(nl):
            row = K3rem[v]
            uv = unmatched_with[v]
            for w in range(nl):
                b = row[w]
                if b <= 0:
                    continue
                target = pi[w]
                # pairs r ∋ v unmatched, and there exists an unmatched pair s ∋ target, s != r
                cnt = 0
                ut = unmatched_with[target]
                if not ut:
                    return False
                for r in uv:
                    if len(ut) > 1 or ut[0] != r:
                        cnt += 1
                        if cnt >= b:
                            break
                if cnt < b:
                    return False
        return True

    def run(self):
        self._rec()
        return self.solutions

    def _rec(self):
        if self.stop:
            return
        self.nodes += 1
        if self.time_limit and (self.nodes & 1023) == 0 and time.time() - self.t0 > self.time_limit:
            self.stop = True
            return
        # fail-first: the unmatched pair with the fewest feasible partners
        match = self.match
        unmatched = [i for i in range(self.npairs) if match[i] < 0]
        if not unmatched:
            self.solutions.append(self.match[:])
            if self.verbose:
                print('  iota found at node', self.nodes, 'time %.1f' % (time.time() - self.t0))
            if self.max_solutions and len(self.solutions) >= self.max_solutions:
                self.stop = True
            return
        K3rem = self.K3rem
        best_p, best_list = -1, None
        for p in unmatched:
            lst = []
            for q in unmatched:
                if q == p:
                    continue
                ok = True
                for (v, w) in self._contrib(min(p, q), max(p, q)):
                    if K3rem[v][w] < 1:
                        ok = False
                        break
                if ok:
                    lst.append(q)
            if best_list is None or len(lst) < len(best_list):
                best_p, best_list = p, lst
                if not lst:
                    return
        p = best_p
        for q in best_list:
            a, b = min(p, q), max(p, q)
            self._apply(a, b, 1)
            if self._coverable():
                self._rec()
            self._apply(a, b, -1)
            if self.stop:
                return


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../structures/etl_detsq_00.json'
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    mu, tau = load_type(path)
    inst = TypeInstance(7, mu, tau)
    print('type', path, 'valid', inst.valid, 'pivots', sum(inst.pivot))
    srch = IotaSearch(inst, time_limit=tl, verbose=1)
    sols = srch.run()
    print('iota search: nodes', srch.nodes, 'time %.1fs' % (time.time() - srch.t0), 'stopped' if srch.stop else 'COMPLETE', 'solutions', len(sols))
