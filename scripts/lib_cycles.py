import itertools
def cycle_structures(V):
    """All 2-regular loopless multigraphs on V as lists of cycles (a 2-cycle = double edge)."""
    V = list(V); res = []
    def rec(remaining, acc):
        if not remaining: res.append(list(acc)); return
        first, rest = remaining[0], remaining[1:]
        for L in range(2, len(remaining) + 1):
            if len(remaining) - L == 1: continue
            for comb in itertools.combinations(rest, L - 1):
                for perm in itertools.permutations(comb):
                    if L > 2 and perm[0] > perm[-1]: continue
                    rec([v for v in remaining if v not in (first,) + perm], acc + [(first,) + perm])
    rec(V, []); return res
if __name__ == "__main__":
    print(len(cycle_structures(range(6))), "structures on 6 vertices (expect 130)")
