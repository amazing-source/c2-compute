"""Block fixing with the unified orbit model (note 43): for one skeleton in ptype format, fix the D-graph at the given blocks one
after the other (DFS; options = every D-graph of the block, mix_enum.dgraphs on its anchored-at-it and twisted pairs).
Node = {block: D-graph}; the D-partners of the one-block orbits pivoting at the fixed blocks are forced through forbidden
(orbit, partner) pairs (which orbit of a pair takes which third is WLOG, as in orbit_fixcamp.forbid_for).
Infeasible node -> certificate logs/orbit_certs/<prefix>_<tag>_n<node>.json.gz (model 'U', spec {ptype, forbid, assign});
an orbit left without admissible column -> 'dead-comb' (recorded).  Feasible node -> next block.
Usage: python run_U_fix.py <ptype json> <index> <blocks, e.g. 0,2,4 | auto> <prefix>      env LOWMEM, TL
(auto: every block with a single D-graph is fixed from the start (forced), then the blocks with more than one D-graph,
 fewest options first)"""
import sys, os, json, time
from collections import defaultdict
sys.path.insert(0, "scripts")
import orbit_colmodel as OM, orbit_colmodel_U as OU, mix_enum as ME
N = 7


def forbid_for(g, assign):
    fb = set()
    for c, edges in assign.items():
        trip = [(c, x, y) for (x, y) in edges] + [(c, y, x) for (x, y) in edges]
        thirds = defaultdict(list)
        for (_, x, y) in trip: thirds[x].append(y)
        lab = {}
        for x, ys in thirds.items():
            os_ = [o for o in g.orbs_of[frozenset((c, x))] if g.otype[o] == ('one', c)]
            assert len(os_) == len(ys), (c, x, os_, ys)
            for o, y in zip(os_, sorted(ys)): lab.setdefault((x, y), []).append(o)
        partner = {}
        for (x, y), Zs in lab.items():
            for t_, Z in enumerate(Zs): partner[Z] = lab[(y, x)][t_]
        for Z, W in partner.items():
            for Dz in range(g.n):
                if Dz != W and g.otype[Dz] == ('one', c) and c in g.pair[Dz] and g.pair[Dz] != g.pair[Z]: fb.add((Z, Dz))
    return fb


def ptype_auts(pt):
    """block permutations preserving the skeleton (anch c -> anch g(c), mix c -> mix g(c), other types kept): the unified model is
    invariant under them (orbit labels are only relabelled), so a dead node kills its whole orbit (orbital branching)"""
    import itertools
    out = []
    for g in itertools.permutations(range(N)):
        ok = True
        for (a, b), t in pt.items():
            t2 = pt[(min(g[a], g[b]), max(g[a], g[b]))]
            if (t2 != (t[0], g[t[1]])) if t[0] in ('anch', 'mix') else (t2 != t): ok = False; break
        if ok: out.append(g)
    return out


def img(A, g):
    return {g[c]: tuple(sorted(tuple(sorted((g[x], g[y]))) for (x, y) in v)) for c, v in A.items()}


if __name__ == "__main__":
    items = json.load(open(sys.argv[1])); it = items[int(sys.argv[2])]; tag = it.get("tag", it.get("rep", sys.argv[2]))
    prefix = sys.argv[4]
    pt = {tuple(int(c) for c in k.split(",")): tuple(v) for k, v in it["ptype"].items()}
    anch = {c: [x for x in range(N) if x != c and pt[(min(c, x), max(c, x))] == ('anch', c)] for c in range(N)}
    twl = {c: [x for x in range(N) if x != c and pt[(min(c, x), max(c, x))][0] == 'tw'] for c in range(N)}
    nopt = {c: len(ME.dgraphs(anch[c], twl[c])) if (anch[c] or twl[c]) else 0 for c in range(N)}
    if sys.argv[3] == 'auto':                 # blocks with more than one D-graph, fewest options first
        blocks = sorted([c for c in range(N) if nopt[c] > 1], key=lambda c: (nopt[c], c))
    else: blocks = [int(b) for b in sys.argv[3].split(",")]
    AUT = ptype_auts(pt)
    print("blocks to fix, in order:", [(c, nopt[c]) for c in blocks], "| skeleton automorphisms:", len(AUT)); sys.stdout.flush()
    root = {c: ME.dgraphs(anch[c], twl[c])[0] for c in range(N) if nopt[c] == 1} if sys.argv[3] == 'auto' else {}
    out = open("logs/run_U_fix_%s_%s.jsonl" % (prefix, tag), "a"); stack = [root]; node = 0; t00 = time.time(); openn = 0
    while stack:
        assign = stack.pop(); node += 1; t0 = time.time()
        g = OU.OrbitModelU(pt, forbid=None)
        fb = forbid_for(g, assign)
        if fb: g = OU.OrbitModelU(pt, forbid=fb)
        rec = {"node": node, "assign": {str(c): [list(e) for e in v] for c, v in assign.items()}}
        if g.empty_orbits:
            rec["status"] = "dead-comb"; rec["empty_orbits"] = g.empty_orbits
        else:
            obj, y = g.solve(tlimit=float(os.environ.get('TL', 6000)), verbose=False)
            if g.done and obj > 1e-6:
                ok, b = g.verify(y); rec["status"] = "dead" if ok else "certfail"; rec["slack"] = round(obj, 4)
                if ok: OM.save_cert(g, y, "logs/orbit_certs/%s_%s_n%d.json.gz" % (prefix, tag, node), "U",
                                    {"ptype": it["ptype"], "forbid": sorted(map(list, fb)), "assign": rec["assign"]}, b)
            else: rec["status"] = "feasible" if g.done else "timeout"
        rec["secs"] = round(time.time() - t0); out.write(json.dumps(rec) + chr(10)); out.flush()
        print("UFIX %s node %d assign %s: %s %ds" % (tag, node, rec["assign"], rec["status"], rec["secs"])); sys.stdout.flush()
        if rec["status"] in ("dead", "dead-comb"): continue
        if rec["status"] != "feasible": openn += 1; continue
        nxt = [c for c in blocks if c not in assign]
        if not nxt: print("UNRESOLVED all given blocks fixed", rec["assign"]); openn += 1; continue
        c = nxt[0]; opts = ME.dgraphs(anch[c], twl[c])
        if not os.environ.get('NOSYM'):            # orbital branching: one option per orbit of the stabiliser of the node and of c
            canon = {cc: tuple(sorted(tuple(sorted(e)) for e in v)) for cc, v in assign.items()}
            H = [gg for gg in AUT if gg[c] == c and img(canon, gg) == canon]
            opts = [o for o in opts if o == min(tuple(sorted(tuple(sorted((gg[x], gg[y]))) for (x, y) in o)) for gg in H)]
            print("  branch on block %d: %d options, %d up to the stabiliser (order %d)" % (c, len(ME.dgraphs(anch[c], twl[c])), len(opts), len(H))); sys.stdout.flush()
        for opt in opts:
            a2 = dict(assign); a2[c] = opt; stack.append(a2)
    print("UFIX %s: nodes %d open %d %s %.0fs" % (tag, node, openn, "ALL DEAD" if openn == 0 else "INCOMPLETE", time.time() - t00))
