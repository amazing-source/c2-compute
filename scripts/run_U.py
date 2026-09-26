"""Run the unified orbit model (orbit_colmodel_U) on a list of pair-type skeletons given in C2's ptype format
{"a,b": ["anch", c] | ["dd"] | ["tw"] | ["mix", c] | ["sib"]} (note 43; AD residues from structures, Sigma motifs from C1).
Input JSON: a list of objects with a "ptype" field and an identifier ("rep", "tag" or the list index).
Certificates: logs/orbit_certs/<prefix>_<id>.json.gz (model 'U', spec {"ptype": ...}); re-check with orbit_verify_U.py.
Usage: python run_U.py <json> <prefix> [indices...]      env BUILDONLY=1: only report the column counts"""
import sys, os, json, time
sys.path.insert(0, "scripts")
import orbit_colmodel as OM, orbit_colmodel_U as OU

items = json.load(open(sys.argv[1])); prefix = sys.argv[2]
idx = [int(a) for a in sys.argv[3:]] or list(range(len(items)))
for t in idx:
    it = items[t]; tag = it.get("rep", it.get("tag", t))
    pt = {tuple(int(c) for c in k.split(",")): tuple(v) for k, v in it["ptype"].items()}
    t0 = time.time(); g = OU.OrbitModelU(pt)
    if g.empty_orbits:
        print("%s %s: DEAD (combinatorial: orbits %s have no admissible column) %.0fs" % (prefix, tag, g.empty_orbits, time.time() - t0)); sys.stdout.flush(); continue
    ncol = sum(len(M) for M in g.cols)
    print("%s %s: columns %d rows %d built %.0fs" % (prefix, tag, ncol, len(g.rname), time.time() - t0)); sys.stdout.flush()
    if os.environ.get('BUILDONLY'): continue
    obj, y = g.solve(tlimit=float(os.environ.get('TL', 6000)), verbose=os.environ.get('V') is not None)
    if g.done and obj > 1e-6:
        ok, b = g.verify(y)
        if ok: OM.save_cert(g, y, "logs/orbit_certs/%s_%s.json.gz" % (prefix, tag), "U", {"ptype": it["ptype"]}, b)
        print("%s %s: INFEASIBLE slack %.4f exact %s %s %.0fs" % (prefix, tag, obj, "VALID" if ok else "FAILED", b, time.time() - t0))
    else: print("%s %s: %s obj %.5f %.0fs" % (prefix, tag, "FEASIBLE" if g.done else "TIME CAP", obj, time.time() - t0))
    sys.stdout.flush()
