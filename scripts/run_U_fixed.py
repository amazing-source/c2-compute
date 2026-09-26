"""Unified orbit model on skeletons with EVERY D-graph given (C1's fixed-D anchored motifs, note 43): one LP per assignment.
Input: a ptype file (list of {tag, ptype}) and an assignment file {tag: [ {block: [[x, y], ...]}, ... ]} (D-graph edges per block,
multi-edges repeated).  The D-partners are imposed by forbidden (orbit, partner) pairs (run_U_fix.forbid_for, WLOG labelling).
Certificates: logs/orbit_certs/<prefix>_<tag>_a<j>.json.gz (model 'U', spec {ptype, forbid, assign}); verdicts in
logs/run_U_fixed_<prefix>.jsonl.
Usage: python run_U_fixed.py <ptype json[,more]> <assign json> <prefix> [tags...]      env LOWMEM, TL"""
import sys, os, json, time
sys.path.insert(0, "scripts")
import orbit_colmodel as OM, orbit_colmodel_U as OU
from run_U_fix import forbid_for

items = {}
for fn in sys.argv[1].split(","):
    for it in json.load(open(fn)): items[it.get("tag", it.get("rep"))] = it
asg_all = json.load(open(sys.argv[2])); prefix = sys.argv[3]
tags = sys.argv[4:] or list(asg_all)
out = open("logs/run_U_fixed_%s.jsonl" % prefix, "a")
for tag in tags:
    it = items[tag]; pt = {tuple(int(c) for c in k.split(",")): tuple(v) for k, v in it["ptype"].items()}
    for j, a in enumerate(asg_all[tag]):
        assign = {int(c): [tuple(e) for e in edges] for c, edges in a.items()}
        t0 = time.time(); g = OU.OrbitModelU(pt); fb = forbid_for(g, assign); g = OU.OrbitModelU(pt, forbid=fb)
        rec = {"tag": tag, "j": j, "assign": a}
        if g.empty_orbits: rec["status"] = "dead-comb"; rec["empty_orbits"] = g.empty_orbits
        else:
            ncol = sum(len(M) for M in g.cols); rec["columns"] = ncol
            obj, y = g.solve(tlimit=float(os.environ.get('TL', 6000)), verbose=False)
            if g.done and obj > 1e-6:
                ok, b = g.verify(y); rec["status"] = "dead" if ok else "certfail"; rec["slack"] = round(obj, 4)
                if ok: OM.save_cert(g, y, "logs/orbit_certs/%s_%s_a%d.json.gz" % (prefix, tag, j), "U",
                                    {"ptype": it["ptype"], "forbid": sorted(map(list, fb)), "assign": a}, b)
            else:
                rec["status"] = "feasible" if g.done else "timeout"
                if g.done and os.environ.get('SAVESOL'):           # the LP point (columns with weight) for hand analysis
                    xs = getattr(g, 'last_x', None); keys = getattr(g, 'final_keys', [])
                    if xs is not None:
                        rec["solution"] = [[int(Z), [int(v) for v in __import__('numpy').frombuffer(qb, dtype='int8')], round(float(w), 6)]
                                           for (Z, qb), w in zip(keys, xs) if w > 1e-7]
        rec["secs"] = round(time.time() - t0); out.write(json.dumps(rec) + chr(10)); out.flush()
        print("UFIXED %s a%d: %s %ds" % (tag, j, rec["status"], rec["secs"])); sys.stdout.flush()
