"""ORIENTATION-FIRST campaign for the twisted classes whose leaf enumeration explodes (large T; note 43).
For a twisted class ci (mix_filter6.twisted_classes()[ci], Sigma = empty), the orientation classes are those of
mix_filter6.orientations2 (19.1, 26.2) up to Aut(T), in a fixed order; shard k of K takes the indices = k mod K.
Each orientation: orbit-level LP with every D-structure variable (orbit_colmodel_gen.OrbitModelGen).  Infeasible -> certificate
logs/orbit_certs/ogen_<ci>_<idx>.json.gz (model 'gen', spec {"In": ...}, re-checked by orbit_verify_certs.py);
feasible -> recorded in logs/orient_<ci>.jsonl for block fixing.
Usage: python orient_campaign.py <ci> [k K]      env TL (s per LP, default 6000), LOWMEM, ONLY (comma-separated indices)"""
import sys, os, json, time
sys.path.insert(0, "scripts")
import mix_filter6 as F, orbit_colmodel as OM, orbit_colmodel_gen as OG
N = 7


def orientation_reps(ci):
    twE = F.twisted_classes()[ci]; auts = F.automorphisms2(twE, []); reps = {}
    for In, tw in F.orientations2(twE, []):
        arcs = [(a, b) for b in range(N) for a in In[b]]
        key = min(tuple(sorted((g[a], g[b]) for a, b in arcs)) for g in auts)
        if key not in reps: reps[key] = {c: sorted(v) for c, v in In.items()}
    return twE, [reps[k] for k in sorted(reps)]


if __name__ == "__main__":
    ci = int(sys.argv[1]); k, K = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    twE, reps = orientation_reps(ci)
    done = set()
    if os.path.exists("logs/orient_%d.jsonl" % ci):
        for line in open("logs/orient_%d.jsonl" % ci): done.add(json.loads(line)["idx"])
    out = open("logs/orient_%d.jsonl" % ci, "a")
    only = {int(x) for x in os.environ['ONLY'].split(',')} if os.environ.get('ONLY') else None      # a chosen sample of indices
    for idx, In in enumerate(reps):
        if idx % K != k or idx in done or (only is not None and idx not in only): continue
        t0 = time.time(); g = OG.OrbitModelGen(In); ncol = sum(len(M) for M in g.cols)
        obj, y = g.solve(tlimit=float(os.environ.get('TL', 6000)), verbose=False)
        rec = {"ci": ci, "idx": idx, "In": {str(c): v for c, v in In.items()}, "A": next((c for c in range(N) if len(In[c]) == 6), None),
               "columns": ncol}
        if g.done and obj > 1e-6:
            ok, b = g.verify(y)
            rec["status"] = "dead" if ok else "certfail"; rec["slack"] = round(obj, 4)
            if ok: OM.save_cert(g, y, "logs/orbit_certs/ogen_%d_%d.json.gz" % (ci, idx), "gen", {"In": rec["In"]}, b)
        else: rec["status"] = "feasible" if g.done else "timeout"
        rec["secs"] = round(time.time() - t0)
        out.write(json.dumps(rec) + chr(10)); out.flush()
        print("ORIENT class %d idx %d/%d: %s columns %d %ds" % (ci, idx, len(reps), rec["status"], ncol, rec["secs"])); sys.stdout.flush()
