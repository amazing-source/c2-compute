"""ROOT-FIRST for the filtered mixture classes (note 44): the orientation-level orbit LP (every D-structure variable) of one
orientation of a twisted class, WITHOUT the leaf files.  The orientation is rebuilt exactly as mix_filter6.run orders its
representatives ('rep' = index in that order; locally, job j of class ci = the j-th rep with leaves in logs/mix6_cls_<ci>_S_0.json).
Infeasible -> certificate logs/orbit_certs/root_<ci>_<rep>.json.gz (model 'gen', spec {"In": ...}: covers every leaf of the
orientation; re-checked by orbit_verify_certs.py); feasible -> the job goes to stage 1 / stage 2.
Usage: python root_job.py <ci> <rep>      env TL, SMOOTH, LOWMEM.   Verdict line: ROOT <ci> <rep>: dead|feasible|timeout ..."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix_filter6 as F, orbit_colmodel as OM, orbit_colmodel_gen as OG
N = 7


_REPS = {}


def orientation(ci, rep):
    if ci not in _REPS:
        twE = F.twisted_classes()[ci]; auts = F.automorphisms2(twE, []); reps = {}
        for In, tw in F.orientations2(twE, []):
            arcs = [(a, b) for b in range(N) for a in In[b]]
            key = min(tuple(sorted((g[a], g[b]) for a, b in arcs)) for g in auts)
            if key not in reps: reps[key] = In                 # insertion order, exactly as mix_filter6.run
        _REPS[ci] = list(reps.values())
    return _REPS[ci][rep]


if __name__ == "__main__":
    ci, rep = int(sys.argv[1]), int(sys.argv[2])
    In = orientation(ci, rep); t0 = time.time()
    g = OG.OrbitModelGen(In); ncol = sum(len(M) for M in g.cols)
    obj, y = g.solve(tlimit=float(os.environ.get('TL', 5000)), verbose=False)
    rec = {"ci": ci, "rep": rep, "In": {str(c): v for c, v in In.items()}, "columns": ncol}
    if g.done and obj > 1e-6:
        ok, b = g.verify(y); rec["status"] = "dead" if ok else "certfail"
        if ok: OM.save_cert(g, y, "logs/orbit_certs/root_%d_%d.json.gz" % (ci, rep), "gen", {"In": rec["In"]}, b)
    else: rec["status"] = "feasible" if g.done else "timeout"
    rec["secs"] = round(time.time() - t0)
    os.makedirs("logs/root", exist_ok=True)
    open("logs/root/root_%d_%d.json" % (ci, rep), "w").write(json.dumps(rec))
    print("ROOT %d %d: %s columns %d %ds" % (ci, rep, rec["status"], ncol, rec["secs"]))
