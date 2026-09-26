"""Runs C1's g- = 18 sweep (conway99-c1, code/q7_g18_sweep.py, commit 7625868; its modules imported unchanged from
external_c1/code) on a chunk of the t = 0 split classes, and RECORDS every class (the original prints only a summary).
Same logic: unary filter -> local model (i)+(ii)+(iii) as a 0/1 MILP -> if feasible, pairwise rigid-row model.
Usage: python c1_g18_sweep_wrap.py START END      env TL1 (s, local MILP, default 60), TL2 (s, pairwise, default 60)
Output: logs/c1_g18/g18_<START>_<END>.jsonl"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1"))
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from q7_g16_test import classes_t_ge, ball_disc, split
from q7_g16_ilp import build_model
start, end = int(sys.argv[1]), int(sys.argv[2]); tl1 = float(os.environ.get('TL1', 60)); tl2 = float(os.environ.get('TL2', 60))
cls = [c for c in classes_t_ge(0) if c[2] == 0]
cls = [c for c in cls if split(*ball_disc(c[0], c[1]))]
os.makedirs("logs/c1_g18", exist_ok=True); fn = "logs/c1_g18/g18_%d_%d.jsonl" % (start, end)
done = {json.loads(l)['idx'] for l in open(fn)} if os.path.exists(fn) else set()
out = open(fn, "a"); print("t = 0 split classes:", len(cls), "chunk", start, end, flush=True)
for idx in range(start, min(end, len(cls))):
    if idx in done: continue
    tau, eta, t, osz = cls[idx]; t0 = time.time()
    rec = {'idx': idx, 'tau': [int(v) for v in tau], 'eta': [int(v) for v in eta], 'orbit_size': int(osz)}
    built = build_model(tau, eta, pairwise=False)
    if built is None: rec['status'] = 'dead-unary'
    else:
        A, lo, hi, nv, info = built
        r = milp(np.zeros(nv), constraints=[LinearConstraint(A, np.array(lo, float), np.array(hi, float))],
                 integrality=np.ones(nv), bounds=Bounds(0, 1), options={'time_limit': tl1})
        rec['status'] = {0: 'local-FEASIBLE', 2: 'dead-local'}.get(r.status, 'local-unknown(%d)' % r.status)
        if r.status == 0:
            A2, lo2, hi2, nv2, info2 = build_model(tau, eta, pairwise=True)
            r2 = milp(np.zeros(nv2), constraints=[LinearConstraint(A2, np.array(lo2, float), np.array(hi2, float))],
                      integrality=np.ones(nv2), bounds=Bounds(0, 1), options={'time_limit': tl2})
            rec['status'] = {0: 'ALIVE-pairwise', 2: 'dead-pairwise'}.get(r2.status, 'pairwise-unknown(%d)' % r2.status)
    rec['secs'] = round(time.time() - t0, 1); out.write(json.dumps(rec) + chr(10)); out.flush()
print("CHUNK DONE", start, end, flush=True)
