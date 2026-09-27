# scripts/fable — independent re-derivation and solver for the involution case (session 2026-09-27)

Independent of the other sessions' code. Everything is derived from lambda = 1, mu = 2 and the t-symmetry only.

- `inv.py` — vertex labelling (Z^m picture, t = -1), forced adjacencies, and the BvLS graph srg(243,22,1,2) built
  from the ternary Golay code (positive control for everything below).
- `analyze.py` — sign matrix S, checks S^2 + S = (2m-2) I, G = rI - S = d x projection, known span at a block
  has dimension 3m-3 (dependency sum_star p_P = -r ell_c), residual of the far orbits = d x projection of rank
  (rank G) - (3m-3): 25 for m = 11 (verified on BvLS), 4 for m = 7.
- `phisolve.py` — block type (mu, tau), K2/K3 counts, local and pairwise bounds (Python prototype).
- `blocktypes.py` — block-type classes for m = 7 up to 2^6 x| S_6: 3580 classes, 3308 valid
  (1787 / 958 / 407 / 106 / 39 / 11 by number of pivot labels), same numbers as C1. Output `types_all.json`.
- `step12.c` — iota enumeration (step 1) then orbit assignment with the rank-4 residual completion (step 2).
- `comb.c` — combined search (Phi orbit by orbit, K2 + K3 budgets, residual, optional psd-remainder
  pruning with `PSD=1`, `NORES=1` disables the residual). Validated: with `replay_bvls.txt` it reconstructs the
  true block structure of BvLS (`export_bvls_replay.py`).
- `iota2.c` — exact-cover (Knuth-style) iota enumeration.
- `run_type.py` — exports a type + G^+ for the C programs.  `runs/`, `runs2/` — logs of calibration runs.

Status: no block type finished. The trees explode in the first five orbits (about 300 x 160 x 300 x 137 x 111
options) before the residual rank constraint can act; iota enumeration yields thousands of iota per type
without completing in 20 minutes. See the session report for the conclusions.
