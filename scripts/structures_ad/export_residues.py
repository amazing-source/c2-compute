"""Export the AD classes that survive the pair level (after enumerating the A-structures) in C2's ptype format:
{"a,b": ["anch", c] | ["dd"]}, plus the surviving A-block D-structures (cycles on In(A)) and the dd pairs."""
import json, re, sys
from pair_level import N, pr
reps = json.load(open(sys.argv[1]))
out = []
for line in open(sys.argv[2]):
    m = re.match(r"rep (\d+) n_dd (\d+) free blocks \[(\d+)\] structures \d+ alive (\d+) unknown (\d+)", line)
    if not m: continue
    t, nd, A, al, un = map(int, m.groups())
    if al == 0 and un == 0: continue
    rep = reps[t]
    head = {pr(a, h): h for a, h in rep['arcs']}
    ptype = {}
    for a in range(N):
        for b in range(a + 1, N):
            if [a, b] in rep['dd'] or [b, a] in rep['dd']: ptype['%d,%d' % (a, b)] = ['dd']
            else: ptype['%d,%d' % (a, b)] = ['anch', head[(a, b)]]
    alive_structs = line.split('] [', 1)[1] if '] [' in line else ''
    out.append({'rep': t, 'n_dd': nd, 'A': A, 'alive': al, 'unknown': un, 'ptype': ptype, 'alive_A_structures_sample': alive_structs.strip()})
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print(len(out), 'residual classes exported to', sys.argv[3])
