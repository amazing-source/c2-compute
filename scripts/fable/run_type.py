"""Driver: export the type + Gpinv for step12.c, run it, parse the result."""
import sys, json, subprocess, os, time
import numpy as np
from phisolve import TypeInstance, load_type

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'step12')


def export(inst, path):
    with open(path, 'w') as f:
        f.write('%d\n' % inst.m)
        f.write(' '.join(map(str, inst.mu)) + '\n')
        f.write(' '.join(map(str, inst.tau)) + '\n')
        for row in inst.Gpinv:
            f.write(' '.join(repr(float(x)) for x in row) + '\n')


def run(mu, tau, max_iota=100000, time_limit=0, verbose=0, node_limit2=0, tag='t'):
    inst = TypeInstance(7, mu, tau)
    data = os.path.join(HERE, 'data_%s.txt' % tag)
    export(inst, data)
    cmd = [BIN, data, str(max_iota), str(time_limit), str(verbose), str(node_limit2)]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout, time.time() - t0, inst


if __name__ == '__main__':
    path = sys.argv[1]
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mu, tau = load_type(path)
    out, dt, inst = run(mu, tau, time_limit=tl, verbose=verbose, tag=os.path.basename(path).replace('.json', ''))
    print(out)
    print('pivots', sum(inst.pivot), 'wall %.1fs' % dt)
