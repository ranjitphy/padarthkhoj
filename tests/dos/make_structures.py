"""Synthetic graphene test structures (N-doped + vacancy), written in several formats."""
import os, sys
import numpy as np
from ase.build import graphene
from ase.io import write


def make(n=5, n_dope=3, vacancy=True, seed=0):
    rng = np.random.RandomState(seed)
    g = graphene(a=2.46, size=(n, n, 1), vacuum=8.0)
    g.pbc = True
    sym = g.get_chemical_symbols()
    idx = list(rng.choice(len(g), n_dope, replace=False))
    for i in idx:
        sym[i] = "N"
    g.set_chemical_symbols(sym)
    if vacancy:
        keep = [i for i in range(len(g)) if i != int(rng.choice([k for k in range(len(g)) if k not in idx]))]
        g = g[keep]
    g.positions += rng.normal(0, 0.03, g.positions.shape)
    return g


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    a = make()
    write(os.path.join(out, "CONTCAR"), a, format="vasp", direct=True, vasp5=True)
    write(os.path.join(out, "ndoped.cif"), a, format="cif")
    write(os.path.join(out, "ndoped.xyz"), a, format="extxyz")
    write(os.path.join(out, "ndoped_plain.xyz"), a, format="xyz")
    with open(os.path.join(out, "ndoped.cc1"), "w") as f:
        f.write("%d\n" % len(a))
        for i, (s, p) in enumerate(zip(a.get_chemical_symbols(), a.get_positions()), 1):
            f.write("%s %d %.6f %.6f %.6f 1\n" % (s, i, *p))
    print("wrote", os.listdir(out))
