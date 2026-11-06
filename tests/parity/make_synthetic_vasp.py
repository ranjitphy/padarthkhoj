#!/usr/bin/env python3
"""Generate a small synthetic VASP run (DOSCAR / CONTCAR / OUTCAR) for testing.

The numbers are NOT physical results - they only have the right file layout and
plausible magnitudes so the whole pipeline (split -> descriptors -> SVR/MLR) can be
exercised without a real calculation.  Structure: 6x6 graphene with a divacancy,
four pyridinic N, a Fe atom in the hole and an *OH adsorbate (73 atoms).

    python make_synthetic_vasp.py OUTDIR [--ispin 1|2] [--cartesian] [--nedos 301] [--no-outcar]
"""
import argparse
import os

import numpy as np

A_CC = 1.42


def build_structure(seed=7):
    a = A_CC * np.sqrt(3.0)  # 2.4595
    nx = ny = 6
    lat = np.array([[a * nx, 0, 0], [-a * ny / 2.0, a * ny * np.sqrt(3) / 2.0, 0], [0, 0, 15.0]])
    pos, sym = [], []
    for i in range(nx):
        for j in range(ny):
            o = i * np.array([a, 0, 0]) + j * np.array([-a / 2.0, a * np.sqrt(3) / 2.0, 0])
            for b in (np.array([0, 0, 0]), np.array([0, A_CC, 0])):
                pos.append(o + b + np.array([0.0, 0.0, 7.5]))
                sym.append("C")
    pos = np.array(pos)
    # work in fractional space with minimum-image distances
    inv = np.linalg.inv(lat)

    def mic(d):
        f = d @ inv
        f -= np.rint(f)
        return f @ lat

    centre = (np.array([0.5, 0.5, 0.5]) @ lat)
    d0 = np.linalg.norm(np.array([mic(p - centre) for p in pos]), axis=1)
    i0 = int(np.argmin(d0))
    dn = np.linalg.norm(np.array([mic(p - pos[i0]) for p in pos]), axis=1)
    dn[i0] = 1e9
    i1 = int(np.argmin(dn))
    hole = pos[i0] + 0.5 * mic(pos[i1] - pos[i0])
    keep = [k for k in range(len(pos)) if k not in (i0, i1)]
    pos = pos[keep]
    sym = [sym[k] for k in keep]
    dh = np.linalg.norm(np.array([mic(p - hole) for p in pos]), axis=1)
    for k in np.argsort(dh)[:4]:
        sym[k] = "N"
    fe = hole.copy()
    o = fe + np.array([0, 0, 1.70])
    h = o + np.array([0.55, 0.0, 0.80])
    pos = np.vstack([pos, fe, o, h])
    sym += ["Fe", "O", "H"]
    order = {"C": 0, "N": 1, "Fe": 2, "O": 3, "H": 4}
    idx = sorted(range(len(sym)), key=lambda k: (order[sym[k]], k))
    pos = pos[idx]
    sym = [sym[k] for k in idx]
    frac = (pos @ inv) % 1.0
    return lat, sym, frac


def write_contcar(path, lat, sym, frac, cartesian=False, scale=1.0):
    species, counts = [], []
    for s in sym:
        if not species or species[-1] != s:
            species.append(s)
            counts.append(0)
        counts[-1] += 1
    with open(path, "w") as f:
        f.write("Fe-N4-C synthetic test structure\n")
        f.write("%.1f\n" % scale)
        for v in lat / scale:
            f.write("  %.16f  %.16f  %.16f\n" % tuple(v))
        f.write("  " + "  ".join(species) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")
        f.write("Selective dynamics\n")
        if cartesian:
            f.write("Cartesian\n")
            xyz = (frac @ lat) / scale
            for p in xyz:
                f.write("  %.16f  %.16f  %.16f   T   T   T\n" % tuple(p))
        else:
            f.write("Direct\n")
            for p in frac:
                f.write("  %.16f  %.16f  %.16f   T   T   T\n" % tuple(p))


def _g(e, mu, sig):
    return np.exp(-0.5 * ((e - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def atom_pdos(rng, symbol, e, ispin):
    """Return array (nedos, ncols) in DOSCAR layout (absolute energies are added later)."""
    n = len(e)
    # orbital order: s, py, pz, px, dxy, dyz, dz2, dxz, dx2
    orbs = []
    for k in range(9):
        c1, c2, c3 = rng.uniform(0.2, 0.5), rng.uniform(0.0, 0.2), rng.uniform(0.1, 0.4)
        mu1, mu2, mu3 = rng.uniform(-6, -2), rng.uniform(-1.0, 0.3), rng.uniform(1.5, 4.0)
        if symbol == "C" and k == 2:  # pz of carbon: pi band
            c1, c2 = rng.uniform(0.40, 0.55), rng.uniform(0.0, 0.16)
            mu1, mu2 = -3.0 + rng.uniform(-0.4, 0.4), -0.25 + rng.uniform(-0.2, 0.2)
        d = c1 * _g(e, mu1, 1.5) + c2 * _g(e, mu2, 0.45) + c3 * _g(e, mu3, 1.1)
        orbs.append(d)
    if ispin == 1:
        return np.column_stack(orbs)  # 9 columns
    cols = []
    for d in orbs:
        split = rng.uniform(-0.04, 0.04)
        cols.append(d * (0.5 + split))  # up
        cols.append(d * (0.5 - split))  # down (positive in DOSCAR)
    return np.column_stack(cols)  # 18 columns


def write_doscar(path, lat, natoms, symbols, ispin, nedos, efermi, rng):
    emax, emin = 10.0, -15.0
    e = np.linspace(emin, emax, nedos)
    vol = abs(np.linalg.det(lat))
    with open(path, "w") as f:
        f.write("%5d%5d    1    1\n" % (natoms, natoms))
        f.write("  %.8E  %.8E  %.8E  %.8E  %.8E\n" % (vol * 1e-30, lat[0, 0] * 1e-10, lat[1, 1] * 1e-10, lat[2, 2] * 1e-10, 5e-16))
        f.write("  1.000000000000000E-003\n")
        f.write("  CAR \n")
        f.write(" synthetic test\n")
        f.write("  %.8f  %.8f  %5d  %.8f  %.8f\n" % (emax, emin, nedos, efermi, 1.0))
        tot = np.zeros(nedos)
        for _ in range(natoms):
            tot += 0.9 * _g(e, -2.0, 2.5)
        cum = np.cumsum(tot) * (e[1] - e[0])
        for k in range(nedos):
            if ispin == 2:
                f.write(" %.8f  %.8E  %.8E  %.8E  %.8E\n" % (e[k], tot[k] / 2, tot[k] / 2, cum[k] / 2, cum[k] / 2))
            else:
                f.write(" %.8f  %.8E  %.8E\n" % (e[k], tot[k], cum[k]))
        for a in range(natoms):
            f.write("  %.8f  %.8f  %5d  %.8f  %.8f\n" % (emax, emin, nedos, efermi, 1.0))
            p = atom_pdos(rng, symbols[a], e - efermi, ispin)
            for k in range(nedos):
                f.write(" %.8f " % e[k] + " ".join("%.8E" % v for v in p[k]) + "\n")


def write_outcar(path, efermi, ispin):
    with open(path, "w") as f:
        f.write(" vasp.6.4.2 synthetic OUTCAR\n")
        f.write("   ISPIN  =      %d    spin polarized calculation?\n" % ispin)
        f.write("   LORBIT =     11    0 simple, 1 ext, 2 COOP (PROOUT), +10 allows to specify RWIGS\n")
        f.write(" E-fermi :  %.4f     XC(G=0):  -9.1234     alpha+bet : -8.5412\n" % (efermi + 0.01))
        f.write(" E-fermi :  %.4f     XC(G=0):  -9.1234     alpha+bet : -8.5412\n" % efermi)


def generate(folder, ispin=2, cartesian=False, nedos=301, outcar=True, seed=7, efermi=-1.2345):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    lat, sym, frac = build_structure(seed)
    write_contcar(os.path.join(folder, "CONTCAR"), lat, sym, frac, cartesian=cartesian)
    write_doscar(os.path.join(folder, "DOSCAR"), lat, len(sym), sym, ispin, nedos, efermi, rng)
    if outcar:
        write_outcar(os.path.join(folder, "OUTCAR"), efermi, ispin)
    return {"natoms": len(sym), "symbols": sym, "efermi": efermi, "ispin": ispin}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--ispin", type=int, default=2, choices=[1, 2])
    ap.add_argument("--cartesian", action="store_true")
    ap.add_argument("--nedos", type=int, default=301)
    ap.add_argument("--no-outcar", action="store_true")
    a = ap.parse_args()
    m = generate(a.outdir, a.ispin, a.cartesian, a.nedos, not a.no_outcar)
    print("wrote %s  (%d atoms, ISPIN=%d)" % (a.outdir, m["natoms"], m["ispin"]))
