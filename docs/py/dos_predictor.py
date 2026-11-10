"""PDOS predictor for graphene-based (N-doped / defective) structures.

Two-stage model trained with gnn_pdos_workflow_v9d.py:
  stage 1  graph transformer  structure -> smooth 8-channel PDOS per atom
  stage 2  1D U-Net           smooth PDOS -> sharpened PDOS
Channels: s_up, s_dn, py_up, py_dn, pz_up, pz_dn, px_up, px_dn  (E - E_F, 3000 points, -25..10 eV).

BROWSER VERSION: the same networks re-implemented in plain NumPy (no PyTorch, no ASE), so they run inside
Pyodide in the web page.  The weights are exported from the PyTorch checkpoints by tools/export_weights.py
(BatchNorm folded into the convolutions); tests/verify_numpy_port.py checks the output against PyTorch.
"""
import io
import json
import math
import os
import re
import zipfile

import numpy as np

ORBITAL_LABELS = ["s_up", "s_dn", "py_up", "py_dn", "pz_up", "pz_dn", "px_up", "px_dn"]
ORBITAL_COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896"]
SPINUP_COLS = [0, 2, 4, 6]
SPINDN_COLS = [1, 3, 5, 7]
DESCRIPTOR_ORBS = [("py", 2, 3), ("pz", 4, 5), ("px", 6, 7)]
DESCRIPTOR_LABELS = ["%s_%s" % (n, q) for n, _, _ in DESCRIPTOR_ORBS for q in ("fermi", "occ")]

CUTOFF_SHORT = 4.0
CUTOFF_LONG = 6.0
GAUSS_DMIN, GAUSS_DMAX, GAUSS_STEP = 0.0, 6.0, 0.09375
IN_SCOPE_ELEMENTS = {"C", "N", "H"}
STAGE2_BATCH = 16
STAGE1_ATOM_CHUNK = 8  # atoms per pass through the energy head (memory: chunk x 3000 x 256 floats)

VALENCE_SPDF = {
    1: [1, 0, 0, 0], 2: [2, 0, 0, 0], 3: [1, 0, 0, 0], 4: [2, 0, 0, 0], 5: [2, 1, 0, 0],
    6: [2, 2, 0, 0], 7: [2, 3, 0, 0], 8: [2, 4, 0, 0], 9: [2, 5, 0, 0], 10: [2, 6, 0, 0],
    11: [1, 0, 0, 0], 12: [2, 0, 0, 0], 13: [2, 1, 0, 0], 14: [2, 2, 0, 0], 15: [2, 3, 0, 0],
    16: [2, 4, 0, 0], 17: [2, 5, 0, 0], 18: [2, 6, 0, 0], 19: [1, 0, 0, 0], 20: [2, 0, 0, 0],
    21: [2, 0, 1, 0], 22: [2, 0, 2, 0], 23: [2, 0, 3, 0], 24: [1, 0, 5, 0], 25: [2, 0, 5, 0],
    26: [2, 0, 6, 0], 27: [2, 0, 7, 0], 28: [2, 0, 8, 0], 29: [1, 0, 10, 0], 30: [2, 0, 10, 0],
    31: [2, 1, 0, 0], 32: [2, 2, 0, 0], 33: [2, 3, 0, 0], 34: [2, 4, 0, 0], 35: [2, 5, 0, 0],
    36: [2, 6, 0, 0],
}


class DosUnavailable(Exception):
    """kept for API compatibility."""


def availability():
    return {"available": True, "missing": []}


# ------------------------------------------------------------------ minimal Atoms
_SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo "
            "Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb "
            "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu").split()
_Z = {s: i for i, s in enumerate(_SYMBOLS)}


class Atoms:
    def __init__(self, symbols, positions, cell=None):
        self.symbols = list(symbols)
        self.pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.cell = np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=np.float64).reshape(3, 3)

    def __len__(self):
        return len(self.symbols)

    def get_chemical_symbols(self):
        return list(self.symbols)

    def get_atomic_numbers(self):
        return np.array([_Z[s] for s in self.symbols], dtype=int)

    def get_positions(self):
        return self.pos

    def get_cell(self):
        return self.cell

    def periodic(self):
        return abs(np.linalg.det(self.cell)) > 1e-6

    def get_scaled_positions(self, wrap=False):
        return self.pos @ np.linalg.inv(self.cell)


def _sym(tok):
    m = re.match(r"([A-Za-z]{1,2})", tok.strip())
    if not m:
        raise ValueError("bad element symbol: %r" % tok)
    s = m.group(1).capitalize()
    if s not in _Z:
        s = s[0]
        if s not in _Z:
            raise ValueError("unknown element: %r" % tok)
    return s


def _read_poscar_text(lines):
    scale_tok = lines[1].split()
    scale = float(scale_tok[0])
    cell = np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:  # negative = target volume
        scale = (-scale / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
    cell = cell * scale
    k = 5
    t = lines[k].split()
    try:
        counts = [int(v) for v in t]
        names = lines[0].split()  # VASP 4: symbols only in the title line
        if len(names) < len(counts):
            raise ValueError("VASP 4 file without element names")
        names = [_sym(n) for n in names[:len(counts)]]
        k += 1
    except ValueError:
        names = [_sym(n) for n in t]
        counts = [int(v) for v in lines[k + 1].split()]
        k += 2
    if lines[k].strip().lower().startswith("s"):  # Selective dynamics
        k += 1
    direct = not lines[k].strip().lower().startswith(("c", "k"))
    k += 1
    n = sum(counts)
    coords = np.array([[float(v) for v in lines[k + i].split()[:3]] for i in range(n)])
    pos = coords @ cell if direct else coords * scale
    symbols = [s for s, c in zip(names, counts) for _ in range(c)]
    return Atoms(symbols, pos, cell)


def _read_xyz_text(lines):
    n = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    cell = None
    m = re.search(r'Lattice="([^"]+)"', comment, re.I)
    if m:
        cell = np.array([float(v) for v in m.group(1).split()]).reshape(3, 3)
    symbols, pos = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        symbols.append(_sym(t[0]))
        pos.append([float(v) for v in t[1:4]])
    return Atoms(symbols, pos, cell)


def _cif_number(s):
    return float(re.sub(r"\(.*\)", "", s))


def _read_cif_text(text):
    toks = []
    for ln in text.splitlines():
        ln = ln.split("#")[0] if not ln.lstrip().startswith(";") else ln
        toks.append(ln.rstrip())
    data = {}
    loops = []
    i = 0
    L = [t for t in toks]
    while i < len(L):
        ln = L[i].strip()
        if ln.lower().startswith("loop_"):
            i += 1
            cols = []
            while i < len(L) and L[i].strip().startswith("_"):
                cols.append(L[i].strip().split()[0].lower()); i += 1
            rows, cur = [], []
            while i < len(L):
                s = L[i].strip()
                if not s:
                    i += 1; continue
                if s.startswith("_") or s.lower().startswith(("loop_", "data_")):
                    break
                for tkn in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", s):
                    cur.append(tkn.strip("'\""))
                    if len(cur) == len(cols):
                        rows.append(cur); cur = []
                i += 1
            loops.append((cols, rows))
            continue
        if ln.startswith("_"):
            parts = ln.split(None, 1)
            if len(parts) == 2:
                data[parts[0].lower()] = parts[1].strip().strip("'\"")
        i += 1
    a, b, c = (_cif_number(data["_cell_length_" + k]) for k in "abc")
    al, be, ga = (math.radians(_cif_number(data["_cell_angle_" + k])) for k in ("alpha", "beta", "gamma"))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    cell = np.array([va, vb, [cx, cy, cz]])
    site_cols = site_rows = None
    ops = ["x,y,z"]
    for cols, rows in loops:
        if "_atom_site_fract_x" in cols:
            site_cols, site_rows = cols, rows
        for key in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
            if key in cols:
                ops = [r[cols.index(key)] if len(r) > cols.index(key) else "x,y,z" for r in rows]
                if len(cols) > 1 and not all("," in o for o in ops):
                    ops = [r[-1] for r in rows]
    if not site_cols:
        raise ValueError("no atom sites in the CIF file")
    ix = [site_cols.index("_atom_site_fract_" + k) for k in "xyz"]
    sym_i = site_cols.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in site_cols else site_cols.index("_atom_site_label")
    occ_i = site_cols.index("_atom_site_occupancy") if "_atom_site_occupancy" in site_cols else None
    fr = []
    for r in site_rows:
        if occ_i is not None:
            try:
                if _cif_number(r[occ_i]) < 0.5:
                    continue
            except ValueError:
                pass
        fr.append((_sym(r[sym_i]), np.array([_cif_number(r[j]) for j in ix])))

    def apply(op, p):
        env = {"x": p[0], "y": p[1], "z": p[2]}
        out = []
        for expr in op.replace(" ", "").lower().split(","):
            if not re.fullmatch(r"[xyz0-9+\-*/().]+", expr):
                raise ValueError("unsupported symmetry operation: %r" % op)
            out.append(eval(_op_to_py(expr), {"__builtins__": {}}, env))
        return np.array(out, dtype=float)

    symbols, frac = [], []
    for s, p in fr:
        seen = []
        for op in ops:
            q = apply(op, p)
            if op.replace(" ", "").lower() != "x,y,z":
                q = q % 1.0           # symmetry copies are wrapped into the cell; the listed site keeps its coordinates (as ASE does)
            dup = any(np.allclose(((q - w) + 0.5) % 1.0 - 0.5, 0.0, atol=1e-3) for w in seen)
            if not dup:
                seen.append(q)
        for q in seen:
            symbols.append(s); frac.append(q)
    return Atoms(symbols, np.array(frac) @ cell, cell)


def _op_to_py(expr):
    """'-x+1/2' -> '-x+(1/2)'   ('2x' style is not used by CIF symmetry strings)"""
    e = re.sub(r"(\d+)/(\d+)", r"(\1/\2)", expr)
    e = re.sub(r"(?<=[\d\)])(?=[xyz])", "*", e)
    return e


def _read_cc1_text(lines):
    n = int(lines[0].split()[0])
    body = lines[1:1 + n]
    symbols, pos = [], []
    for ln in body:
        t = ln.split()
        if len(t) < 4:
            continue
        xyz = None
        for k in (2, 1):
            try:
                xyz = [float(v) for v in t[k:k + 3]]
                if len(xyz) == 3:
                    break
            except ValueError:
                xyz = None
        symbols.append(_sym(t[0])); pos.append(xyz)
    return Atoms(symbols, pos)


def read_structure(path):
    """POSCAR / CONTCAR / .vasp, CIF, XYZ / extxyz, .cc1 (parsed without ASE)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Structure file not found: %s" % path)
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    with open(path, "r", errors="replace") as f:
        text = f.read()
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    nonblank = [ln for ln in lines if ln.strip()]
    errors = []
    order = []
    if ext == ".cif":
        order = ["cif", "vasp", "xyz"]
    elif ext in (".xyz", ".extxyz"):
        order = ["xyz", "vasp", "cif"]
    elif ext == ".cc1":
        order = ["cc1", "xyz"]
    else:
        order = ["vasp", "cif", "xyz", "cc1"]
    for fmt in order:
        try:
            if fmt == "vasp":
                atoms = _read_poscar_text(lines)
            elif fmt == "cif":
                atoms = _read_cif_text(text)
            elif fmt == "xyz":
                atoms = _read_xyz_text(lines)
            else:
                atoms = _read_cc1_text(nonblank)
            if len(atoms) > 0:
                return atoms
        except Exception as e:  # noqa: BLE001
            errors.append("%s: %s" % (fmt, str(e).splitlines()[0] if str(e) else type(e).__name__))
    raise ValueError("Could not read %s as a structure (%s)" % (base, "; ".join(errors[:3]) or "no atoms found"))


def structure_summary(atoms, path=None):
    symbols = atoms.get_chemical_symbols()
    cell = np.array(atoms.get_cell())
    periodic = bool(abs(np.linalg.det(cell)) > 1e-6)
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    formula = "".join("%s%s" % (k, "" if v == 1 else v) for k, v in sorted(counts.items()))
    cart = atoms.get_positions()
    frac = atoms.get_scaled_positions(wrap=False) if periodic else None
    warnings = []
    out_of_scope = sorted(set(symbols) - IN_SCOPE_ELEMENTS)
    if out_of_scope:
        warnings.append("Elements %s are outside what the model was built for (graphene-based C / N structures); "
                        "predictions for them are unreliable." % ", ".join(out_of_scope))
    if not periodic:
        warnings.append("The file has no unit cell: the structure is treated as an isolated cluster "
                        "(no periodic images). The model was trained on periodic CONTCARs.")
    return {
        "success": True, "path": path, "name": os.path.basename(path) if path else "structure", "format": "numpy",
        "title": "", "natoms": len(symbols), "formula": formula, "symbols": symbols,
        "lattice": cell.tolist() if periodic else None, "cart": cart.tolist(),
        "frac": frac.tolist() if frac is not None else None, "periodic": periodic,
        "n_carbon": int(counts.get("C", 0)), "warnings": warnings,
    }


# ------------------------------------------------------------------ neighbour list (what ase.neighborlist.neighbor_list("ijdD") does)
def neighbor_list_ijdD(atoms, cutoff):
    pos = atoms.get_positions()
    n = len(pos)
    cell = np.asarray(atoms.get_cell(), dtype=np.float64)
    if atoms.periodic():
        inv = np.linalg.inv(cell)
        # number of cell repeats needed along each lattice vector: cutoff / distance between opposite faces
        heights = 1.0 / np.linalg.norm(inv, axis=0)
        reps = np.ceil(cutoff / heights).astype(int)
        rng = [np.arange(-r, r + 1) for r in reps]
        shifts = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
    else:
        shifts = np.zeros((1, 3), dtype=int)
    I, J, D = [], [], []
    for sh in shifts:
        off = sh @ cell
        # vectors from every atom i to every image of atom j
        vec = (pos[None, :, :] + off) - pos[:, None, :]
        d2 = np.einsum("ijk,ijk->ij", vec, vec)
        mask = d2 < cutoff * cutoff
        if not sh.any():
            np.fill_diagonal(mask, False)
        ii, jj = np.nonzero(mask)
        if ii.size:
            I.append(ii); J.append(jj); D.append(vec[ii, jj])
    if not I:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 3))
    I = np.concatenate(I); J = np.concatenate(J); D = np.concatenate(D)
    return I, J, np.linalg.norm(D, axis=1), D


# ------------------------------------------------------------------ features (as in the training script)
def _gaussian_expansion(distances):
    centers = np.arange(GAUSS_DMIN, GAUSS_DMAX + GAUSS_STEP, GAUSS_STEP, dtype=np.float32)
    width = GAUSS_STEP ** 2
    return np.exp(-((distances[:, None] - centers) ** 2) / width).astype(np.float32)


def build_graph(atoms):
    """node features (N,104), short edges, long edges -> numpy arrays."""
    n = len(atoms)
    Z_all = atoms.get_atomic_numbers()
    i_l, j_l, d_l, D_l = neighbor_list_ijdD(atoms, CUTOFF_LONG)
    short_mask = d_l < CUTOFF_SHORT
    i_s, D_s = i_l[short_mask], D_l[short_mask]

    order_l = np.argsort(i_l, kind="stable")
    bounds_l = np.searchsorted(i_l[order_l], np.arange(n + 1))
    order_s = np.argsort(i_s, kind="stable")
    bounds_s = np.searchsorted(i_s[order_s], np.arange(n + 1))

    node = np.zeros((n, 104), dtype=np.float32)
    for a in range(n):
        sel = order_l[bounds_l[a]:bounds_l[a + 1]]
        nb_z = Z_all[j_l[sel]] if len(sel) else np.array([])
        d = d_l[sel]
        Z = int(Z_all[a])
        onehot = np.zeros(92, dtype=np.float32)
        onehot[min(Z, 92) - 1] = 1.0
        valence = np.array(VALENCE_SPDF.get(Z, [0, 0, 0, 0]), dtype=np.float32)
        local = np.array([
            float(Z == 6), float(Z == 7),
            float(np.sum(nb_z == 6)), float(np.sum(nb_z == 7)),
            float(np.mean(d)) if len(d) > 0 else 0.0,
            float(len(d)),
        ], dtype=np.float32)
        sel_s = order_s[bounds_s[a]:bounds_s[a + 1]]
        vecs = D_s[sel_s]
        n_nb = vecs.shape[0]
        if n_nb >= 3:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vn = vecs / (norms + 1e-8)
            cov = (vn.T @ vn) / n_nb
            eig = np.sort(np.linalg.eigvalsh(cov))
            planarity = float(eig[0] / (eig.sum() + 1e-8))
            pyramid = float(np.linalg.norm(vn.mean(axis=0)))
        else:
            planarity, pyramid = 1.0 / 3.0, 0.0
        node[a] = np.concatenate([onehot, valence, local, np.array([planarity, pyramid], dtype=np.float32)])

    unit = D_l / (d_l[:, None] + 1e-8)
    attr = np.concatenate([_gaussian_expansion(d_l), unit], axis=1).astype(np.float32)
    short = d_l <= CUTOFF_SHORT
    return (node,
            np.stack([i_l[short], j_l[short]]), attr[short],
            np.stack([i_l[~short], j_l[~short]]), attr[~short])


# ------------------------------------------------------------------ NumPy networks
def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _prelu(x, a):
    return np.where(x >= 0, x, x * a)


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _lin(x, W, b=None):
    y = x @ W.T
    return y if b is None else y + b


def _layernorm(x, g, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _seg_softmax(alpha, dst, n):
    mx = np.full((n,) + alpha.shape[1:], -np.inf, dtype=alpha.dtype)
    np.maximum.at(mx, dst, alpha)
    ex = np.exp(alpha - mx[dst])
    den = np.zeros_like(mx)
    np.add.at(den, dst, ex)
    return ex / (den[dst] + 1e-16)


class Stage1:
    def __init__(self, w, cfg):
        self.w = w
        self.heads = cfg["n_heads"]

    def _conv(self, p, x, ei, ea):
        w, H = self.w, self.heads
        n = x.shape[0]
        C = w[p + "lin_key.weight"].shape[0] // H
        src, dst = ei[0], ei[1]
        q = _lin(x, w[p + "lin_query.weight"], w[p + "lin_query.bias"]).reshape(-1, H, C)
        k = _lin(x, w[p + "lin_key.weight"], w[p + "lin_key.bias"]).reshape(-1, H, C)
        v = _lin(x, w[p + "lin_value.weight"], w[p + "lin_value.bias"]).reshape(-1, H, C)
        e = _lin(ea, w[p + "lin_edge.weight"]).reshape(-1, H, C)
        alpha = (q[dst] * (k[src] + e)).sum(-1) / math.sqrt(C)
        alpha = _seg_softmax(alpha, dst, n)
        msg = (v[src] + e) * alpha[..., None]
        out = np.zeros((n, H, C), dtype=x.dtype)
        np.add.at(out, dst, msg)
        out = out.reshape(n, H * C)
        xr = _lin(x, w[p + "lin_skip.weight"], w[p + "lin_skip.bias"])
        beta = _sigmoid(_lin(np.concatenate([out, xr, out - xr], axis=-1), w[p + "lin_beta.weight"]))
        return beta * xr + (1 - beta) * out

    def _layer(self, p, x, ei, ea):
        w = self.w
        xn = self._conv(p + "conv.", x, ei, ea)
        g = _sigmoid(_lin(np.concatenate([x, xn], axis=-1), w[p + "gate_res.gate.0.weight"], w[p + "gate_res.gate.0.bias"]))
        x = _layernorm(g * xn + (1 - g) * x, w[p + "gate_res.norm.weight"], w[p + "gate_res.norm.bias"])
        f = _lin(_prelu(_lin(x, w[p + "ffn.0.weight"], w[p + "ffn.0.bias"]), w[p + "ffn.1.weight"]), w[p + "ffn.2.weight"], w[p + "ffn.2.bias"])
        return _layernorm(x + f, w[p + "norm2.weight"], w[p + "norm2.bias"])

    def _mlp_silu(self, p, x, final=None):
        w = self.w
        x = _silu(_lin(x, w[p + "0.weight"], w[p + "0.bias"]))
        x = _silu(_lin(x, w[p + "2.weight"], w[p + "2.bias"]))
        if final is not None:
            x = _lin(x, w[p + "4.weight"], w[p + "4.bias"])
            x = _softplus(x) if final == "softplus" else x
        return x

    def forward(self, x_in, ei_s, ea_s, ei_l, ea_l, energy, atom_chunk=8, progress=None):
        w = self.w
        x = _silu(_layernorm(_lin(x_in, w["node_embed.0.weight"], w["node_embed.0.bias"]), w["node_embed.1.weight"], w["node_embed.1.bias"]))
        ea = _prelu(_lin(ea_s, w["edge_embed.0.weight"], w["edge_embed.0.bias"]), w["edge_embed.1.weight"])
        xs = x
        for i in range(sum(1 for k in w if k.startswith("short_layers.") and k.endswith("norm2.weight"))):
            xs = self._layer("short_layers.%d." % i, xs, ei_s, ea)
        eal = _prelu(_lin(ea_l, w["edge_embed.0.weight"], w["edge_embed.0.bias"]), w["edge_embed.1.weight"])
        xl = x
        for i in range(sum(1 for k in w if k.startswith("long_layers.") and k.endswith("norm2.weight"))):
            xl = self._layer("long_layers.%d." % i, xl, ei_l, eal)
        xf = _layernorm(_silu(_lin(np.concatenate([xs, xl], axis=-1), w["fuse.0.weight"], w["fuse.0.bias"])), w["fuse.2.weight"], w["fuse.2.bias"])
        # energy encoder
        e = energy[:, None]
        coarse = self._mlp_silu("energy_encoder.coarse.", e)
        sigma = np.exp(w["energy_encoder.log_sigma"])[None, :]
        env = np.exp(-0.5 * (e / sigma) ** 2)
        om = w["energy_encoder.omega0"][None, :]
        morlet = np.concatenate([env * np.cos(om * e), env * np.sin(om * e)], axis=-1)
        fine = self._mlp_silu("energy_encoder.fine.", morlet)
        enc = np.concatenate([coarse, fine], axis=-1).astype(np.float32)          # (E, 128)
        scale = np.concatenate([self._head_scale("head_scale_s.", xf), self._head_scale("head_scale_p.", xf)], axis=-1)
        n = xf.shape[0]
        outs = []
        gate_tab = _sigmoid(w["atom_type_gate.gate_embed.weight"])
        atype = np.where(x_in[:, 96] > 0.5, 0, np.where(x_in[:, 97] > 0.5, 1, 2))
        for a in range(0, n, atom_chunk):
            b = min(n, a + atom_chunk)
            gam = _lin(xf[a:b], w["film.to_gamma.weight"], w["film.to_gamma.bias"])
            bet = _lin(xf[a:b], w["film.to_beta.weight"], w["film.to_beta.bias"])
            mod = gam[:, None, :] * enc[None, :, :] + bet[:, None, :]
            mod = mod * gate_tab[atype[a:b]][:, None, :]
            h = _prelu(_lin(mod, w["head_dos.0.weight"], w["head_dos.0.bias"]), w["head_dos.1.weight"])
            h = _prelu(_lin(h, w["head_dos.2.weight"], w["head_dos.2.bias"]), w["head_dos.3.weight"])
            h = _lin(h, w["head_dos.4.weight"], w["head_dos.4.bias"])
            outs.append(h * scale[a:b][:, None, :])
            if progress:
                progress("stage1", "Stage 1: energy head, atoms %d-%d of %d" % (a + 1, b, n), 20 + 40.0 * b / n)
        return np.concatenate(outs, axis=0)

    def _head_scale(self, p, x):
        w = self.w
        x = _silu(_lin(x, w[p + "0.weight"], w[p + "0.bias"]))
        x = _silu(_lin(x, w[p + "2.weight"], w[p + "2.bias"]))
        return _softplus(_lin(x, w[p + "4.weight"], w[p + "4.bias"]))


def _conv1d(x, w, b, stride=1, pad=0):
    """x (B, C, L), w (O, C, K) -> (B, O, Lout); im2col + matmul."""
    B, C, L = x.shape
    O, _, K = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    Lout = (L + 2 * pad - K) // stride + 1
    win = np.lib.stride_tricks.sliding_window_view(xp, K, axis=2)[:, :, ::stride, :][:, :, :Lout, :]    # B, C, Lout, K
    cols = win.transpose(0, 2, 1, 3).reshape(B * Lout, C * K)
    y = cols @ w.reshape(O, C * K).T + b
    return y.reshape(B, Lout, O).transpose(0, 2, 1)


def _convT1d(x, w, b, stride, pad, out_pad):
    """ConvTranspose1d; w (Cin, Cout, K)."""
    B, C, L = x.shape
    K = w.shape[2]
    up = np.zeros((B, C, (L - 1) * stride + 1), dtype=x.dtype)
    up[:, :, ::stride] = x
    left = K - 1 - pad
    up = np.pad(up, ((0, 0), (0, 0), (left, left + out_pad)))
    wc = np.ascontiguousarray(w.transpose(1, 0, 2)[:, :, ::-1])
    return _conv1d(up, wc, b, 1, 0)


def _interp_nearest(x, size):
    L = x.shape[-1]
    if L == size:
        return x
    idx = np.floor(np.arange(size) * (L / size)).astype(int)
    return x[..., idx]


class Stage2:
    def __init__(self, w, cfg):
        self.w = w
        self.K = cfg["kernel"]
        self.pad = self.K // 2

    def _res(self, p, x, act):
        w = self.w
        y = _conv1d(x, w[p + "net.0.weight"], w[p + "net.0.bias"], 1, self.pad)
        y = _prelu(y, w[p + "net.2.weight"])
        y = _conv1d(y, w[p + "net.3.weight"], w[p + "net.3.bias"], 1, self.pad)
        return _prelu(x + y, w[p + "act.weight"])

    def forward(self, x):
        w, pad = self.w, self.pad
        e1 = _prelu(_conv1d(x, w["enc1.0.weight"], w["enc1.0.bias"], 1, pad), w["enc1.1.weight"])
        e1 = self._res("enc1.2.", e1, None)
        e2 = _prelu(_conv1d(e1, w["enc2.0.weight"], w["enc2.0.bias"], 2, pad), w["enc2.1.weight"])
        e2 = self._res("enc2.2.", e2, None)
        e3 = _prelu(_conv1d(e2, w["enc3.0.weight"], w["enc3.0.bias"], 2, pad), w["enc3.1.weight"])
        e3 = self._res("enc3.2.", e3, None)
        b = self._res("bottleneck.1.", self._res("bottleneck.0.", e3, None), None)
        d3 = _prelu(_convT1d(b, w["dec3.0.weight"], w["dec3.0.bias"], 2, pad, 1), w["dec3.1.weight"])
        d3 = self._res("dec3.2.", d3, None)
        d3 = _interp_nearest(d3, e2.shape[-1])
        d2 = _prelu(_convT1d(np.concatenate([d3, e2], axis=1), w["dec2.0.weight"], w["dec2.0.bias"], 2, pad, 1), w["dec2.1.weight"])
        d2 = self._res("dec2.2.", d2, None)
        d2 = _interp_nearest(d2, e1.shape[-1])
        out = _conv1d(np.concatenate([d2, e1], axis=1), w["out.weight"], w["out.bias"], 1, 0)
        return x + out


class Predictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model_config.json")) as f:
            self.cfg = json.load(f)
        w1 = dict(np.load(os.path.join(model_dir, "stage1.npz")))
        w2 = dict(np.load(os.path.join(model_dir, "stage2.npz")))
        self.stage1 = Stage1(w1, self.cfg["stage1"])
        self.stage2 = Stage2(w2, self.cfg["stage2"])
        self.energy = np.load(os.path.join(model_dir, "energy_grid.npy")).astype(np.float64)
        self.energy32 = self.energy.astype(np.float32)
        self.scale_up = float(self.cfg["normalizer"]["scale_up"])
        self.scale_dn = float(self.cfg["normalizer"]["scale_dn"])

    def predict(self, atoms, progress=None):
        n = len(atoms)
        if progress:
            progress("features", "Building graph (%d atoms)" % n, 5)
        node, ei_s, ea_s, ei_l, ea_l = build_graph(atoms)
        if progress:
            progress("stage1", "Stage 1: graph network", 12)
        smooth = self.stage1.forward(node, ei_s, ea_s, ei_l, ea_l, self.energy32, atom_chunk=STAGE1_ATOM_CHUNK, progress=progress)
        chunks = []
        for a in range(0, n, STAGE2_BATCH):
            chunk = smooth[a:a + STAGE2_BATCH].transpose(0, 2, 1).astype(np.float32)
            chunks.append(self.stage2.forward(chunk).transpose(0, 2, 1))
            if progress:
                progress("stage2", "Stage 2: sharpening, atoms %d-%d of %d" % (a + 1, min(n, a + STAGE2_BATCH), n), 60 + 38.0 * min(n, a + STAGE2_BATCH) / n)
        sharp = np.concatenate(chunks, axis=0).astype(np.float64)
        sharp[:, :, SPINUP_COLS] *= self.scale_up
        sharp[:, :, SPINDN_COLS] *= self.scale_dn
        return sharp.astype(np.float32)


_PREDICTOR = {"obj": None, "dir": None}


def get_predictor(model_dir):
    if _PREDICTOR["obj"] is None or _PREDICTOR["dir"] != model_dir:
        _PREDICTOR["obj"] = Predictor(model_dir)
        _PREDICTOR["dir"] = model_dir
    return _PREDICTOR["obj"]


# ------------------------------------------------------------------ descriptors (same definition as training)
def orbital_descriptors(pdos, energy):
    """py/pz/px {fermi, occ}: total = up - |dn|; fermi = value at last point below 0 eV; occ = trapz over E<0."""
    neg = energy < 0.0
    idx_neg = np.where(neg)[0]
    out = {}
    if idx_neg.size == 0:
        for name, _, _ in DESCRIPTOR_ORBS:
            out[name + "_fermi"] = np.zeros(pdos.shape[0])
            out[name + "_occ"] = np.zeros(pdos.shape[0])
        return out
    idx_ef = idx_neg[np.argmax(energy[idx_neg])]
    e_neg = energy[neg]
    order = np.argsort(e_neg)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    for name, up, dn in DESCRIPTOR_ORBS:
        total = pdos[:, :, up].astype(np.float64) - np.abs(pdos[:, :, dn].astype(np.float64))
        out[name + "_fermi"] = total[:, idx_ef]
        out[name + "_occ"] = trapz(total[:, neg][:, order], e_neg[order], axis=1)
    return out


# ------------------------------------------------------------------ plots
def _plot_atom(fig, ax, energy, curves, title, emin, emax, orbitals=None):
    ax.clear()
    sel = orbitals if orbitals is not None else list(range(8))
    for k in sel:
        ax.plot(energy, curves[:, k], color=ORBITAL_COLORS[k], lw=1.3,
                label=ORBITAL_LABELS[k].replace("py", "Py").replace("pz", "Pz").replace("px", "Px").replace("_up", " up").replace("_dn", " down"))
    ax.axvline(0.0, color="#555", lw=0.9, ls="--")
    ax.axhline(0.0, color="#999", lw=0.6)
    ax.set_xlim(emin, emax)
    m = (energy >= emin) & (energy <= emax)
    if m.any():
        vis = curves[m][:, sel]
        lim = float(np.abs(vis).max()) if vis.size else 1.0
        lim = max(lim * 1.08, 1e-3)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel("E - E$_F$ (eV)")
    ax.set_ylabel("PDOS (states/eV)")
    ax.set_title(title, fontsize=11)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False)
    fig.tight_layout()


def export_zip(path_out, energy, pdos, symbols, fmt="png", dpi=150, emin=-10.0, emax=5.0, orbitals=None,
               progress=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fmt = (fmt or "png").lower()
    if fmt in ("jpg", "jpeg"):
        fmt, pil_ext = "jpg", "jpg"
    else:
        fmt, pil_ext = "png", "png"
    n = pdos.shape[0]
    width = len(str(n))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    with zipfile.ZipFile(path_out, "w", zipfile.ZIP_DEFLATED) as z:
        for a in range(n):
            _plot_atom(fig, ax, energy, pdos[a], "Atom %d (%s)" % (a + 1, symbols[a]), emin, emax, orbitals)
            buf = io.BytesIO()
            if pil_ext == "jpg":
                fig.savefig(buf, format="jpeg", dpi=dpi, facecolor="white", bbox_inches="tight", pil_kwargs={"quality": 95})
            else:
                fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
            z.writestr("PDOS_atom_%s_%s.%s" % (str(a + 1).zfill(width), symbols[a], pil_ext), buf.getvalue())
            if progress and (a % 5 == 0 or a == n - 1):
                progress("export", "Rendering plots %d / %d" % (a + 1, n), int(100 * (a + 1) / n))
    plt.close(fig)
    return path_out
