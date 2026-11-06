"""PDOS predictor for graphene-based (N-doped / defective) structures.

Two-stage model trained with gnn_pdos_workflow_v9d.py:
  stage 1  graph transformer  structure -> smooth 8-channel PDOS per atom
  stage 2  1D U-Net           smooth PDOS -> sharpened PDOS
Channels: s_up, s_dn, py_up, py_dn, pz_up, pz_dn, px_up, px_dn  (E - E_F, 3000 points, -25..10 eV).

Everything here is a faithful port of the training script (same features, same layers, same
normalisation).  PyTorch Geometric's TransformerConv is re-implemented in plain torch so the app
needs only torch + ase + numpy + matplotlib.  torch / ase are imported lazily: the rest of the
engine keeps working when they are missing.
"""
import io
import json
import math
import os
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
STAGE1_ATOM_CHUNK = 24  # atoms per pass through the energy head (memory: chunk x 3000 x 256 floats)

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
    """torch / ase / matplotlib missing."""


def _require():
    try:
        import torch  # noqa: F401
        import ase  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise DosUnavailable(
            "The PDOS predictor needs PyTorch and ASE (%s: %s). Run 'npm run setup:python' "
            "(or: pip install -r engine/requirements.txt) and restart the app." % (type(e).__name__, e)
        )


def availability():
    """Cheap check (does not import torch): are the packages installed?"""
    import importlib.util
    missing = [m for m in ("torch", "ase", "matplotlib") if importlib.util.find_spec(m) is None]
    return {"available": not missing, "missing": missing}


# ------------------------------------------------------------------ structures
def _read_cc1(path):
    """ChemDraw / Chem3D .cc1: first line = atom count, then 'Symbol serial x y z type [bonds...]'."""
    from ase import Atoms
    from ase.data import chemical_symbols
    with open(path, "r", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    try:
        n = int(lines[0].split()[0])
        body = lines[1:1 + n]
    except ValueError:
        body = lines
    symbols, pos = [], []
    for ln in body:
        t = ln.split()
        if len(t) < 4:
            continue
        sym = t[0].capitalize()
        if sym not in chemical_symbols:
            sym = "".join(ch for ch in sym if ch.isalpha()).capitalize()
        for k in (2, 1):
            try:
                xyz = [float(v) for v in t[k:k + 3]]
                if len(xyz) == 3:
                    break
            except ValueError:
                xyz = None
        if sym not in chemical_symbols or not xyz or len(xyz) != 3:
            raise ValueError("cannot read atom line: %r" % ln)
        symbols.append(sym)
        pos.append(xyz)
    if not symbols:
        raise ValueError("no atoms found")
    return Atoms(symbols=symbols, positions=pos)


def read_structure(path):
    """Any ASE-readable format (cif, POSCAR/CONTCAR/.vasp, xyz/extxyz, pdb, mol, sdf, gen, ...) plus .cc1."""
    _require()
    from ase.io import read
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Structure file not found: %s" % path)
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()
    errors = []
    atoms = None
    if ext == ".cc1":
        atoms = _read_cc1(path)
    else:
        attempts = [None]
        if ext in ("", ".vasp") or base.upper().startswith(("POSCAR", "CONTCAR")):
            attempts = ["vasp", None]
        attempts += ["vasp", "cif", "extxyz", "xyz", "cc1"]
        seen = set()
        for fmt in attempts:
            if fmt in seen:
                continue
            seen.add(fmt)
            try:
                if fmt == "cc1":
                    atoms = _read_cc1(path)
                else:
                    atoms = read(path, format=fmt) if fmt else read(path)
                    if isinstance(atoms, list):
                        atoms = atoms[-1]
                if atoms is not None and len(atoms) > 0:
                    break
            except Exception as e:  # noqa: BLE001
                errors.append("%s: %s" % (fmt or "auto", str(e).splitlines()[0] if str(e) else type(e).__name__))
                atoms = None
    if atoms is None or len(atoms) == 0:
        raise ValueError("Could not read %s as a structure (%s)" % (base, "; ".join(errors[:3]) or "no atoms found"))
    return atoms


def structure_summary(atoms, path=None):
    symbols = atoms.get_chemical_symbols()
    cell = np.array(atoms.get_cell())
    periodic = bool(abs(np.linalg.det(cell)) > 1e-6)
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    formula = "".join("%s%s" % (k, "" if v == 1 else v) for k, v in sorted(counts.items()))
    cart = atoms.get_positions()
    frac = None
    if periodic:
        frac = atoms.get_scaled_positions(wrap=False)
    warnings = []
    out_of_scope = sorted(set(symbols) - IN_SCOPE_ELEMENTS)
    if out_of_scope:
        warnings.append("Elements %s are outside what the model was built for (graphene-based C / N structures); "
                        "predictions for them are unreliable." % ", ".join(out_of_scope))
    if not periodic:
        warnings.append("The file has no unit cell: the structure is treated as an isolated cluster "
                        "(no periodic images). The model was trained on periodic CONTCARs.")
    return {
        "success": True,
        "path": path,
        "name": os.path.basename(path) if path else "structure",
        "format": "ase",
        "title": "",
        "natoms": len(symbols),
        "formula": formula,
        "symbols": symbols,
        "lattice": cell.tolist() if periodic else None,
        "cart": cart.tolist(),
        "frac": frac.tolist() if frac is not None else None,
        "periodic": periodic,
        "n_carbon": int(counts.get("C", 0)),
        "warnings": warnings,
    }


# ------------------------------------------------------------------ features (as in the training script)
def _gaussian_expansion(distances):
    centers = np.arange(GAUSS_DMIN, GAUSS_DMAX + GAUSS_STEP, GAUSS_STEP, dtype=np.float32)
    width = GAUSS_STEP ** 2
    return np.exp(-((distances[:, None] - centers) ** 2) / width).astype(np.float32)


def build_graph(atoms):
    """node features (N,104), short edges, long edges -> numpy arrays."""
    from ase.neighborlist import neighbor_list
    n = len(atoms)
    Z_all = atoms.get_atomic_numbers()

    i_l, j_l, d_l, D_l = neighbor_list("ijdD", atoms, cutoff=CUTOFF_LONG)
    i_s, _j_s, D_s = neighbor_list("ijD", atoms, cutoff=CUTOFF_SHORT)[0:3]

    # group per atom once (training code recomputed the neighbour list for every atom; same numbers)
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


# ------------------------------------------------------------------ networks
def _define_networks():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def seg_softmax(src, index, n):
        shape = (n,) + tuple(src.shape[1:])
        idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        mx = torch.full(shape, float("-inf"), dtype=src.dtype).scatter_reduce(0, idx, src, "amax", include_self=True)
        ex = (src - mx[index]).exp()
        den = torch.zeros(shape, dtype=src.dtype).index_add_(0, index, ex)
        return ex / (den[index] + 1e-16)

    class TransformerConvLite(nn.Module):
        """torch_geometric.nn.TransformerConv(in, out, heads, concat=True, beta=True, edge_dim)."""
        def __init__(self, in_dim, out_dim, heads, edge_dim):
            super().__init__()
            self.heads, self.out_dim = heads, out_dim
            hc = heads * out_dim
            self.lin_key = nn.Linear(in_dim, hc)
            self.lin_query = nn.Linear(in_dim, hc)
            self.lin_value = nn.Linear(in_dim, hc)
            self.lin_edge = nn.Linear(edge_dim, hc, bias=False)
            self.lin_skip = nn.Linear(in_dim, hc)
            self.lin_beta = nn.Linear(3 * hc, 1, bias=False)

        def forward(self, x, edge_index, edge_attr):
            H, C = self.heads, self.out_dim
            n = x.shape[0]
            src, dst = edge_index[0], edge_index[1]       # flow: source -> target
            q = self.lin_query(x).view(-1, H, C)
            k = self.lin_key(x).view(-1, H, C)
            v = self.lin_value(x).view(-1, H, C)
            e = self.lin_edge(edge_attr).view(-1, H, C)
            key_j = k[src] + e
            alpha = (q[dst] * key_j).sum(-1) / math.sqrt(C)
            alpha = seg_softmax(alpha, dst, n)
            msg = (v[src] + e) * alpha.unsqueeze(-1)
            out = torch.zeros(n, H, C, dtype=x.dtype).index_add_(0, dst, msg).view(n, H * C)
            x_r = self.lin_skip(x)
            beta = torch.sigmoid(self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1)))
            return beta * x_r + (1 - beta) * out

    class GatedResidual(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
            self.norm = nn.LayerNorm(dim)

        def forward(self, x, x_new):
            g = self.gate(torch.cat([x, x_new], dim=-1))
            return self.norm(g * x_new + (1 - g) * x)

    class GTLayer(nn.Module):
        def __init__(self, hidden_dim, edge_dim, n_heads=4):
            super().__init__()
            self.conv = TransformerConvLite(hidden_dim, hidden_dim // n_heads, n_heads, edge_dim)
            self.gate_res = GatedResidual(hidden_dim)
            self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.PReLU(), nn.Linear(hidden_dim * 2, hidden_dim))
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, x, edge_index, edge_attr):
            x = self.gate_res(x, self.conv(x, edge_index, edge_attr))
            return self.norm2(x + self.ffn(x))

    class MorletEnergyEncoder(nn.Module):
        def __init__(self, energy_dim=64, n_scales=8):
            super().__init__()
            self.log_sigma = nn.Parameter(torch.linspace(math.log(0.1), math.log(5.0), n_scales))
            self.omega0 = nn.Parameter(torch.linspace(0.5, 20.0, n_scales))
            self.coarse = nn.Sequential(nn.Linear(1, energy_dim), nn.SiLU(), nn.Linear(energy_dim, energy_dim), nn.SiLU())
            self.fine = nn.Sequential(nn.Linear(2 * n_scales, energy_dim), nn.SiLU(), nn.Linear(energy_dim, energy_dim), nn.SiLU())

        def forward(self, energy_grid):
            coarse = self.coarse(energy_grid)
            e = energy_grid
            sigma = torch.exp(self.log_sigma)
            envelope = torch.exp(-0.5 * (e / sigma.unsqueeze(0)) ** 2)
            morlet = torch.cat([envelope * torch.cos(self.omega0.unsqueeze(0) * e),
                                envelope * torch.sin(self.omega0.unsqueeze(0) * e)], dim=-1)
            return torch.cat([coarse, self.fine(morlet)], dim=-1)

    class AtomTypeGate(nn.Module):
        def __init__(self, energy_dim):
            super().__init__()
            self.gate_embed = nn.Embedding(3, energy_dim)

        def forward(self, x, modulated):
            is_C = x[:, 96] > 0.5
            is_N = x[:, 97] > 0.5
            atype = torch.where(is_C, torch.zeros_like(is_C, dtype=torch.long),
                                torch.where(is_N, torch.ones_like(is_N, dtype=torch.long),
                                            torch.full_like(is_N, 2, dtype=torch.long)))
            return modulated * torch.sigmoid(self.gate_embed(atype)).unsqueeze(1)

    class FiLM(nn.Module):
        def __init__(self, atom_dim, energy_dim):
            super().__init__()
            self.to_gamma = nn.Linear(atom_dim, energy_dim)
            self.to_beta = nn.Linear(atom_dim, energy_dim)

        def forward(self, h, e):
            return self.to_gamma(h).unsqueeze(1) * e.unsqueeze(0) + self.to_beta(h).unsqueeze(1)

    class GNN_Stage1(nn.Module):
        def __init__(self, node_feat_dim=104, edge_feat_dim=68, hidden_dim=128, n_layers=4, n_heads=4,
                     n_energies=3000, n_orbitals=8, energy_dim=64):
            super().__init__()
            total_e = 2 * energy_dim
            self.node_embed = nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
            self.edge_embed = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.PReLU())
            self.short_layers = nn.ModuleList([GTLayer(hidden_dim, hidden_dim, n_heads) for _ in range(n_layers)])
            self.long_layers = nn.ModuleList([GTLayer(hidden_dim, hidden_dim, n_heads) for _ in range(2)])
            self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
            self.energy_encoder = MorletEnergyEncoder(energy_dim)
            self.film = FiLM(hidden_dim, total_e)
            self.atom_type_gate = AtomTypeGate(total_e)
            self.head_dos = nn.Sequential(nn.Linear(total_e, total_e * 2), nn.PReLU(),
                                          nn.Linear(total_e * 2, total_e), nn.PReLU(),
                                          nn.Linear(total_e, n_orbitals))
            self.head_scale_s = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                              nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 2), nn.Softplus())
            self.head_scale_p = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                              nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 6), nn.Softplus())
            self.head_descriptors = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                                  nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 6))

        def forward(self, x_in, ei_s, ea_s, ei_l, ea_l, energy_grid, atom_chunk=None):
            x = self.node_embed(x_in)
            ea = self.edge_embed(ea_s)
            xs = x
            for layer in self.short_layers:
                xs = layer(xs, ei_s, ea)
            ea_long = self.edge_embed(ea_l)
            xl = x
            for layer in self.long_layers:
                xl = layer(xl, ei_l, ea_long)
            x_fused = self.fuse(torch.cat([xs, xl], dim=-1))
            e = self.energy_encoder(energy_grid)
            scale = torch.cat([self.head_scale_s(x_fused), self.head_scale_p(x_fused)], dim=-1)
            desc = self.head_descriptors(x_fused)
            n = x_fused.shape[0]
            step = atom_chunk or n
            outs = []
            for a in range(0, n, step):
                b = min(n, a + step)
                mod = self.film(x_fused[a:b], e)
                mod = self.atom_type_gate(x_in[a:b], mod)
                dos_norm = self.head_dos(mod)
                outs.append(dos_norm * scale[a:b].unsqueeze(1))
            return torch.cat(outs, dim=0), scale, desc

    class ResBlock1D(nn.Module):
        def __init__(self, channels, kernel_size=15):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(nn.Conv1d(channels, channels, kernel_size, padding=pad), nn.BatchNorm1d(channels), nn.PReLU(),
                                     nn.Conv1d(channels, channels, kernel_size, padding=pad), nn.BatchNorm1d(channels))
            self.act = nn.PReLU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class SharpeningNet(nn.Module):
        def __init__(self, n_channels=8, base_ch=32, kernel=15, n_grid=3000):
            super().__init__()
            pad, K = kernel // 2, kernel
            self.enc1 = nn.Sequential(nn.Conv1d(n_channels, base_ch, K, padding=pad), nn.PReLU(), ResBlock1D(base_ch, K))
            self.enc2 = nn.Sequential(nn.Conv1d(base_ch, base_ch * 2, K, stride=2, padding=pad), nn.PReLU(), ResBlock1D(base_ch * 2, K))
            self.enc3 = nn.Sequential(nn.Conv1d(base_ch * 2, base_ch * 4, K, stride=2, padding=pad), nn.PReLU(), ResBlock1D(base_ch * 4, K))
            self.bottleneck = nn.Sequential(ResBlock1D(base_ch * 4, K), ResBlock1D(base_ch * 4, K))
            self.dec3 = nn.Sequential(nn.ConvTranspose1d(base_ch * 4, base_ch * 2, K, stride=2, padding=pad, output_padding=1),
                                      nn.PReLU(), ResBlock1D(base_ch * 2, K))
            self.dec2 = nn.Sequential(nn.ConvTranspose1d(base_ch * 4, base_ch, K, stride=2, padding=pad, output_padding=1),
                                      nn.PReLU(), ResBlock1D(base_ch, K))
            self.out = nn.Conv1d(base_ch * 2, n_channels, 1)

        def forward(self, x_smooth):
            e1 = self.enc1(x_smooth)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            b = self.bottleneck(e3)
            d3 = self.dec3(b)
            if d3.shape[-1] != e2.shape[-1]:
                d3 = F.interpolate(d3, size=e2.shape[-1])
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            if d2.shape[-1] != e1.shape[-1]:
                d2 = F.interpolate(d2, size=e1.shape[-1])
            return x_smooth + self.out(torch.cat([d2, e1], dim=1))

    return GNN_Stage1, SharpeningNet, TransformerConvLite


# ------------------------------------------------------------------ predictor
class Predictor:
    def __init__(self, model_dir):
        _require()
        import torch
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2))))
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model_config.json")) as f:
            self.cfg = json.load(f)
        s1 = self.cfg["stage1"]
        s2 = self.cfg["stage2"]
        GNN_Stage1, SharpeningNet, _ = _define_networks()
        self.stage1 = GNN_Stage1(node_feat_dim=s1["node_feat_dim"], edge_feat_dim=s1["edge_feat_dim"],
                                 hidden_dim=s1["hidden_dim"], n_layers=s1["n_layers"], n_heads=s1["n_heads"],
                                 n_energies=s1["n_energies"], n_orbitals=s1["n_orbitals"], energy_dim=s1["energy_dim"])
        self.stage1.load_state_dict(torch.load(os.path.join(model_dir, "stage1_best.pt"), map_location="cpu"), strict=True)
        self.stage1.eval()
        self.stage2 = SharpeningNet(n_channels=s2["n_channels"], base_ch=s2["base_ch"], kernel=s2["kernel"], n_grid=s2["n_grid"])
        self.stage2.load_state_dict(torch.load(os.path.join(model_dir, "stage2_best.pt"), map_location="cpu"), strict=True)
        self.stage2.eval()
        self.energy = np.load(os.path.join(model_dir, "energy_grid.npy")).astype(np.float64)
        self.energy_t = torch.tensor(self.energy, dtype=torch.float32).unsqueeze(1)
        self.scale_up = float(self.cfg["normalizer"]["scale_up"])
        self.scale_dn = float(self.cfg["normalizer"]["scale_dn"])

    def predict(self, atoms, progress=None):
        import torch
        n = len(atoms)
        if progress:
            progress("features", "Building graph (%d atoms)" % n, 5)
        node, ei_s, ea_s, ei_l, ea_l = build_graph(atoms)
        x = torch.tensor(node, dtype=torch.float32)
        ei_s_t = torch.tensor(ei_s, dtype=torch.long)
        ei_l_t = torch.tensor(ei_l, dtype=torch.long)
        ea_s_t = torch.tensor(ea_s, dtype=torch.float32)
        ea_l_t = torch.tensor(ea_l, dtype=torch.float32)
        if progress:
            progress("stage1", "Stage 1: graph network", 20)
        with torch.no_grad():
            smooth, _scale, _desc = self.stage1(x, ei_s_t, ea_s_t, ei_l_t, ea_l_t, self.energy_t, atom_chunk=STAGE1_ATOM_CHUNK)
            if progress:
                progress("stage2", "Stage 2: sharpening network", 60)
            chunks = []
            for a in range(0, n, STAGE2_BATCH):
                chunk = smooth[a:a + STAGE2_BATCH].permute(0, 2, 1)
                chunks.append(self.stage2(chunk).permute(0, 2, 1))
            sharp = torch.cat(chunks, dim=0).numpy().astype(np.float64)
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
