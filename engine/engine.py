#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padarth Khoj - analysis engine for the Electron app.

This is the backend of the original Tkinter notebook (padarth_khoj.py) with the
GUI removed and a JSON-lines protocol added so that Electron's main process can
drive it.

Protocol (default when started with no arguments, or with --serve):
    stdin  : one JSON object per line   {"id": "...", "cmd": "...", "args": {...}}
    stdout : one JSON object per line   {"id": "...", "type": "result"|"progress"|"error", ...}
    stderr : free-form log text (never parsed)

Commands: ping, init_models, parse_vasp_header, parse_structure, run_analysis, shutdown

One-shot CLI (for scripting / debugging):
    python engine.py --init-models
    python engine.py --run --doscar /path/DOSCAR --reaction OER --ml-method SVR
    python engine.py --parse-structure /path/CONTCAR

Numerical behaviour is identical to the notebook (same training data handling,
same descriptors, same overpotential formulas). The deliberate differences are
listed in the "NOTEBOOK DIFFERENCES" block below and in the README.
"""
import argparse
import io
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ENGINE_VERSION = "1.1.0"

# NOTEBOOK DIFFERENCES (intentional)
#  1. ORR descriptors are passed to the model in the same column order as orr.csv
#     [D(EF), R_O] = [pz_fermi, r_occ].  The notebook passed [r_occ, pz_fermi],
#     which makes the ORR predictions meaningless (training R2 0.86 -> -7.0).
#  2. read_contcar() returns *fractional* coordinates.  The notebook returned the raw
#     file coordinates, which is wrong for Cartesian CONTCARs.  (Identical for Direct.)
#  3. ISPIN = 1 DOSn files (10 columns) use column 3 (p_z).  The notebook always used
#     columns 5/6, which are d_xy / d_yz for ISPIN = 1.  (Identical for ISPIN = 2.)
#  4. OUTCAR is optional: E-fermi / ISPIN fall back to the DOSCAR header (a note is shown).
#  5. A carbon whose neighbours all lack DOS data falls back to the single-site branch
#     instead of producing an infinite overpotential.
#  6. numpy 2 / pandas 3 compatible (np.trapezoid, sep=r"\s+").
#  7. UI-only extras (Atom_Number for ORR, Element, Best_Partner) are added to the
#     records sent to the app; the CSV / Excel files keep the notebook's columns.

# Safe integration function compatible with both NumPy 1.x and 2.x
trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))

# --------------------------------------------------------------------------- utils


def resource_path(relative_path):
    """Absolute path to a bundled resource (works from source and from PyInstaller)."""
    base_path = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


_progress_cb = None


def progress(stage, message, percent=None):
    if _progress_cb is not None:
        try:
            _progress_cb({"stage": stage, "message": message, "percent": percent})
        except Exception:
            pass


def to_jsonable(obj):
    """Recursively convert numpy / pandas / NaN values so json.dumps(allow_nan=False) works."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


# =========================================================================
# === MODEL TRAINING SECTION (same data handling as the notebook)       ===
# =========================================================================

trained_models = {}
training_info = {}

MODEL_SPECS = {
    "OER": {
        "csv": "oer.csv",
        "n_features": 4,
        "features": ["D(EF)[S]", "R_O[S]", "D(EF)[D]", "R_O[D]"],
        "target": "GO_GOH",
    },
    "ORR": {
        "csv": "orr.csv",
        "n_features": 2,
        "features": ["D(EF)", "R_O"],
        "target": "GOH",
    },
}


def get_oer_csv_path():
    path = resource_path(os.path.join("data", "oer.csv"))
    if os.path.exists(path):
        return path
    return resource_path(os.path.join("data", "oer_sama.csv"))  # legacy name used by the notebook


def _csv_path(reaction):
    return get_oer_csv_path() if reaction == "OER" else resource_path(os.path.join("data", MODEL_SPECS[reaction]["csv"]))


def _fit(reaction, kind):
    spec = MODEL_SPECS[reaction]
    csv_path = _csv_path(reaction)
    df_train = pd.read_csv(csv_path)
    nf = spec["n_features"]
    if df_train.shape[1] < nf + 1:
        raise ValueError("%s has %d columns, expected at least %d" % (os.path.basename(csv_path), df_train.shape[1], nf + 1))
    if df_train.iloc[:, : nf + 1].isna().any().any():
        raise ValueError("%s contains empty cells" % os.path.basename(csv_path))

    x_train_orig = df_train.iloc[:, 0:nf].values
    y_train_orig = df_train.iloc[:, nf].values.reshape(-1, 1)

    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(x_train_orig)
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train_orig)

    regressor = SVR(kernel="linear") if kind == "SVR" else LinearRegression()
    regressor.fit(X_train_scaled, y_train_scaled.ravel())

    key = "%s_%s" % (reaction, kind)
    trained_models[key] = {"model": regressor, "scaler_X": sc_X, "scaler_y": sc_y}

    pred = sc_y.inverse_transform(regressor.predict(X_train_scaled).reshape(-1, 1)).ravel()
    training_info[key] = {
        "csv": os.path.basename(csv_path),
        "n_samples": int(x_train_orig.shape[0]),
        "features": list(df_train.columns[:nf]),
        "target": str(df_train.columns[nf]),
        "r2_train": float(r2_score(y_train_orig.ravel(), pred)),
    }


def train_oer_model():
    _fit("OER", "SVR")


def train_orr_model():
    _fit("ORR", "SVR")


def train_oer_mlr_model():
    _fit("OER", "MLR")


def train_orr_mlr_model():
    _fit("ORR", "MLR")


def initialize_all_models():
    train_oer_model()
    train_orr_model()
    train_oer_mlr_model()
    train_orr_mlr_model()


def ensure_models():
    if len(trained_models) < 4:
        initialize_all_models()


# =========================================================================
# === STRUCTURE FILES (POSCAR / CONTCAR / XYZ)                          ===
# =========================================================================

_EL_RE = re.compile(r"[A-Z][a-z]?")


def _element_of(token):
    m = _EL_RE.match(token.strip())
    return m.group(0) if m else token.strip()


def _symbols_from_outcar(folder, expected):
    """VASP4-style POSCARs have no symbol line; try OUTCAR's VRHFIN lines."""
    path = os.path.join(folder, "OUTCAR")
    if not os.path.exists(path):
        return None
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "VRHFIN" in line:
                    m = re.search(r"VRHFIN\s*=\s*([A-Za-z]+)", line)
                    if m:
                        out.append(m.group(1))
                    if len(out) >= expected:
                        break
    except Exception:
        return None
    return out if len(out) == expected else None


def _read_poscar(filename):
    """Parse a VASP POSCAR/CONTCAR. Returns a dict or None."""
    if not filename or not os.path.exists(filename):
        return None
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    if len(lines) < 8:
        return None

    title = lines[0].strip()
    scale = float(lines[1].split()[0])
    lat_raw = np.array([list(map(float, lines[i].split()[:3])) for i in range(2, 5)])
    if scale < 0:  # negative scale = target cell volume
        scale = (-scale / abs(np.linalg.det(lat_raw))) ** (1.0 / 3.0)
    lattice = lat_raw * scale

    tokens5 = lines[5].split()
    if tokens5 and tokens5[0][0].isalpha():
        species = [_element_of(t) for t in tokens5]
        natoms = list(map(int, lines[6].split()))
        idx = 7
    else:  # VASP 4: counts directly, no symbols
        species = None
        natoms = list(map(int, tokens5))
        idx = 6
    total = int(sum(natoms))
    if species is None:
        species = _symbols_from_outcar(os.path.dirname(os.path.abspath(filename)), len(natoms)) or ["X"] * len(natoms)
    if len(species) != len(natoms):
        raise ValueError("species / counts mismatch in %s" % os.path.basename(filename))

    selective = lines[idx].strip().lower().startswith("s")
    if selective:
        idx += 1
    coord_mode = lines[idx].strip().lower()
    is_cartesian = coord_mode[:1] in ("c", "k")
    start = idx + 1
    if start + total > len(lines):
        raise ValueError("not enough coordinate lines in %s (expected %d)" % (os.path.basename(filename), total))

    raw = []
    for ln in lines[start : start + total]:
        parts = ln.split()
        if len(parts) < 3:
            raise ValueError("malformed coordinate line: %r" % ln)
        raw.append(list(map(float, parts[:3])))
    raw = np.array(raw)

    if is_cartesian:
        cart = raw * scale
        frac = cart @ np.linalg.inv(lattice)
    else:
        frac = raw
        cart = frac @ lattice

    symbols = []
    for sp, n in zip(species, natoms):
        symbols.extend([sp] * n)

    return {
        "title": title,
        "lattice": lattice,
        "species": species,
        "counts": natoms,
        "symbols": symbols,
        "raw": raw,
        "frac": frac,
        "cart": cart,
        "direct": not is_cartesian,
        "selective": selective,
        "total": total,
    }


def _formula(symbols):
    order, counts = [], {}
    for s in symbols:
        if s not in counts:
            order.append(s)
            counts[s] = 0
        counts[s] += 1
    return " ".join("%s%s" % (s, counts[s] if counts[s] > 1 else "") for s in order)


def _read_xyz(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f.read().splitlines()]
    n = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    lattice = None
    m = re.search(r'Lattice\s*=\s*"([^"]+)"', comment, flags=re.I)
    if m:
        vals = list(map(float, m.group(1).split()))
        if len(vals) == 9:
            lattice = np.array(vals).reshape(3, 3)
    symbols, cart = [], []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        symbols.append(_element_of(p[0]))
        cart.append(list(map(float, p[1:4])))
    cart = np.array(cart)
    frac = cart @ np.linalg.inv(lattice) if lattice is not None else None
    return {"title": comment.strip(), "lattice": lattice, "symbols": symbols, "cart": cart, "frac": frac}


def find_structure_file(folder):
    for name in ("CONTCAR", "POSCAR"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    try:
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".vasp", ".contcar", ".poscar")):
                return os.path.join(folder, fn)
    except OSError:
        pass
    return None


def parse_structure_file(path):
    """Structure for the 3D viewer. Atom indices are 1-based (VASP / DOSn numbering)."""
    if not path or not os.path.exists(path):
        return {"success": False, "error": "Structure file not found: %s" % path}
    try:
        if path.lower().endswith(".xyz"):
            d = _read_xyz(path)
            fmt = "xyz"
        else:
            d = _read_poscar(path)
            fmt = "vasp"
            if d is None:
                return {"success": False, "error": "Could not parse %s (file too short?)" % os.path.basename(path)}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": "Could not parse %s: %s" % (os.path.basename(path), e)}

    symbols = d["symbols"]
    lattice = d["lattice"]
    cart = d["cart"]
    frac = d["frac"]
    return {
        "success": True,
        "path": path,
        "name": os.path.basename(path),
        "format": fmt,
        "title": d.get("title", ""),
        "natoms": len(symbols),
        "formula": _formula(symbols),
        "symbols": symbols,
        "lattice": lattice.tolist() if lattice is not None else None,
        "cart": cart.tolist(),
        "frac": frac.tolist() if frac is not None else None,
        "periodic": lattice is not None,
        "n_carbon": int(sum(1 for s in symbols if s == "C")),
    }


# --------------------- notebook-compatible CONTCAR helpers -------------------


def read_contcar(filename):
    """lattice, fractional coordinates, 0-based carbon indices (or None, None, None)."""
    try:
        d = _read_poscar(filename)
    except Exception:  # noqa: BLE001
        return None, None, None
    if d is None or "C" not in d["species"]:
        return None, None, None
    c_symbol_index = d["species"].index("C")
    start_indices = [0] + list(np.cumsum(d["counts"][:-1]))
    carbon_start_index = int(start_indices[c_symbol_index])
    num_carbon = d["counts"][c_symbol_index]
    carbon_indices = list(range(carbon_start_index, carbon_start_index + num_carbon))
    return d["lattice"], d["frac"], carbon_indices


def min_image_distance(frac1, frac2, lattice):
    df = frac2 - frac1
    df -= np.rint(df)
    dc = np.dot(df, lattice)
    return np.linalg.norm(dc)


def find_adjacent_pairs(frac_coords, carbon_indices, lattice, cutoff=1.6):
    """Pairs (idx1, idx2) of carbon atoms closer than cutoff (minimum-image convention)."""
    if frac_coords is None or carbon_indices is None or lattice is None:
        return None
    if cutoff <= 0:
        return []
    ci = np.array(carbon_indices, dtype=int)
    valid = ci < len(frac_coords)
    ci = ci[valid]
    fc = np.asarray(frac_coords)[ci]
    pairs = []
    n = len(ci)
    for i in range(n - 1):
        df = fc[i + 1 :] - fc[i]
        df -= np.rint(df)
        dist = np.linalg.norm(df @ lattice, axis=1)
        for j in np.nonzero(dist < cutoff)[0]:
            pairs.append((int(ci[i]), int(ci[i + 1 + j])))
    return pairs


# =========================================================================
# === DOSCAR HANDLING                                                    ===
# =========================================================================


def _parse_outcar(outcar_path):
    """Returns (efermi or None, ispin or None). Last occurrence wins, as in the notebook."""
    efermi, nspin = None, None
    if not os.path.exists(outcar_path):
        return efermi, nspin
    try:
        with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "E-fermi" in line:
                    parts = line.replace(":", " ").split()
                    if "E-fermi" in parts:
                        i = parts.index("E-fermi")
                        try:
                            efermi = float(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
                elif "ISPIN" in line:
                    parts = line.replace("=", " ").split()
                    if "ISPIN" in parts:
                        i = parts.index("ISPIN")
                        try:
                            nspin = int(parts[i + 1])
                        except (ValueError, IndexError):
                            pass
    except Exception:  # noqa: BLE001
        pass
    return efermi, nspin


def _read_doscar_head(doscar_path, extra_lines=2):
    """First 7 + NEDOS + a few lines (header, total DOS tail, first atom block line)."""
    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        head = list(itertools.islice(f, 7))
        natom = int(head[0].split()[0])
        nedos = int(head[5].split()[2])
        rest = list(itertools.islice(f, nedos + extra_lines))  # remaining total-DOS lines + atom header + first atom line
    return head, rest, natom, nedos


def _nspin_from_cols(ncols):
    if ncols in (19, 33):
        return 2
    if ncols in (10, 17):
        return 1
    return None


def parse_vasp_header(doscar_path):
    folder = os.path.dirname(os.path.abspath(doscar_path))
    if not os.path.exists(doscar_path):
        return {"success": False, "error": "DOSCAR file not found: %s" % doscar_path}
    warnings = []
    try:
        head, rest, natom, nedos = _read_doscar_head(doscar_path)
        emax, emin = float(head[5].split()[0]), float(head[5].split()[1])
        efermi_hdr = float(head[5].split()[3])
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": "Could not parse the DOSCAR header: %s" % e}

    total_cols = len(head[6].split()) if len(head) > 6 else 0
    pdos_cols = None
    if len(rest) >= nedos + 1:  # rest[nedos - 1] is the first atom header, rest[nedos] its first data line
        pdos_cols = len(rest[nedos].split())
    has_pdos = pdos_cols is not None
    nspin_cols = _nspin_from_cols(pdos_cols) if pdos_cols else None
    if not has_pdos:
        warnings.append("DOSCAR contains no projected DOS. Re-run VASP with LORBIT = 11.")
    elif nspin_cols is None:
        warnings.append("Projected DOS has %d columns; expected 10 / 19 (LORBIT = 11). Re-run with LORBIT = 11." % pdos_cols)

    outcar = os.path.join(folder, "OUTCAR")
    ef_out, ispin_out = _parse_outcar(outcar)
    nspin = nspin_cols or ispin_out or (2 if total_cols == 5 else 1)
    if ispin_out and nspin_cols and ispin_out != nspin_cols:
        warnings.append("OUTCAR says ISPIN = %d but the DOSCAR columns indicate ISPIN = %d; using the DOSCAR." % (ispin_out, nspin_cols))

    struct_path = find_structure_file(folder)
    contcar = os.path.join(folder, "CONTCAR")
    struct_summary = None
    if struct_path:
        s = parse_structure_file(struct_path)
        if s.get("success"):
            struct_summary = {"name": s["name"], "natoms": s["natoms"], "formula": s["formula"], "n_carbon": s["n_carbon"]}
            if s["natoms"] != natom:
                warnings.append("DOSCAR has %d atoms but %s has %d atoms." % (natom, s["name"], s["natoms"]))
        else:
            warnings.append(s.get("error", "structure file unreadable"))
    if not os.path.exists(contcar):
        warnings.append("CONTCAR not found in this folder (required for OER).")
    if not os.path.exists(outcar):
        warnings.append("OUTCAR not found: E-fermi and ISPIN will be read from the DOSCAR header.")

    return {
        "success": True,
        "folder": folder,
        "natoms": natom,
        "nedos": nedos,
        "emax": emax,
        "emin": emin,
        "efermi": ef_out if ef_out is not None else efermi_hdr,
        "efermi_source": "OUTCAR" if ef_out is not None else "DOSCAR header",
        "nspin": nspin,
        "pdos_columns": pdos_cols,
        "has_pdos": has_pdos,
        "has_outcar": os.path.exists(outcar),
        "has_contcar": os.path.exists(contcar),
        "structure_path": struct_path,
        "structure": struct_summary,
        "warnings": warnings,
    }


def split_doscar(doscar_path, output_folder):
    """Split DOSCAR into DOS1..DOSn (E - E_F shifted; spin-down negated), as in the notebook.

    Returns (n_atoms_written, message_or_None)."""
    outcar_path = os.path.join(output_folder, "OUTCAR")
    efermi_out, nspin_out = _parse_outcar(outcar_path)

    try:
        head, rest, natom, nedos = _read_doscar_head(doscar_path)
        efermi_hdr = float(head[5].split()[3])
    except Exception as e:  # noqa: BLE001
        return 0, "Could not parse DOSCAR header: %s" % e

    efermi = efermi_out if efermi_out is not None else efermi_hdr
    pdos_cols = len(rest[nedos].split()) if len(rest) >= nedos + 1 else None
    if pdos_cols is None:
        return 0, "DOSCAR contains no projected DOS (set LORBIT = 11 in the INCAR and re-run VASP)."
    nspin = _nspin_from_cols(pdos_cols)
    note = None
    if nspin is None:
        return 0, "Projected DOS has %d columns; l,m-decomposed PDOS (LORBIT = 11) is required." % pdos_cols
    if nspin_out is not None and nspin_out != nspin:
        note = "OUTCAR says ISPIN = %d but the DOSCAR columns indicate ISPIN = %d; using the DOSCAR." % (nspin_out, nspin)

    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_line = 6 + nedos
    try:
        pos_data = _read_poscar(os.path.join(output_folder, "CONTCAR"))
    except Exception:  # noqa: BLE001
        pos_data = None
    positions = pos_data["raw"] if pos_data is not None else None

    ncols_out = 19 if nspin == 2 else 10
    row_fmt = " ".join(["%12.8f"] * ncols_out)

    atom_index = 1
    split_count = 0
    while current_line < len(lines):
        if current_line + nedos + 1 > len(lines):
            break
        block = lines[current_line + 1 : current_line + 1 + nedos]
        try:
            arr = np.array([ln.split() for ln in block], dtype=float)
        except ValueError as e:
            return split_count, "Atom %d: malformed PDOS block (%s)" % (atom_index, e)
        if arr.shape[1] < (19 if nspin == 2 else 10):
            return split_count, "Atom %d: PDOS block has %d columns" % (atom_index, arr.shape[1])

        out = np.empty((nedos, ncols_out))
        out[:, 0] = arr[:, 0] - efermi
        if nspin == 2:
            out[:, 1::2] = arr[:, 1:18:2]
            out[:, 2::2] = -arr[:, 2:19:2]
        else:
            out[:, 1:10] = arr[:, 1:10]

        dos_file_path = os.path.join(output_folder, "DOS%d" % atom_index)
        with open(dos_file_path, "w") as outf:
            if positions is not None and len(positions) >= atom_index:
                pos = positions[atom_index - 1]
                outf.write("# %12.8f %12.8f %12.8f\n" % (pos[0], pos[1], pos[2]))
            outf.write("\n".join(row_fmt % tuple(r) for r in out) + "\n")

        current_line += 1 + nedos
        atom_index += 1
        split_count += 1
        if split_count % 10 == 0 or split_count == natom:
            progress("split", "Split PDOS for atom %d / %d" % (split_count, natom), 5 + 40.0 * split_count / max(natom, 1))

    if natom != split_count:
        return split_count, "Warning: Expected %d atoms, but split %d PDOS files." % (natom, split_count)
    return natom, note


# =========================================================================
# === DESCRIPTORS AND OVERPOTENTIALS                                     ===
# =========================================================================


def calculate_overpotential_orr(del_goh):
    if del_goh < 0.13914:
        return 0.3672 - 0.8268 * del_goh
    else:
        return 0.11252 + 0.98707 * del_goh


def calculate_overpotential_oer(del_goh_minus_del_go):
    if del_goh_minus_del_go < 2.31212:
        return 2.3612 - 0.6811 * del_goh_minus_del_go
    else:
        return 2.032 + 1 * del_goh_minus_del_go


def compute_pz_parameters(dos_file):
    """pz_fermi (p_z DOS at the last point below E_F) and pz_occ (integral of p_z DOS for E < 0)."""
    try:
        dos_data = pd.read_csv(dos_file, header=None, sep=r"\s+", comment="#")
        ncols = dos_data.shape[1]
        if ncols >= 19:  # ISPIN = 2: pz_up = col 5, pz_down = col 6 (stored negative)
            pz_up_col, pz_down_col = 5, 6
        elif ncols >= 10:  # ISPIN = 1: single p_z column
            pz_up_col, pz_down_col = 3, None
        else:
            return 0.0, 0.0, "Error: File '%s' has insufficient columns." % os.path.basename(dos_file)

        energy = dos_data.iloc[:, 0].values
        if pz_down_col is None:
            pz = dos_data.iloc[:, pz_up_col].values
        else:
            pz = dos_data.iloc[:, pz_up_col].values - dos_data.iloc[:, pz_down_col].values

        negative_mask = energy < 0
        if np.any(negative_mask):
            neg_indices = np.where(negative_mask)[0]
            if neg_indices.size > 0:
                idx_of_max_neg_energy = neg_indices[np.argmax(energy[neg_indices])]
                pz_fermi = pz[idx_of_max_neg_energy]
                sort_indices = np.argsort(energy[negative_mask])
                pz_occ = trapz_func(pz[negative_mask][sort_indices], energy[negative_mask][sort_indices])
                return pz_fermi, pz_occ, None
        return 0.0, 0.0, "Warning: File '%s' has no points below 0 eV." % os.path.basename(dos_file)
    except Exception as e:  # noqa: BLE001
        return 0.0, 0.0, str(e)


R_OCC_SHIFT = 0.47064  # r_occ = pz_occ - 0.47064 (as in the notebook)


def _dos_number(path):
    m = re.search(r"DOS(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else None


def analyze_orr_ml(folder_path, n, model, sc_X, sc_y, method="SVR"):
    dos_files = [os.path.join(folder_path, "DOS%d" % i) for i in range(1, n + 1) if os.path.exists(os.path.join(folder_path, "DOS%d" % i))]
    if not dos_files:
        return "No DOS files found.", None, None, []

    rows, warnings = [], []
    for k, dos_file in enumerate(dos_files):
        pz_fermi, pz_occ, err = compute_pz_parameters(dos_file)
        if err:
            warnings.append("Skipping %s due to an error: %s" % (os.path.basename(dos_file), err))
            continue

        r_occ = pz_occ - R_OCC_SHIFT
        # Feature order must match orr.csv: [D(EF), R_O] = [pz_fermi, r_occ]
        input_features = np.array([[pz_fermi, r_occ]])
        scaled_features = sc_X.transform(input_features)

        predicted_del_goh_scaled = model.predict(scaled_features)
        del_goh = sc_y.inverse_transform(predicted_del_goh_scaled.reshape(-1, 1))[0][0]

        overpotential = calculate_overpotential_orr(del_goh)
        rows.append([dos_file, pz_occ, pz_fermi, r_occ, del_goh, overpotential])
        if (k + 1) % 20 == 0:
            progress("predict", "Predicted %d / %d atoms" % (k + 1, len(dos_files)), 50 + 40.0 * (k + 1) / len(dos_files))

    results_df = pd.DataFrame(rows, columns=["File", "Pz_Occupancy", "Pz_at_Fermi", "r_Occupancy", "Del_Goh", "Overpotential"])
    if results_df.empty:
        return "No valid DOS files were processed.", None, None, []

    output_str = "Analysis for ORR using %s\n\n" % method
    if warnings:
        output_str += "--- Warnings ---\n" + "\n".join(warnings) + "\n\n"
    output_str += results_df[["File", "Del_Goh", "Overpotential"]].to_string()

    excel_file_path = os.path.join(folder_path, "dos_results.xlsx")
    results_df.to_excel(excel_file_path, index=False)

    csv_file_path = os.path.join(folder_path, "overpotentialsORR_%s.csv" % method)
    results_df.to_csv(csv_file_path, index=False)

    records = results_df.to_dict(orient="records")
    for rec in records:  # UI-only extra (not written to CSV/Excel)
        rec["Atom_Number"] = _dos_number(rec["File"])
    return output_str, excel_file_path, csv_file_path, records


def analyze_oer_ml(folder_path, n_atoms, model, sc_X, sc_y, method="SVR"):
    contcar_file = os.path.join(folder_path, "CONTCAR")
    lattice, coords, carbon_indices = read_contcar(contcar_file)
    if lattice is None or coords is None or carbon_indices is None:
        return "Error: CONTCAR file not found or could not be parsed.", None, None, []

    pairs_0based = find_adjacent_pairs(coords, carbon_indices, lattice, cutoff=1.6)
    if pairs_0based is None:
        return "Error: Could not find adjacent carbon pairs.", None, None, []

    s_params = {}
    existing_dos_indices = [idx for idx in carbon_indices if os.path.exists(os.path.join(folder_path, "DOS%d" % (idx + 1)))]

    for k, original_idx in enumerate(existing_dos_indices):
        dos_file = os.path.join(folder_path, "DOS%d" % (original_idx + 1))
        pz_fermi_s, pz_occ_s, err = compute_pz_parameters(dos_file)
        if not err:
            r_occ_s = pz_occ_s - R_OCC_SHIFT
            s_params[original_idx] = {"pz_fermi_s": pz_fermi_s, "pz_occ_s": pz_occ_s, "r_occ_s": r_occ_s}
        if (k + 1) % 20 == 0:
            progress("descriptors", "Descriptors for %d / %d carbons" % (k + 1, len(existing_dos_indices)), 45 + 20.0 * (k + 1) / len(existing_dos_indices))

    if not s_params:
        return "No valid DOS files for carbon atoms were processed.", None, None, []

    original_idx_to_list_idx = {orig_idx: list_idx for list_idx, orig_idx in enumerate(carbon_indices)}
    adj_list = {i: [] for i in range(len(carbon_indices))}
    for idx1_orig, idx2_orig in pairs_0based:
        if idx1_orig in original_idx_to_list_idx and idx2_orig in original_idx_to_list_idx:
            l1 = original_idx_to_list_idx[idx1_orig]
            l2 = original_idx_to_list_idx[idx2_orig]
            adj_list[l1].append(l2)
            adj_list[l2].append(l1)

    results_for_csv = []
    partners = {}
    for i in range(len(carbon_indices)):
        original_idx = carbon_indices[i]
        if original_idx not in s_params:
            continue

        pz_fermi_s_i = s_params[original_idx]["pz_fermi_s"]
        pz_occ_s_i_orig = s_params[original_idx]["pz_occ_s"]
        r_occ_s_i = s_params[original_idx]["r_occ_s"]

        neighbor_list_indices = adj_list[original_idx_to_list_idx[original_idx]]

        max_del_g_for_i = -np.inf
        best_pz_fermi_d = pz_fermi_s_i
        best_pz_occ_d_orig = pz_occ_s_i_orig
        best_r_occ_d = r_occ_s_i
        best_partner = None

        for j in neighbor_list_indices:
            neighbor_original_idx = carbon_indices[j]
            if neighbor_original_idx not in s_params:
                continue
            pz_fermi_s_j = s_params[neighbor_original_idx]["pz_fermi_s"]
            pz_occ_s_j_orig = s_params[neighbor_original_idx]["pz_occ_s"]
            r_occ_s_j = s_params[neighbor_original_idx]["r_occ_s"]

            pz_fermi_d_pair = pz_fermi_s_i + pz_fermi_s_j
            pz_occ_d_pair_orig = pz_occ_s_i_orig + pz_occ_s_j_orig
            r_occ_d_pair = r_occ_s_i + r_occ_s_j

            input_features = np.array([[pz_fermi_s_i, r_occ_s_i, pz_fermi_d_pair, r_occ_d_pair]])
            scaled_features = sc_X.transform(input_features)
            predicted_del_g_scaled = model.predict(scaled_features)
            predicted_del_g = sc_y.inverse_transform(predicted_del_g_scaled.reshape(-1, 1))[0][0]

            if predicted_del_g > max_del_g_for_i:
                max_del_g_for_i = predicted_del_g
                best_pz_fermi_d = pz_fermi_d_pair
                best_pz_occ_d_orig = pz_occ_d_pair_orig
                best_r_occ_d = r_occ_d_pair
                best_partner = neighbor_original_idx + 1

        if best_partner is not None:
            calculated_del_g = max_del_g_for_i
        else:  # no neighbour with data: single-site descriptors used for the D slot
            input_features = np.array([[pz_fermi_s_i, r_occ_s_i, pz_fermi_s_i, r_occ_s_i]])
            scaled_features = sc_X.transform(input_features)
            predicted_del_g_scaled = model.predict(scaled_features)
            calculated_del_g = sc_y.inverse_transform(predicted_del_g_scaled.reshape(-1, 1))[0][0]

        overpotential = calculate_overpotential_oer(calculated_del_g)

        results_for_csv.append({
            "Atom_Number": int(original_idx + 1),
            "Pz_Fermi_S": float(pz_fermi_s_i),
            "Pz_Occ_S": float(pz_occ_s_i_orig),
            "r_Occ_S": float(r_occ_s_i),
            "Pz_Fermi_D_maxG": float(best_pz_fermi_d),
            "Pz_Occ_D_maxG": float(best_pz_occ_d_orig),
            "r_Occ_D_maxG": float(best_r_occ_d),
            "Max_DelGoh_minus_DelGo": float(calculated_del_g),
            "Overpotential": float(overpotential),
        })
        partners[int(original_idx + 1)] = best_partner

    if not results_for_csv:
        return "No valid data was available for analysis. Check your input files.", None, None, []

    df_out = pd.DataFrame(results_for_csv)
    output_str = "Analysis for OER using %s\n\n" % method
    output_str += df_out[["Atom_Number", "Max_DelGoh_minus_DelGo", "Overpotential"]].to_string(index=False)

    excel_file_path = os.path.join(folder_path, "dos_results.xlsx")
    df_out.to_excel(excel_file_path, index=False)

    csv_file_path = os.path.join(folder_path, "overpotentialsOER_%s.csv" % method)
    df_out.to_csv(csv_file_path, index=False)

    records = [dict(r, Best_Partner=partners.get(r["Atom_Number"])) for r in results_for_csv]  # UI-only extra
    return output_str, excel_file_path, csv_file_path, records


# =========================================================================
# === PIPELINE                                                           ===
# =========================================================================


def _is_writable_dir(path):
    try:
        probe = os.path.join(path, ".padarth_write_test")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def run_analysis_pipeline(doscar_source, active_type, method, reaction, ml_method, output_dir=None, fallback_dir=None):
    t0 = time.time()
    if not doscar_source or not os.path.exists(doscar_source):
        return {"success": False, "error": "DOSCAR file not found: %s" % doscar_source}
    if reaction not in MODEL_SPECS:
        return {"success": False, "error": "Unknown reaction '%s' (use OER or ORR)." % reaction}
    if ml_method not in ("SVR", "MLR"):
        return {"success": False, "error": "Unknown ML method '%s' (use SVR or MLR)." % ml_method}

    # Quantum Espresso check as in the notebook
    if active_type == "DFT" and method == "Quantum Espresso":
        msg = "Quantum Espresso analysis is not yet implemented. Please select VASP or switch to ML analysis."
        return {"success": False, "error": msg, "output_str": msg}

    source_dir = os.path.dirname(os.path.abspath(doscar_source))
    if not output_dir:
        output_dir = source_dir
    notes = []

    if not _is_writable_dir(output_dir):
        if not fallback_dir:
            return {"success": False, "error": "Folder is read-only: %s" % output_dir}
        os.makedirs(fallback_dir, exist_ok=True)
        notes.append("Source folder is read-only; results were written to %s" % fallback_dir)
        output_dir = fallback_dir

    outcar_source = os.path.join(source_dir, "OUTCAR")
    contcar_source = os.path.join(source_dir, "CONTCAR")
    if reaction == "OER" and not os.path.exists(contcar_source):
        return {"success": False, "error": "CONTCAR file is required for OER analysis and was not found in: %s" % source_dir}
    if not os.path.exists(outcar_source):
        notes.append("OUTCAR not found: E-fermi and ISPIN were read from the DOSCAR header.")

    progress("prepare", "Preparing files", 2)
    doscar_path_for_analysis = os.path.join(output_dir, "DOSCAR")
    if os.path.abspath(source_dir) != os.path.abspath(output_dir):
        shutil.copy(doscar_source, doscar_path_for_analysis)
        if os.path.exists(outcar_source):
            shutil.copy(outcar_source, os.path.join(output_dir, "OUTCAR"))
        if os.path.exists(contcar_source):
            shutil.copy(contcar_source, os.path.join(output_dir, "CONTCAR"))
    else:
        doscar_path_for_analysis = os.path.abspath(doscar_source)

    n_atoms, split_error_msg = split_doscar(doscar_path_for_analysis, output_dir)
    if n_atoms == 0:
        return {"success": False, "error": "Could not split DOSCAR into individual DOS files. %s" % (split_error_msg or "")}
    if split_error_msg:
        notes.append(split_error_msg)

    progress("model", "Loading %s %s model" % (reaction, ml_method), 46)
    ensure_models()
    model_key = "%s_%s" % (reaction, ml_method)
    if model_key not in trained_models:
        return {"success": False, "error": "Model for '%s' not found or not trained." % model_key}
    md = trained_models[model_key]

    progress("analyze", "Computing p_z descriptors and predicting", 48)
    if reaction == "ORR":
        output_str, excel_path, csv_path, records = analyze_orr_ml(output_dir, n_atoms, md["model"], md["scaler_X"], md["scaler_y"], method=ml_method)
    else:
        output_str, excel_path, csv_path, records = analyze_oer_ml(output_dir, n_atoms, md["model"], md["scaler_X"], md["scaler_y"], method=ml_method)

    if not records:
        return {"success": False, "error": output_str, "output_str": output_str}

    # Element tags (UI only)
    struct = None
    sp = find_structure_file(output_dir)
    if sp:
        try:
            struct = _read_poscar(sp)
        except Exception:  # noqa: BLE001
            struct = None
    if struct is not None:
        for rec in records:
            a = rec.get("Atom_Number")
            if a and 1 <= a <= len(struct["symbols"]):
                rec["Element"] = struct["symbols"][a - 1]

    def _best(rs):
        rs = [r for r in rs if r.get("Overpotential") is not None and math.isfinite(r["Overpotential"])]
        return min(rs, key=lambda r: r["Overpotential"]) if rs else None

    best_all = _best(records)
    carbon_records = [r for r in records if r.get("Element") == "C"]
    best_c = _best(carbon_records) if carbon_records else best_all
    summary_src = best_c
    progress("done", "Analysis complete", 100)

    return {
        "success": True,
        "warning": split_error_msg,
        "notes": notes,
        "output_str": output_str,
        "excel_path": excel_path,
        "csv_path": csv_path,
        "output_dir": output_dir,
        "structure_path": sp,
        "records": records,
        "summary": {
            "total_atoms": len(records),
            "reaction": reaction,
            "method": method if active_type == "DFT" else "ML Direct",
            "ml_method": ml_method,
            "best_atom": summary_src.get("Atom_Number") if summary_src else None,
            "lowest_overpotential": summary_src.get("Overpotential") if summary_src else None,
            "best_atom_all": best_all.get("Atom_Number") if best_all else None,
            "lowest_overpotential_all": best_all.get("Overpotential") if best_all else None,
            "elapsed_s": round(time.time() - t0, 2),
        },
    }


# =========================================================================
# === COMMAND HANDLERS (JSON protocol)                                   ===
# =========================================================================


def _arg(args, *names, default=None):
    for n in names:
        if n in args and args[n] is not None:
            return args[n]
    return default


def cmd_ping(args):
    import sklearn

    return {
        "success": True,
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "models_ready": len(trained_models) == 4,
        "dos": _dos().availability(),
    }


def cmd_init_models(args):
    t0 = time.time()
    ensure_models()
    return {
        "success": True,
        "status": "ready",
        "models": sorted(trained_models.keys()),
        "training": training_info,
        "elapsed_s": round(time.time() - t0, 2),
        "message": "All ML models (OER/ORR SVR & MLR) trained successfully.",
    }


def cmd_parse_vasp_header(args):
    return parse_vasp_header(_arg(args, "doscarPath", "doscar_path", "path"))


def cmd_parse_structure(args):
    path = _arg(args, "path", "structurePath")
    if not path:
        d = _arg(args, "doscarPath", "doscar_path")
        if d:
            path = find_structure_file(os.path.dirname(os.path.abspath(d)))
            if not path:
                return {"success": False, "error": "No CONTCAR / POSCAR found next to the DOSCAR."}
    return parse_structure_file(path)


def cmd_run_analysis(args):
    return run_analysis_pipeline(
        doscar_source=_arg(args, "doscarPath", "doscar_path", "doscar"),
        active_type=_arg(args, "activeType", "active_type", default="DFT"),
        method=_arg(args, "method", default="VASP"),
        reaction=_arg(args, "reaction", default="OER"),
        ml_method=_arg(args, "mlMethod", "ml_method", default="SVR"),
        output_dir=_arg(args, "outputDir", "output_dir"),
        fallback_dir=_arg(args, "fallbackDir", "fallback_dir"),
    )


# --------------------------- PDOS predictor (stage 1 + stage 2 networks) ---------------------------

_DOS = {"atoms": None, "symbols": None, "pdos": None, "energy": None, "name": None}


def _dos():
    import dos_predictor  # lazy: torch / ase are only needed for this feature

    return dos_predictor


def _dos_curves(index):
    y = _DOS["pdos"][index]
    return [np.round(y[:, k].astype(np.float64), 6).tolist() for k in range(8)]


def cmd_dos_status(args):
    dp = _dos()
    info = dp.availability()
    mdir = resource_path("models")
    info["models_dir"] = mdir
    info["models_present"] = all(os.path.exists(os.path.join(mdir, f)) for f in ("stage1_best.pt", "stage2_best.pt", "model_config.json", "energy_grid.npy"))
    try:
        with open(os.path.join(mdir, "metrics.json")) as f:
            info["metrics"] = json.load(f)
    except Exception:  # noqa: BLE001
        pass
    info["success"] = True
    return info


def cmd_dos_load_structure(args):
    dp = _dos()
    path = _arg(args, "path")
    try:
        atoms = dp.read_structure(path)
    except dp.DosUnavailable as e:
        return {"success": False, "error": str(e), "unavailable": True}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}
    return dp.structure_summary(atoms, path)


def cmd_dos_predict(args):
    dp = _dos()
    path = _arg(args, "path")
    t0 = time.time()
    try:
        progress("read", "Reading structure", 2)
        atoms = dp.read_structure(path)
        summary = dp.structure_summary(atoms, path)
        predictor = dp.get_predictor(resource_path("models"))
    except dp.DosUnavailable as e:
        return {"success": False, "error": str(e), "unavailable": True}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}
    pdos = predictor.predict(atoms, progress=progress)
    desc = dp.orbital_descriptors(pdos, predictor.energy)
    symbols = summary["symbols"]
    _DOS.update(atoms=atoms, symbols=symbols, pdos=pdos, energy=predictor.energy, name=summary["name"])
    table = []
    for i, sym in enumerate(symbols):
        row = {"index": i + 1, "symbol": sym}
        for k in dp.DESCRIPTOR_LABELS:
            row[k] = float(desc[k][i])
        table.append(row)
    progress("done", "Predicted PDOS for %d atoms" % len(symbols), 100)
    return {
        "success": True,
        "structure": summary,
        "energy": np.round(predictor.energy, 5).tolist(),
        "labels": dp.ORBITAL_LABELS,
        "colors": dp.ORBITAL_COLORS,
        "atom_index": 1,
        "curves": _dos_curves(0),
        "atoms": table,
        "elapsed_s": round(time.time() - t0, 2),
    }


def cmd_dos_get_atom(args):
    if _DOS["pdos"] is None:
        return {"success": False, "error": "Predict a structure first."}
    i = int(_arg(args, "index", default=1)) - 1
    if not 0 <= i < _DOS["pdos"].shape[0]:
        return {"success": False, "error": "Atom index out of range."}
    return {"success": True, "atom_index": i + 1, "symbol": _DOS["symbols"][i], "curves": _dos_curves(i)}


def cmd_dos_export(args):
    dp = _dos()
    if _DOS["pdos"] is None:
        return {"success": False, "error": "Predict a structure first."}
    out = _arg(args, "outPath", "out_path")
    if not out:
        return {"success": False, "error": "No output path."}
    orbitals = _arg(args, "orbitals")
    dp.export_zip(
        out, _DOS["energy"], _DOS["pdos"], _DOS["symbols"],
        fmt=_arg(args, "format", default="png"),
        dpi=int(_arg(args, "dpi", default=150)),
        emin=float(_arg(args, "emin", default=-10.0)),
        emax=float(_arg(args, "emax", default=5.0)),
        orbitals=[int(k) for k in orbitals] if orbitals else None,
        progress=progress,
    )
    return {"success": True, "path": out, "count": int(_DOS["pdos"].shape[0]), "bytes": os.path.getsize(out)}


# --------------------------- Structure Relaxor (vasp-mace + MACE potentials) ---------------------------


def _relaxor():
    import relaxor  # lazy

    return relaxor


def cmd_relax_status(args):
    return _relaxor().status(_arg(args, "modelDir", "model_dir"))


def cmd_relax_download_model(args):
    return _relaxor().download_model(_arg(args, "modelDir", "model_dir"), progress=progress, key=_arg(args, "model", "modelKey"))


def cmd_relax_run(args):
    r = _relaxor()
    return r.run(args, parse_structure_file, progress=progress, fallback_dir=_arg(args, "fallbackDir", "fallback_dir"),
                 python=_arg(args, "pythonPath", "python_path"))


def cmd_relax_check_env(args):
    return _relaxor().check_env(_arg(args, "pythonPath", "python_path"), timeout=int(_arg(args, "timeoutS", default=240)), progress=progress)


def cmd_relax_zip(args):
    return _relaxor().zip_run(_arg(args, "runDir", "run_dir"), _arg(args, "outPath", "out_path"))


HANDLERS = {
    "ping": cmd_ping,
    "init_models": cmd_init_models,
    "parse_vasp_header": cmd_parse_vasp_header,
    "parse_structure": cmd_parse_structure,
    "run_analysis": cmd_run_analysis,
    "dos_status": cmd_dos_status,
    "dos_load_structure": cmd_dos_load_structure,
    "dos_predict": cmd_dos_predict,
    "dos_get_atom": cmd_dos_get_atom,
    "dos_export": cmd_dos_export,
    "relax_status": cmd_relax_status,
    "relax_download_model": cmd_relax_download_model,
    "relax_run": cmd_relax_run,
    "relax_check_env": cmd_relax_check_env,
    "relax_zip": cmd_relax_zip,
}


def serve():
    """JSON-lines server on stdin/stdout. Library prints are diverted to stderr."""
    global _progress_cb

    proto_fd = os.dup(1)
    os.dup2(2, 1)  # anything that prints to fd 1 now goes to stderr
    proto = io.TextIOWrapper(os.fdopen(proto_fd, "wb"), encoding="utf-8", newline="\n", write_through=True)
    sys.stdout = sys.stderr
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

    def send(obj):
        proto.write(json.dumps(to_jsonable(obj), allow_nan=False, ensure_ascii=True) + "\n")
        proto.flush()

    current = {"id": None}
    _progress_cb = lambda p: send(dict(p, type="progress", id=current["id"]))  # noqa: E731

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            send({"type": "error", "id": None, "error": "Invalid JSON request"})
            continue
        rid = req.get("id")
        cmd = req.get("cmd")
        args = req.get("args") or {}
        current["id"] = rid
        if cmd == "shutdown":
            send({"type": "result", "id": rid, "success": True})
            break
        try:
            handler = HANDLERS.get(cmd)
            if handler is None:
                raise ValueError("Unknown command: %s" % cmd)
            result = handler(args)
            result.setdefault("success", True)
            result["type"] = "result"
            result["id"] = rid
            send(result)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            send({"type": "error", "id": rid, "error": "%s: %s" % (type(e).__name__, e)})


def main():
    parser = argparse.ArgumentParser(description="Padarth Khoj Electrocatalyst Engine")
    parser.add_argument("--serve", action="store_true", help="JSON-lines server on stdin/stdout (default)")
    parser.add_argument("--init-models", action="store_true", help="Train all ML models and print a JSON summary")
    parser.add_argument("--run", action="store_true", help="Run the analysis pipeline once")
    parser.add_argument("--parse-structure", type=str, help="Print the parsed structure (CONTCAR/POSCAR/xyz) as JSON")
    parser.add_argument("--doscar", type=str, help="Path to DOSCAR file")
    parser.add_argument("--active-type", type=str, default="DFT", choices=["DFT", "Machine Learning"])
    parser.add_argument("--method", type=str, default="VASP", choices=["VASP", "Quantum Espresso"])
    parser.add_argument("--reaction", type=str, default="OER", choices=["OER", "ORR"])
    parser.add_argument("--ml-method", type=str, default="SVR", choices=["SVR", "MLR"])
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.init_models:
        print(json.dumps(to_jsonable(cmd_init_models({}))))
        return
    if args.parse_structure:
        print(json.dumps(to_jsonable(parse_structure_file(args.parse_structure))))
        return
    if args.run:
        if not args.doscar:
            print(json.dumps({"success": False, "error": "Missing --doscar parameter."}))
            return
        result = run_analysis_pipeline(args.doscar, args.active_type, args.method, args.reaction, args.ml_method, args.output_dir)
        print(json.dumps(to_jsonable(result)))
        return
    serve()


if __name__ == "__main__":
    main()
