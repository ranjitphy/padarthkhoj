"""Structure relaxation with MACE machine-learning potentials through vasp-mace.

vasp-mace (R. Grau-Crespo, MIT licence, https://github.com/rgraucrespo/vasp-mace) reads VASP-style
POSCAR + INCAR and writes CONTCAR / OSZICAR / OUTCAR / XDATCAR / vasprun.xml.  It is run as a
separate process (relax_worker.py) so a job can be cancelled cleanly.  KPOINTS is copied to the run
folder to keep the VASP-style file set, but MACE does not use k-points.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

_GH = "https://github.com/ACEsuit/mace-foundations/releases/download/"
HEADS_MH1 = [   # read from the real mace-mh-1.model file (heads listed on the model card plus mp_pbe_refit_add)
    ("omat_pbe", "PBE / PBE+U, general materials (model default, balanced)"),
    ("oc20_usemppbe", "PBE, surface catalysis and adsorbates"),
    ("mp_pbe_refit_add", "PBE, Materials-Project-style data (extra head in the file, not described on the model card)"),
    ("matpes_r2scan", "r2SCAN meta-GGA, high-accuracy materials"),
    ("omol", "wB97M-VV10, molecules and organometallics"),
    ("spice_wB97M", "wB97M-D3(BJ), molecules and organic chemistry"),
]
# key -> model description.  "heads" None = single-head model (no head is passed to MACE).
MODELS = {
    "mace-mh-1": {
        "label": "MACE-MH-1 (multi-head, recommended)",
        "file": "mace-mh-1.model",
        "url": _GH + "mace_mh_1/mace-mh-1.model",
        "size_mb": 57,
        "heads": HEADS_MH1,
        "default_head": "omat_pbe",
        "license": "ASL (Academic Software License)",
    },
    "mace-mp-0": {
        "label": "MACE-MP-0 (medium, PBE materials)",
        "file": "2024-01-07-mace-128-L2_epoch-199.model",
        "url": _GH + "mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model",
        "size_mb": 61,
        "heads": None,
        "default_head": None,
        "license": "see the mace-foundations release",
    },
}
DEFAULT_MODEL = "mace-mh-1"
MODEL_NAME = MODELS[DEFAULT_MODEL]["file"]   # kept for older callers
MODEL_URL = MODELS[DEFAULT_MODEL]["url"]
MODEL_MIN_BYTES = 10 * 1024 * 1024
MAX_TEXT = 1024 * 1024

_STEP_RE = re.compile(r"\[step\s+(\d+)\]\s+E=(\S+)\s+eV\s+\|\s+dE=(\S+)\s+eV\s+\|\s+Fmax=(\S+)")


def _have(mod):
    return importlib.util.find_spec(mod) is not None


def _model_path(model_dir, key):
    return os.path.join(model_dir, MODELS[key]["file"]) if model_dir else None


def status(model_dir):
    missing = [m for m in ("vasp_mace", "mace", "torch", "ase") if not _have(m)]
    models = {}
    known_files = set()
    for key, m in MODELS.items():
        p = _model_path(model_dir, key)
        known_files.add(m["file"])
        present = bool(p and os.path.isfile(p) and os.path.getsize(p) >= MODEL_MIN_BYTES)
        models[key] = {
            "key": key, "label": m["label"], "file": m["file"], "url": m["url"], "license": m["license"],
            "expected_mb": m["size_mb"], "present": present, "path": p,
            "size_mb": round(os.path.getsize(p) / 1048576, 1) if present else None,
            "heads": [{"name": n, "info": i} for n, i in m["heads"]] if m["heads"] else None,
            "default_head": m["default_head"],
        }
    others = []
    if model_dir and os.path.isdir(model_dir):
        others = sorted(f for f in os.listdir(model_dir) if f.endswith(".model") and f not in known_files)
    d = models[DEFAULT_MODEL]
    return {
        "success": True,
        "available": not missing,
        "missing": missing,
        "models": models,
        "default_model": DEFAULT_MODEL,
        "other_models": others,
        # fields used by older front-ends: the default model
        "model_name": d["file"], "model_url": d["url"], "model_path": d["path"],
        "model_present": d["present"], "model_size_mb": d["size_mb"],
    }


def download_model(model_dir, progress=None, key=None):
    key = key or DEFAULT_MODEL
    if key not in MODELS:
        return {"success": False, "error": "Unknown model '%s'." % key}
    m = MODELS[key]
    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.join(model_dir, m["file"])
    part = dest + ".part"
    try:
        req = urllib.request.Request(m["url"], headers={"User-Agent": "PadarthKhoj"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            last = 0.0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if progress and now - last > 0.3:
                    last = now
                    pct = int(100 * done / total) if total else None
                    progress("download", "Downloading %s: %%.0f MB%%s" % m["label"].split(" (")[0] % (done / 1048576, (" of %.0f MB" % (total / 1048576)) if total else ""), pct)
        if os.path.getsize(part) < MODEL_MIN_BYTES or not zipfile.is_zipfile(part):
            raise IOError("the downloaded file is not a valid MACE model (%d bytes)" % os.path.getsize(part))
        os.replace(part, dest)
    except Exception as e:  # noqa: BLE001
        try:
            os.remove(part)
        except OSError:
            pass
        return {"success": False, "error": "Model download failed: %s: %s. Check the internet connection, or download %s manually and use 'Use another .model file'." % (type(e).__name__, e, m["url"])}
    return {"success": True, "model_path": dest, "size_mb": round(os.path.getsize(dest) / 1048576, 1)}


def _writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".pk_write_test")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def _incar_tag(text, tag, default=None):
    for line in text.splitlines():
        line = re.split(r"[#!]", line, 1)[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip().upper() == tag:
                    return v.strip()
    return default


_PY_NAME = re.compile(r"^python[0-9.]*(\.exe)?$", re.I)


def valid_python(path):
    """A user-chosen interpreter: must be an existing file called python / python3 / python3.11 / python.exe."""
    return bool(path) and os.path.isfile(path) and bool(_PY_NAME.match(os.path.basename(path)))


_CHECK_SNIPPET = r"""
import sys, time, faulthandler, importlib
faulthandler.dump_traceback_later(30, repeat=True, file=sys.stdout)
t0 = time.time()
print("python %s %s" % (sys.version.split()[0], sys.executable), flush=True)
for name in ("numpy", "torch", "ase", "e3nn", "mace", "vasp_mace"):
    t = time.time()
    print("start %s" % name, flush=True)
    try:
        m = importlib.import_module(name)
        print("ok %s %s %.1f" % (name, getattr(m, "__version__", "?"), time.time() - t), flush=True)
    except Exception as e:
        print("fail %s %s: %s" % (name, type(e).__name__, e), flush=True)
print("done %.1f" % (time.time() - t0), flush=True)
"""


def check_env(python=None, timeout=240, progress=None):
    """Import the packages the relaxor needs, one by one, in a fresh process; report the time of each.
    If something hangs, the process is stopped at `timeout` and the stack traces it printed are returned."""
    py = python or sys.executable
    if python and not valid_python(python):
        return {"success": False, "error": "The Python path must point to a python executable (python.exe / python3)."}
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    t0 = time.time()
    try:
        proc = subprocess.Popen([py, "-u", "-c", _CHECK_SNIPPET], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, **kwargs)
    except OSError as e:
        return {"success": False, "error": "Could not start %s: %s" % (py, e)}
    lines, steps = [], []
    state = {"current": None}

    def pump():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            lines.append(line)
            m = re.match(r"^(start|ok|fail) (\S+)(?: (\S+) ([\d.]+))?", line)
            if m:
                if m.group(1) == "start":
                    state["current"] = m.group(2)
                    if progress:
                        progress("check", "Importing %s ..." % m.group(2), None)
                else:
                    steps.append({"name": m.group(2), "ok": m.group(1) == "ok", "version": m.group(3) if m.group(1) == "ok" else None,
                                  "seconds": float(m.group(4)) if m.group(1) == "ok" and m.group(4) else None,
                                  "error": line.split(" ", 2)[2] if m.group(1) == "fail" and len(line.split(" ", 2)) > 2 else None})
                    if progress:
                        progress("check", line, None)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
    th.join(5)
    total = time.time() - t0
    info = next((l for l in lines if l.startswith("python ")), "")
    bad = [s for s in steps if not s["ok"]]
    return {
        "success": True, "python": py, "info": info[7:], "steps": steps, "total_s": round(total, 1),
        "timed_out": timed_out, "stuck_on": state["current"] if timed_out else None,
        "ok": (not timed_out) and not bad and any(l.startswith("done") for l in lines),
        "log": lines[-60:],
    }


def run(args, parse_structure, progress=None, fallback_dir=None, python=None):
    """args: poscarPath, incarText, kpointsPath, modelPath, head, device, dtype, optimizer, outRoot."""
    poscar = args.get("poscarPath")
    incar_text = args.get("incarText")
    model = args.get("modelPath")
    if not poscar or not os.path.isfile(poscar):
        return {"success": False, "error": "POSCAR not found."}
    if not incar_text or len(incar_text) > MAX_TEXT:
        return {"success": False, "error": "INCAR is empty (or too large)."}
    if not model or not os.path.isfile(model):
        return {"success": False, "error": "MACE model file not found. Download a model (or choose a .model file) first."}
    miss = [m for m in ("vasp_mace", "mace", "torch", "ase") if not _have(m)]
    if miss:
        return {"success": False, "unavailable": True, "error": "Missing Python packages: %s. Run 'npm run setup:python' and restart the app." % ", ".join(miss)}
    if int(float(_incar_tag(incar_text, "IMAGES", "0") or 0)) > 0:
        return {"success": False, "error": "IMAGES > 0 selects NEB, which needs numbered image folders. Remove IMAGES from the INCAR."}
    ibr = _incar_tag(incar_text, "IBRION", "-1")
    if str(ibr).split()[0] in ("0", "5", "6"):
        return {"success": False, "error": "IBRION = %s is not a relaxation (0 = molecular dynamics, 5/6 = phonons). Use IBRION = 2 (or -1)." % ibr}
    nsw = int(float(_incar_tag(incar_text, "NSW", "0") or 0))
    if nsw <= 0:
        return {"success": False, "error": "NSW = 0 is a single-point calculation, not a relaxation. Set NSW to the maximum number of ionic steps (for example 200)."}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = "mace_relax_" + stamp
    out_root = args.get("outRoot") or os.path.dirname(os.path.abspath(poscar))
    run_dir = os.path.join(out_root, run_name)
    if not _writable(out_root):
        run_dir = os.path.join(fallback_dir or os.path.join(os.path.expanduser("~"), "PadarthKhoj"), run_name)
    os.makedirs(run_dir, exist_ok=True)

    shutil.copyfile(poscar, os.path.join(run_dir, "POSCAR"))
    with open(os.path.join(run_dir, "INCAR"), "w", encoding="utf-8", newline="\n") as f:
        f.write(incar_text.rstrip() + "\n")
    kp = args.get("kpointsPath")
    kp_copied = False
    if kp and os.path.isfile(kp):
        shutil.copyfile(kp, os.path.join(run_dir, "KPOINTS"))
        kp_copied = True

    initial = parse_structure(os.path.join(run_dir, "POSCAR"))
    custom_py = (python or args.get("pythonPath") or "").strip() or None
    if custom_py and not valid_python(custom_py):
        return {"success": False, "error": "The Python path must point to a python executable (python.exe / python3): %s" % custom_py}
    cmd = [custom_py or sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "relax_worker.py"),
           "--parent-pid", str(os.getpid()),
           "--model", model, "--device", args.get("device") or "auto", "--dtype", args.get("dtype") or "auto",
           "--optimizer", args.get("optimizer") or "BFGS"]
    head = (args.get("head") or "").strip()
    if head:
        if not re.match(r"^[A-Za-z0-9_.+-]{1,64}$", head):
            return {"success": False, "error": "Invalid head name."}
        cmd += ["--head", head]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    if progress:
        progress("relax", "Starting the relaxation worker (%s)" % (custom_py or "the app's own Python"), 1)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    log, history, info = [], [], {"device": None, "dtype": None}
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            log.append(line)
            m = _STEP_RE.search(line)
            if m:
                history.append({"step": int(m.group(1)), "energy": float(m.group(2)), "dE": float(m.group(3)), "fmax": float(m.group(4))})
            mi = re.search(r"device=(\w+), dtype=(\w+)", line)
            if mi:
                info["device"], info["dtype"] = mi.group(1), mi.group(2)
            if progress:
                pct = min(99, max(2, int(100 * (history[-1]["step"] if history else 0) / max(nsw, 1)))) if history else 3
                progress("relax", line, pct)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    elapsed = time.time() - t0
    contcar = os.path.join(run_dir, "CONTCAR")
    errors = [l for l in log if l.startswith("[error]")]
    if proc.returncode != 0 or not os.path.isfile(contcar):
        msg = errors[-1][7:].strip() if errors else ("vasp-mace stopped (exit code %s)." % proc.returncode)
        return {"success": False, "error": msg, "run_dir": run_dir, "log": log[-80:]}
    final = parse_structure(contcar)
    converged = not any(l.startswith("[note]") and "without meeting" in l for l in log)
    files = []
    for dp, _dn, fn in os.walk(run_dir):
        for name in sorted(fn):
            p = os.path.join(dp, name)
            files.append({"name": os.path.relpath(p, run_dir).replace("\\", "/"), "bytes": os.path.getsize(p)})
    return {
        "success": True,
        "run_dir": run_dir,
        "contcar": contcar,
        "converged": converged,
        "nsw": nsw,
        "steps": len(history),
        "history": history,
        "initial_energy": history[0]["energy"] if history else None,
        "final_energy": history[-1]["energy"] if history else None,
        "final_fmax": history[-1]["fmax"] if history else None,
        "initial": initial,
        "final": final,
        "files": files,
        "kpoints_copied": kp_copied,
        "device": info["device"],
        "dtype": info["dtype"],
        "elapsed_s": round(elapsed, 1),
        "log": log[-200:],
    }


def zip_run(run_dir, out_path):
    if not run_dir or not os.path.isdir(run_dir):
        return {"success": False, "error": "Run folder not found."}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _dn, fn in os.walk(run_dir):
            for name in fn:
                p = os.path.join(dp, name)
                z.write(p, os.path.join(os.path.basename(run_dir), os.path.relpath(p, run_dir)))
    return {"success": True, "path": out_path, "bytes": os.path.getsize(out_path)}
