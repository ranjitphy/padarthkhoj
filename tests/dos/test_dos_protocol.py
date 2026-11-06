"""Drives engine.py (JSON lines) like main.js: dos_status, dos_predict, dos_get_atom, dos_export (png + jpg)."""
import json, os, subprocess, sys, tempfile, zipfile, io
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = os.environ.get("PADARTH_PYTHON", sys.executable)
tmp = tempfile.mkdtemp()
subprocess.check_call([sys.executable, os.path.join(HERE, "make_structures.py"), tmp], stdout=subprocess.DEVNULL)
p = subprocess.Popen([PY, os.path.join(ROOT, "engine", "engine.py"), "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True)
n = [0]; fails = []
def call(cmd, **args):
    n[0] += 1; rid = n[0]
    p.stdin.write(json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n"); p.stdin.flush()
    prog = 0
    while True:
        m = json.loads(p.stdout.readline())
        if m["id"] != rid: continue
        if m["type"] == "progress": prog += 1; continue
        m["_progress"] = prog; return m
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (" - " + str(extra) if extra else ""))
    if not cond: fails.append(name)

st = call("dos_status"); check("dos_status", st.get("available") and st.get("models_present"), st.get("missing"))
for fn in ["CONTCAR", "ndoped.cif", "ndoped.xyz", "ndoped.cc1"]:
    r = call("dos_load_structure", path=os.path.join(tmp, fn))
    check("load %s" % fn, r.get("success") and r["natoms"] == 48, r.get("error") or r.get("formula"))
r = call("dos_load_structure", path=os.path.join(tmp, "nope.cif")); check("missing file -> error", r["type"] == "result" and not r["success"], r.get("error"))
open(os.path.join(tmp, "bad.cif"), "w").write("garbage"); r = call("dos_load_structure", path=os.path.join(tmp, "bad.cif")); check("garbage file -> error", not r["success"], r.get("error", "")[:80])
r = call("dos_get_atom", index=1); check("get_atom before predict -> error", not r["success"])
r = call("dos_export", outPath=os.path.join(tmp, "x.zip")); check("export before predict -> error", not r["success"])

r = call("dos_predict", path=os.path.join(tmp, "CONTCAR"))
check("predict", r["success"] and len(r["atoms"]) == 48 and len(r["curves"]) == 8 and len(r["curves"][0]) == 3000 and len(r["energy"]) == 3000,
      "%.1fs, %d progress msgs" % (r.get("elapsed_s", 0), r["_progress"]))
check("energy grid -25..10", abs(r["energy"][0] + 25) < 1e-3 and abs(r["energy"][-1] - 10) < 1e-3)
check("first atom is atom 1 with descriptors", r["atoms"][0]["index"] == 1 and "pz_fermi" in r["atoms"][0] and "pz_occ" in r["atoms"][0])
c0 = r["curves"][4]
r2 = call("dos_get_atom", index=7); check("get atom 7", r2["success"] and r2["atom_index"] == 7 and r2["curves"][4] != c0)
r3 = call("dos_get_atom", index=999); check("atom out of range -> error", not r3["success"])
for fmt in ("png", "jpg"):
    out = os.path.join(tmp, "plots_%s.zip" % fmt)
    e = call("dos_export", outPath=out, format=fmt, dpi=80, emin=-10, emax=5)
    ok = e["success"] and e["count"] == 48
    names = []
    if ok:
        z = zipfile.ZipFile(out); names = z.namelist()
        from PIL import Image
        im = Image.open(io.BytesIO(z.read(names[0]))); ok = len(names) == 48 and im.format.lower() in ("png", "jpeg") and im.size[0] > 300
    check("export %s zip (48 images)" % fmt, ok, "%s ... %s, %d KB" % (names[:1], names[-1:], e.get("bytes", 0) // 1024))
p.stdin.write(json.dumps({"id": 99, "cmd": "shutdown"}) + "\n"); p.stdin.flush()
print("\nALL PASS" if not fails else "\nFAILED: %s" % fails); sys.exit(1 if fails else 0)
