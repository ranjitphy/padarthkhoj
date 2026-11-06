"""Drives engine.py like main.js does: relax_status, validation errors, a real vasp-mace relaxation, zip.
Needs a Python with vasp-mace.  PADARTH_TEST_MODEL = path to a MACE .model file (the real run is skipped without it)."""
import shutil, json, os, subprocess, sys, tempfile, zipfile, time, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = os.environ.get("PADARTH_PYTHON", sys.executable)
MODEL = os.environ.get("PADARTH_TEST_MODEL")
tmp = tempfile.mkdtemp()
subprocess.check_call([sys.executable, os.path.join(ROOT, "tests", "dos", "make_structures.py"), os.path.join(tmp, "s")], stdout=subprocess.DEVNULL)
poscar = os.path.join(tmp, "s", "CONTCAR")
kp = os.path.join(tmp, "KPOINTS"); open(kp, "w").write("Automatic\n0\nGamma\n3 3 1\n0 0 0\n")
mdir = os.path.join(tmp, "models"); os.makedirs(mdir)
p = subprocess.Popen([PY, os.path.join(ROOT, "engine", "engine.py"), "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
n = [0]; fails = []
def call(cmd, **args):
    n[0] += 1; rid = n[0]
    p.stdin.write(json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n"); p.stdin.flush()
    prog = []
    while True:
        m = json.loads(p.stdout.readline())
        if m["id"] != rid: continue
        if m["type"] == "progress": prog.append(m); continue
        m["_progress"] = prog; return m
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (" - " + str(extra) if extra else ""))
    if not cond: fails.append(name)

st = call("relax_status", modelDir=mdir)
check("relax_status: packages available, no model downloaded yet", st["available"] and not st["model_present"] and set(st["models"]) >= {"mace-mh-1", "mace-mp-0"}, st.get("missing"))
check("MACE-MH-1 is the default model, with 6 heads and omat_pbe default", st["default_model"] == "mace-mh-1" and len(st["models"]["mace-mh-1"]["heads"]) == 6 and st["models"]["mace-mh-1"]["default_head"] == "omat_pbe" and st["models"]["mace-mp-0"]["heads"] is None)
INC_OK = "NSW = 40\nIBRION = 2\nISIF = 2\nEDIFFG = -0.05\n"
base = dict(poscarPath=poscar, kpointsPath=kp, outRoot=os.path.join(tmp, "out"), device="cpu")
r = call("relax_run", incarText=INC_OK, modelPath=os.path.join(mdir, "nope.model"), **base); check("missing model -> clear error", not r["success"] and "model" in r["error"].lower(), r["error"][:70])
open(os.path.join(mdir, "fake.model"), "wb").write(b"x" * 100)
for name, inc, key in [("NSW=0", "NSW = 0\nIBRION = 2\n", "NSW"), ("MD", "NSW = 10\nIBRION = 0\n", "IBRION"), ("phonons", "NSW = 10\nIBRION = 6\n", "IBRION"), ("NEB", "NSW = 10\nIBRION = 2\nIMAGES = 3\n", "IMAGES"), ("empty INCAR", "", "empty")]:
    r = call("relax_run", incarText=inc, modelPath=os.path.join(mdir, "fake.model"), **base)
    check("INCAR check: %s refused" % name, (not r["success"]) and key.lower() in r["error"].lower(), r["error"][:80])
r = call("relax_run", incarText=INC_OK, modelPath=os.path.join(mdir, "fake.model"), poscarPath=os.path.join(tmp, "none"), outRoot=tmp); check("missing POSCAR -> error", not r["success"])
r = call("relax_run", incarText=INC_OK, modelPath=os.path.join(mdir, "fake.model"), **base)
check("invalid model file -> error (not a crash)", not r["success"] and r["type"] == "result", r.get("error", "")[:90])

if MODEL and os.path.isfile(MODEL):
    t = time.time()
    r = call("relax_run", incarText=INC_OK, modelPath=MODEL, **base)
    check("real relaxation succeeded", r["success"], r.get("error"))
    if r["success"]:
        check("steps / history parsed", r["steps"] > 5 and len(r["history"]) == r["steps"], "%d steps, %.1fs" % (r["steps"], time.time() - t))
        check("energy decreased", r["final_energy"] < r["initial_energy"], "%.3f -> %.3f eV" % (r["initial_energy"], r["final_energy"]))
        check("initial / final structures returned", r["initial"]["natoms"] == 48 and r["final"]["natoms"] == 48 and r["final"]["cart"] != r["initial"]["cart"])
        names = {f["name"] for f in r["files"]}
        check("VASP-style outputs written", {"CONTCAR", "OSZICAR", "OUTCAR", "XDATCAR", "vasprun.xml", "POSCAR", "INCAR", "KPOINTS"} <= names, sorted(names))
        check("run folder created next to POSCAR's outRoot", r["run_dir"].startswith(os.path.join(tmp, "out")))
        check("progress events streamed", len(r["_progress"]) > 5, "%d events" % len(r["_progress"]))
        z = call("relax_zip", runDir=r["run_dir"], outPath=os.path.join(tmp, "out.zip"))
        check("zip of outputs", z["success"] and "CONTCAR" in " ".join(zipfile.ZipFile(z["outPath"] if "outPath" in z else z["path"]).namelist()))
    # a tight criterion and a very small NSW -> not converged, still a result
    r2 = call("relax_run", incarText="NSW = 3\nIBRION = 2\nEDIFFG = -0.0001\n", modelPath=MODEL, **base)
    check("NSW reached -> converged = False, still succeeds", r2["success"] and r2["converged"] is False)
    # cancel safety: kill the engine mid-run and make sure the worker dies with it
    n[0] += 1
    MH1 = os.environ.get("PADARTH_TEST_MH1")
    if MH1 and os.path.isfile(MH1):
        INC_S = "NSW = 3\nIBRION = 2\nEDIFFG = -0.0001\n"
        shutil.copy(MH1, os.path.join(mdir, "mace-mh-1.model"))
        st2 = call("relax_status", modelDir=mdir)
        check("status sees the MH-1 file", st2["models"]["mace-mh-1"]["present"] and st2["model_present"], st2["models"]["mace-mh-1"]["size_mb"])
        r = call("relax_run", incarText=INC_S, modelPath=MH1, head="omat_pbe", **base)
        check("MH-1 relaxation with head omat_pbe", r["success"] and r["final_energy"] < r["initial_energy"], r.get("error") or (r["initial_energy"], r["final_energy"]))
        check("log reports the head", any("Head=omat_pbe" in l for l in r.get("log", [])))
        r = call("relax_run", incarText=INC_S, modelPath=MH1, head="oc20_usemppbe", dtype="float32", **base)
        check("MH-1 head oc20_usemppbe, float32", r["success"] and r["dtype"] == "float32", r.get("error"))
        r = call("relax_run", incarText=INC_S, modelPath=MH1, head="not_a_head", **base)
        check("unknown head -> error listing the real heads", not r["success"] and "omat_pbe" in r["error"] and "not in this model" in r["error"], r["error"][:90])
        r = call("relax_run", incarText=INC_S, modelPath=MH1, **base)
        check("multi-head model without head -> clear error", not r["success"] and "head" in r["error"].lower(), r["error"][:90])
        r = call("relax_run", incarText=INC_S, modelPath=MH1, head="bad head;rm", **base)
        check("invalid head name refused", not r["success"] and "Invalid head" in r["error"], r["error"])
    else:
        print("SKIP  MH-1 checks (set PADARTH_TEST_MH1)")
    ce = call("relax_check_env", timeoutS=120)
    check("check environment: all packages import, timings listed", ce["success"] and ce["ok"] and {x["name"] for x in ce["steps"]} >= {"torch", "mace", "vasp_mace"}, ce.get("total_s"))
    ce = call("relax_check_env", pythonPath="/etc/passwd")
    check("check environment: refuses a non-python path", not ce["success"])
    r = call("relax_run", incarText="NSW = 3\nIBRION = 2\nEDIFFG = -0.0001\n", modelPath=MODEL, pythonPath="/bin/ls", **base)
    check("relax_run refuses a non-python executable", not r["success"] and "python" in r["error"].lower(), r["error"][:60])
    r = call("relax_run", incarText="NSW = 3\nIBRION = 2\nEDIFFG = -0.0001\n", modelPath=MODEL, pythonPath=sys.executable, **base)
    check("relax_run with an explicit Python (the user's own environment)", r["success"] and r["steps"] >= 2, r.get("error"))
    p.stdin.write(json.dumps({"id": n[0], "cmd": "relax_run", "args": dict(incarText="NSW = 500\nIBRION = 2\nEDIFFG = -0.00001\n", modelPath=MODEL, **base)}) + "\n"); p.stdin.flush()
    time.sleep(12)
    kids = subprocess.run(["pgrep", "-P", str(p.pid), "-f", "relax_worker.py"], capture_output=True, text=True).stdout.split()
    check("worker is a child of the engine while the job runs", len(kids) == 1, kids)
    p.kill()
    gone = False
    for _ in range(40):
        time.sleep(0.5)
        if all(subprocess.run(["ps", "-p", k], capture_output=True).returncode != 0 for k in kids):
            gone = True; break
    check("worker exits by itself when the engine is killed (cancel)", gone, kids)
else:
    print("SKIP  real relaxation (set PADARTH_TEST_MODEL)")
try: p.stdin.write(json.dumps({"id": 999, "cmd": "shutdown"}) + "\n"); p.stdin.flush()
except Exception: pass
print("\nALL PASS" if not fails else "\nFAILED: %s" % fails); sys.exit(1 if fails else 0)
