#!/usr/bin/env python3
"""Drives engine/engine.py exactly the way Electron's main.js does (JSON lines over stdin/stdout)."""
import json
import os
import subprocess
import sys
import tempfile
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "parity"))
from make_synthetic_vasp import generate  # noqa: E402


class Engine:
    def __init__(self):
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        self.p = subprocess.Popen([sys.executable, os.path.join(ROOT, "engine", "engine.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env)
        self.n = 0
        self.progress = 0

    def call(self, cmd, **args):
        self.n += 1
        rid = "t%d" % self.n
        self.p.stdin.write(json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            assert line, "engine closed stdout"
            msg = json.loads(line)  # must be strict JSON (no NaN / Infinity)
            if msg.get("type") == "progress":
                self.progress += 1
                continue
            assert msg["id"] == rid, msg
            return msg

    def close(self):
        self.call("shutdown")
        self.p.wait(timeout=10)


def check(cond, label):
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    tmp = tempfile.mkdtemp(prefix="padarth_proto_")
    try:
        generate(tmp, ispin=2)
        e = Engine()
        r = e.call("ping")
        check(r["type"] == "result" and r["success"] and r["status"] == "ok", "ping -> ok (python %s, sklearn %s)" % (r["python"], r["sklearn"]))
        r = e.call("init_models")
        check(r["success"] and len(r["models"]) == 4, "init_models -> 4 models, R2: " + ", ".join("%s %.3f" % (k, v["r2_train"]) for k, v in sorted(r["training"].items())))
        r = e.call("parse_vasp_header", doscarPath=os.path.join(tmp, "DOSCAR"))
        check(r["success"] and r["natoms"] == 73 and r["nspin"] == 2 and r["has_outcar"], "parse_vasp_header -> %d atoms, nedos %d, E_F %.4f (%s), ISPIN %d" % (r["natoms"], r["nedos"], r["efermi"], r["efermi_source"], r["nspin"]))
        r = e.call("parse_structure", doscarPath=os.path.join(tmp, "DOSCAR"))
        check(r["success"] and r["natoms"] == 73 and len(r["cart"]) == 73 and r["formula"] == "C66 N4 Fe O H", "parse_structure -> %s" % r["formula"])
        for reaction in ("OER", "ORR"):
            for ml in ("SVR", "MLR"):
                r = e.call("run_analysis", doscarPath=os.path.join(tmp, "DOSCAR"), activeType="Machine Learning", method="VASP", reaction=reaction, mlMethod=ml)
                ok = r["success"] and os.path.exists(r["excel_path"]) and os.path.exists(r["csv_path"]) and r["records"]
                check(ok, "run_analysis %s/%s -> %d records, best atom %s eta=%.3f V (progress msgs so far: %d)" % (reaction, ml, len(r["records"]), r["summary"]["best_atom"], r["summary"]["lowest_overpotential"], e.progress))
        r = e.call("run_analysis", doscarPath=os.path.join(tmp, "DOSCAR"), activeType="DFT", method="Quantum Espresso", reaction="OER", mlMethod="SVR")
        check(r["success"] is False and "Quantum Espresso" in r["error"], "Quantum Espresso -> not implemented message")
        r = e.call("run_analysis", doscarPath=os.path.join(tmp, "nope", "DOSCAR"), reaction="OER")
        check(r["success"] is False, "missing DOSCAR -> success false")
        os.remove(os.path.join(tmp, "CONTCAR"))
        r = e.call("run_analysis", doscarPath=os.path.join(tmp, "DOSCAR"), reaction="OER", mlMethod="SVR")
        check(r["success"] is False and "CONTCAR" in r["error"], "OER without CONTCAR -> clear error")
        r = e.call("run_analysis", doscarPath=os.path.join(tmp, "DOSCAR"), reaction="ORR", mlMethod="SVR")
        check(r["success"], "ORR without CONTCAR still works (%d records, no Element tags: %s)" % (len(r["records"]), "Element" not in r["records"][0]))
        r = e.call("bogus")
        check(r["type"] == "error", "unknown command -> type=error")
        e.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%s" % ("ALL PASS" if not check.failed else "%d FAILED" % check.failed))
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
