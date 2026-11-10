"""Glue between the web page and the desktop engine, running inside Pyodide.
The page calls call(name, json_args) with the same names the desktop / server API uses; files live in an in-memory
folder (/sess) that the page fills with the user's uploads. Nothing leaves the browser."""
import json
import os
import re
import sys
import traceback

sys.path.insert(0, "/pk/py")
import engine  # noqa: E402

BASE, SESS = "/pk", "/sess"
engine.resource_path = lambda rel: os.path.join(BASE, rel)
os.makedirs(SESS, exist_ok=True)


def _post(msg):
    try:
        import js
        from pyodide.ffi import to_js
        js.postMessage(to_js({"type": "progress", **msg}, dict_converter=js.Object.fromEntries))
    except Exception:  # noqa: BLE001
        pass


engine._progress_cb = _post


def real(p):
    if not isinstance(p, str) or not p.startswith("~/") or "\x00" in p:
        raise ValueError("Invalid file reference.")
    r = os.path.normpath(os.path.join(SESS, p[2:]))
    if r != SESS and not r.startswith(SESS + os.sep):
        raise ValueError("Invalid file reference.")
    return r


def fix(x):
    if isinstance(x, str):
        return x.replace(SESS + "/", "~/").replace(BASE + "/", "engine/")
    if isinstance(x, list):
        return [fix(i) for i in x]
    if isinstance(x, dict):
        return {k: fix(v) for k, v in x.items()}
    return x


NO_RELAX = {"success": False, "unavailable": True,
            "error": "The Structure Relaxor needs PyTorch and MACE, which cannot run in a web page. Use the desktop app or a self-hosted server."}


def _handle(name, a):
    if name == "initModels":
        return engine.cmd_init_models({})
    if name == "getEngineStatus":
        return dict(engine.cmd_ping({}), launcher="in your browser (Pyodide)")
    if name == "parseVaspHeader":
        return engine.cmd_parse_vasp_header({"doscarPath": real(a)})
    if name == "parseStructure":
        args = {"path": a} if isinstance(a, str) else dict(a or {})
        for k in ("path", "structurePath", "doscarPath"):
            if args.get(k) is not None:
                args[k] = real(args[k])
        return engine.cmd_parse_structure(args)
    if name == "runAnalysis":
        o = dict(a or {})
        o["doscarPath"] = real(o.get("doscarPath"))
        os.makedirs(SESS + "/work", exist_ok=True)
        o["fallbackDir"] = SESS + "/work"
        return engine.cmd_run_analysis(o)
    if name == "dosStatus":
        return engine.cmd_dos_status({})
    if name == "dosLoadStructure":
        return engine.cmd_dos_load_structure({"path": real(a)})
    if name == "dosPredict":
        return engine.cmd_dos_predict({"path": real(a)})
    if name == "dosGetAtom":
        return engine.cmd_dos_get_atom({"index": a})
    if name == "dosExport":
        o = dict(a or {})
        fmt = "jpg" if o.get("format") == "jpg" else "png"
        base = re.sub(r"[^\w.-]+", "_", re.sub(r"\.[^.]*$", "", str(o.get("baseName") or "structure"))) or "structure"
        os.makedirs(SESS + "/outputs", exist_ok=True)
        out = "%s/outputs/%s_PDOS_%s.zip" % (SESS, base, fmt.upper())
        o.update(format=fmt, outPath=out)
        r = engine.cmd_dos_export(o)
        return dict(r, savedPath=out, downloadPath=out)
    if name in ("relaxStatus", "relaxDownloadModel", "relaxRun", "relaxExportZip"):
        return dict(NO_RELAX)
    if name == "cancelJob":
        return {"success": True}
    raise KeyError("unknown call: %s" % name)


def call(name, arg_json):
    try:
        a = json.loads(arg_json) if arg_json else None
        r = _handle(name, a)
        return json.dumps(fix(engine.to_jsonable(r)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return json.dumps({"success": False, "error": "%s: %s" % (type(e).__name__, e)})
