"""Padarth Khoj web server.

Serves the browser interface (../web) and exposes the Python engine over HTTP.
Every browser session gets its own engine process and its own scratch folder, so users never see
each other's files. Uploads and results live in that folder and are deleted after the session goes idle.

Run:  python server/app.py            (then open http://127.0.0.1:8000)
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "engine"
WEB_DIR = ROOT / "web"
VERSION = "1.5.0-web"


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


DATA_DIR = Path(os.environ.get("PK_DATA_DIR", ROOT / ".padarth-data")).resolve()
SESS_DIR = DATA_DIR / "sessions"
MODEL_DIR = DATA_DIR / "mace-models"
MAX_SESSIONS = _int("PK_MAX_SESSIONS", 6)          # engine processes alive at once
MAX_JOBS = _int("PK_MAX_JOBS", 2)                  # heavy jobs running at once (analysis, PDOS, relaxation)
IDLE_TTL = _int("PK_IDLE_TTL", 1800)               # seconds without activity before a session is removed
JOB_TIMEOUT = _int("PK_JOB_TIMEOUT", 1800)         # seconds a single heavy job may run
MAX_UPLOAD = _int("PK_MAX_UPLOAD_MB", 500) * 1024 * 1024
MAX_NSW = _int("PK_MAX_NSW", 500)                  # cap on INCAR NSW (ionic steps)
ALLOW_CUSTOM_MODEL = os.environ.get("PK_ALLOW_CUSTOM_MODEL", "0") == "1"  # .model files are pickles: off by default
PYTHON = os.environ.get("PK_PYTHON", sys.executable)

HEAVY = {"run_analysis", "dos_predict", "dos_export", "relax_run", "relax_download_model"}
SID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- sessions
class Session:
    def __init__(self, sid):
        self.sid = sid
        self.dir = (SESS_DIR / sid).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.proc = None
        self.reader = None
        self.pending = {}
        self.listeners = set()
        self.last = time.time()
        self.next_id = 0
        self.start_lock = asyncio.Lock()
        self.cancelled = False
        self.stderr_tail = []

    def touch(self):
        self.last = time.time()

    @property
    def busy(self):
        return bool(self.pending)

    def alive(self):
        return self.proc is not None and self.proc.returncode is None

    def emit(self, msg):
        for q in list(self.listeners):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def ensure(self):
        async with self.start_lock:
            if self.alive():
                return
            await make_room(self)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            self.stderr_tail = []
            self.cancelled = False
            self.proc = await asyncio.create_subprocess_exec(
                PYTHON, "-u", str(ENGINE_DIR / "engine.py"), "--serve",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(ENGINE_DIR), env=env, limit=1 << 29,
            )
            self.reader = asyncio.create_task(self._read(self.proc))
            asyncio.create_task(self._read_err(self.proc))

    async def _read_err(self, proc):
        try:
            async for raw in proc.stderr:
                self.stderr_tail.append(raw.decode("utf-8", "replace").rstrip())
                del self.stderr_tail[:-30]
        except Exception:
            pass

    async def _read(self, proc):
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("type") == "progress":
                    self.emit(rewrite_out(self, msg))
                    continue
                fut = self.pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(msg)
        except Exception:
            pass
        finally:
            await proc.wait()
            for fut in list(self.pending.values()):
                if not fut.done():
                    if self.cancelled:
                        fut.set_exception(ApiError(409, "CANCELLED"))
                    else:
                        tail = "\n".join(self.stderr_tail[-5:])
                        fut.set_exception(ApiError(500, "Engine process stopped unexpectedly." + ("\n" + tail if tail else "")))
            self.pending.clear()

    async def command(self, cmd, args, timeout=None):
        await self.ensure()
        self.touch()
        self.next_id += 1
        rid = self.next_id
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        try:
            self.proc.stdin.write((json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n").encode())
            await self.proc.stdin.drain()
        except Exception as e:  # engine died between ensure() and write
            self.pending.pop(rid, None)
            raise ApiError(500, "Engine is not running: %s" % e)
        try:
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self.cancelled = True
            self.kill()
            raise ApiError(504, "Time limit exceeded (%d s). The job was stopped." % (timeout or 0))
        finally:
            self.touch()
        if msg.get("type") == "error":
            raise ApiError(500, msg.get("error") or "Engine error")
        return msg

    def kill(self):
        p = self.proc
        if p is not None and p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass

    def cancel(self):
        self.cancelled = True
        self.kill()

    async def close(self):
        self.kill()
        if self.proc is not None:
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except Exception:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)


SESSIONS = {}
JOB_SEM = None
MODEL_LOCK = None


def get_session(sid):
    if not sid or not SID_RE.match(sid):
        raise ApiError(400, "Missing or invalid session id.")
    s = SESSIONS.get(sid)
    if s is None:
        s = SESSIONS[sid] = Session(sid)
    s.touch()
    return s


async def make_room(current):
    live = [s for s in SESSIONS.values() if s.alive() and s is not current]
    if len(live) < MAX_SESSIONS:
        return
    idle = sorted((s for s in live if not s.busy and not s.listeners), key=lambda s: s.last)
    idle = idle or sorted((s for s in live if not s.busy), key=lambda s: s.last)
    if not idle:
        raise ApiError(503, "The server is busy (all %d slots are running jobs). Try again in a few minutes." % MAX_SESSIONS)
    victim = idle[0]
    SESSIONS.pop(victim.sid, None)
    await victim.close()


async def reaper():
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for sid, s in list(SESSIONS.items()):
            if now - s.last > IDLE_TTL:
                SESSIONS.pop(sid, None)
                await s.close()


# ---------------------------------------------------------------- path mapping (client sees "~/..." only)
def to_real(s, p):
    if not isinstance(p, str) or not p.startswith("~/") or "\x00" in p:
        raise ApiError(400, "Invalid file reference.")
    real = (s.dir / p[2:]).resolve()
    if real != s.dir and s.dir not in real.parents:
        raise ApiError(400, "Invalid file reference.")
    return str(real)


def model_real(s, p):
    if isinstance(p, str) and p.startswith("@models/"):
        name = p[len("@models/"):]
        if "/" in name or "\\" in name or not name.endswith(".model"):
            raise ApiError(400, "Invalid model reference.")
        return str(MODEL_DIR / name)
    if ALLOW_CUSTOM_MODEL:
        return to_real(s, p)
    raise ApiError(403, "Custom model files are disabled on this server (they are pickle files and can run code).")


def rewrite_out(s, obj):
    sp, mp, ep = str(s.dir) + os.sep, str(MODEL_DIR) + os.sep, str(ENGINE_DIR) + os.sep

    def fix(x):
        if isinstance(x, str):
            if sp in x:
                x = x.replace(sp, "~/")
                if os.sep == "\\":
                    x = re.sub(r"~/[^\s\"']*", lambda m: m.group(0).replace("\\", "/"), x)
            if mp in x:
                x = x.replace(mp, "@models/")
            if ep in x:
                x = x.replace(ep, "engine/")
            return x
        if isinstance(x, list):
            return [fix(i) for i in x]
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        return x

    return fix(obj)


def safe_name(name):
    name = os.path.basename(str(name or "").replace("\\", "/"))
    name = re.sub(r"[^\w.\-+() ]", "_", name).strip(" .")[:120]
    if not name:
        raise ApiError(400, "Invalid file name.")
    return name


# ---------------------------------------------------------------- API handlers
async def run_cmd(s, cmd, args):
    if cmd in HEAVY:
        async with JOB_SEM:
            return await s.command(cmd, args, timeout=JOB_TIMEOUT)
    return await s.command(cmd, args, timeout=300)


def out_dir(s, sub):
    d = s.dir / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


async def h_initModels(s, a):
    return await run_cmd(s, "init_models", {})


async def h_getEngineStatus(s, a):
    r = await run_cmd(s, "ping", {})
    return dict(r, launcher="web server")


async def h_parseVaspHeader(s, a):
    return await run_cmd(s, "parse_vasp_header", {"doscarPath": to_real(s, a)})


async def h_parseStructure(s, a):
    args = {"path": a} if isinstance(a, str) else dict(a or {})
    for k in ("path", "structurePath", "doscarPath"):
        if args.get(k) is not None:
            args[k] = to_real(s, args[k])
    return await run_cmd(s, "parse_structure", args)


async def h_runAnalysis(s, a):
    o = dict(a or {})
    o["doscarPath"] = to_real(s, o.get("doscarPath"))
    o["fallbackDir"] = str(out_dir(s, "work"))
    return await run_cmd(s, "run_analysis", o)


async def h_dosStatus(s, a):
    return await run_cmd(s, "dos_status", {})


async def h_dosLoadStructure(s, a):
    return await run_cmd(s, "dos_load_structure", {"path": to_real(s, a)})


async def h_dosPredict(s, a):
    return await run_cmd(s, "dos_predict", {"path": to_real(s, a)})


async def h_dosGetAtom(s, a):
    return await run_cmd(s, "dos_get_atom", {"index": a})


async def h_dosExport(s, a):
    o = dict(a or {})
    fmt = "jpg" if o.get("format") == "jpg" else "png"
    base = re.sub(r"[^\w.-]+", "_", re.sub(r"\.[^.]*$", "", str(o.get("baseName") or "structure"))) or "structure"
    out = out_dir(s, "outputs") / ("%s_PDOS_%s.zip" % (base, fmt.upper()))
    o.update(format=fmt, outPath=str(out))
    r = await run_cmd(s, "dos_export", o)
    return dict(r, savedPath=str(out), downloadPath=str(out))


async def h_relaxStatus(s, a):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    r = await run_cmd(s, "relax_status", {"modelDir": str(MODEL_DIR)})
    r["web"] = {"allow_custom_model": ALLOW_CUSTOM_MODEL, "max_nsw": MAX_NSW}
    return r


async def h_relaxDownloadModel(s, a):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    async with MODEL_LOCK:  # one download at a time; later callers find the file already there
        key = (a or {}).get("model") if isinstance(a, dict) else None
        if key is not None and not re.match(r"^[a-z0-9-]{1,32}$", str(key)):
            raise ApiError(400, "Invalid model name.")
        return await run_cmd(s, "relax_download_model", {"modelDir": str(MODEL_DIR), "model": key})


async def h_relaxRun(s, a):
    o = dict(a or {})
    m = re.search(r"^\s*NSW\s*=\s*(\d+)", str(o.get("incarText") or ""), re.I | re.M)
    if m and int(m.group(1)) > MAX_NSW:
        raise ApiError(400, "NSW = %s is above the limit of %d ionic steps set on this server." % (m.group(1), MAX_NSW))
    o.pop("pythonPath", None)  # never let a visitor choose which executable runs
    o["poscarPath"] = to_real(s, o.get("poscarPath"))
    if o.get("kpointsPath"):
        o["kpointsPath"] = to_real(s, o["kpointsPath"])
    o["modelPath"] = model_real(s, o.get("modelPath"))
    o["outRoot"] = str(out_dir(s, "relax_runs"))
    o["fallbackDir"] = str(out_dir(s, "relax_runs"))
    return await run_cmd(s, "relax_run", o)


async def h_relaxExportZip(s, a):
    run_dir = to_real(s, (a or {}).get("runDir"))
    out = out_dir(s, "outputs") / ("%s.zip" % safe_name(os.path.basename(run_dir) or "mace_relax"))
    r = await run_cmd(s, "relax_zip", {"runDir": run_dir, "outPath": str(out)})
    return dict(r, savedPath=str(out), downloadPath=str(out))


async def h_cancelJob(s, a):
    s.cancel()
    return {"success": True, "message": "Current job terminated."}


HANDLERS = {
    "initModels": h_initModels, "getEngineStatus": h_getEngineStatus, "parseVaspHeader": h_parseVaspHeader,
    "parseStructure": h_parseStructure, "runAnalysis": h_runAnalysis, "dosStatus": h_dosStatus,
    "dosLoadStructure": h_dosLoadStructure, "dosPredict": h_dosPredict, "dosGetAtom": h_dosGetAtom,
    "dosExport": h_dosExport, "relaxStatus": h_relaxStatus, "relaxDownloadModel": h_relaxDownloadModel,
    "relaxRun": h_relaxRun, "relaxExportZip": h_relaxExportZip, "cancelJob": h_cancelJob,
}


# ---------------------------------------------------------------- app
app = FastAPI(title="Padarth Khoj", docs_url=None, redoc_url=None, openapi_url=None)


@app.on_event("startup")
async def _startup():
    global JOB_SEM, MODEL_LOCK
    JOB_SEM = asyncio.Semaphore(MAX_JOBS)
    MODEL_LOCK = asyncio.Lock()
    shutil.rmtree(SESS_DIR, ignore_errors=True)
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(reaper())


@app.on_event("shutdown")
async def _shutdown():
    for s in list(SESSIONS.values()):
        await s.close()


@app.exception_handler(ApiError)
async def _api_error(request, exc):
    return JSONResponse({"error": exc.message}, status_code=exc.status)


@app.middleware("http")
async def _headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/info")
async def info():
    return {"version": VERSION, "allow_custom_model": ALLOW_CUSTOM_MODEL, "max_upload_mb": MAX_UPLOAD // 1048576,
            "max_nsw": MAX_NSW, "max_sessions": MAX_SESSIONS, "idle_minutes": IDLE_TTL // 60}


@app.post("/api/call/{name}")
async def call(name: str, request: Request):
    h = HANDLERS.get(name)
    if h is None:
        raise ApiError(404, "Unknown call: %s" % name)
    s = get_session(request.headers.get("x-session"))
    try:
        body = await request.json()
    except Exception:
        body = {}
    result = await h(s, (body or {}).get("arg"))
    return JSONResponse(rewrite_out(s, result))


@app.post("/api/upload")
async def upload(request: Request):
    s = get_session(request.headers.get("x-session"))
    group = request.query_params.get("group") or uuid.uuid4().hex[:10]
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", group):
        raise ApiError(400, "Invalid upload group.")
    name = safe_name(request.query_params.get("name"))
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD:
        raise ApiError(413, "File is larger than the %d MB limit." % (MAX_UPLOAD // 1048576))
    d = s.dir / "inputs" / group
    d.mkdir(parents=True, exist_ok=True)
    target = d / name
    size = 0
    try:
        with open(target, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise ApiError(413, "File is larger than the %d MB limit." % (MAX_UPLOAD // 1048576))
                f.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {"path": "~/inputs/%s/%s" % (group, name), "name": name, "directory": "uploaded file", "sizeBytes": size, "group": group}


@app.get("/api/download")
async def download(request: Request, path: str):
    s = get_session(request.headers.get("x-session"))
    real = Path(to_real(s, path))
    if not real.is_file():
        raise ApiError(404, "File not found.")
    return FileResponse(real, filename=real.name, media_type="application/octet-stream")


@app.get("/api/events")
async def events(request: Request, sid: str):
    s = get_session(sid)
    q = asyncio.Queue(maxsize=500)
    s.listeners.add(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    msg = await asyncio.wait_for(q.get(), 15)
                    yield "data: %s\n\n" % json.dumps(msg)
                except asyncio.TimeoutError:
                    s.touch()
                    yield ": keep-alive\n\n"
        finally:
            s.listeners.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# ---------------------------------------------------------------- entry point
def prefetch_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    code = ("import sys; sys.path.insert(0, %r); import relaxor; "
            "p = relaxor.download_model(%r, None); print(p)") % (str(ENGINE_DIR), str(MODEL_DIR))
    return subprocess.call([PYTHON, "-c", code])


def main():
    ap = argparse.ArgumentParser(description="Padarth Khoj web application")
    ap.add_argument("--host", default=os.environ.get("PK_HOST", "127.0.0.1"), help="use 0.0.0.0 to serve other computers")
    ap.add_argument("--port", type=int, default=_int("PK_PORT", 8000))
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--prefetch-model", action="store_true", help="download the MACE-MH-1 model and exit")
    a = ap.parse_args()
    if a.prefetch_model:
        sys.exit(prefetch_model())
    import uvicorn
    if a.open:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open("http://%s:%d" % ("127.0.0.1" if a.host == "0.0.0.0" else a.host, a.port))).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info", timeout_keep_alive=120)


if __name__ == "__main__":
    main()
