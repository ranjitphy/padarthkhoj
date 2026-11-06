"""Runs vasp-mace (https://github.com/rgraucrespo/vasp-mace) in the current directory.

Started by relaxor.py.  Arguments: the vasp-mace options, plus two of ours that are removed first:
  --head NAME        head of a multi-head MACE model (vasp-mace itself has no such option)
  --parent-pid PID   the worker exits by itself when this process (the engine) goes away, so a
                     cancelled job can never keep running in the background

No stdin pipe is used for that: a thread blocked on a pipe read can stall other I/O on Windows.
While PyTorch / MACE are being imported (and until the first step is printed) a watchdog prints a
Python stack trace every 45 s, so a hang shows in the app's Log tab with the exact place it is stuck.
"""
import faulthandler
import os
import sys
import threading
import time

_T0 = time.time()


def _say(msg):
    print("[load] %s (%.0f s)" % (msg, time.time() - _T0), flush=True)


def _take(argv, name):
    """Remove '--name VALUE' from argv; return (value or None, remaining argv)."""
    out, val, i = [], None, 0
    while i < len(argv):
        if argv[i] == name and i + 1 < len(argv):
            val = argv[i + 1]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return val, out


def _watch_parent(pid):
    """Exit when the process `pid` is gone."""
    try:
        if os.name == "nt":
            import ctypes

            k = ctypes.windll.kernel32
            k.OpenProcess.restype = ctypes.c_void_p
            k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            k.WaitForSingleObject.restype = ctypes.c_uint32
            k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            h = k.OpenProcess(0x00100000, 0, pid)  # SYNCHRONIZE
            if not h:
                return  # cannot watch it: keep running
            while True:
                if k.WaitForSingleObject(h, 1000) == 0:  # the process has ended
                    os._exit(3)
        else:
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    os._exit(3)
                except PermissionError:
                    pass
                time.sleep(1.0)
    except Exception:  # noqa: BLE001
        return


class _Out:
    """Pass-through for stdout that stops the stack-trace watchdog at the first relaxation step."""

    def __init__(self, inner):
        self._inner = inner
        self._seen = False

    def write(self, s):
        if not self._seen and "[step" in s:
            self._seen = True
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:  # noqa: BLE001
                pass
        return self._inner.write(s)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch_head(head):
    """vasp-mace builds MACECalculator without a head, which fails for multi-head models such as MACE-MH-1.
    Give every MACECalculator the requested head, and fail loudly if the model does not have it
    (mace-torch would otherwise silently fall back to the last head)."""
    import io
    import warnings
    from contextlib import redirect_stderr, redirect_stdout

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from mace.calculators.mace import MACECalculator

    orig = MACECalculator.__init__

    def init(self, *a, **kw):
        if kw.get("head") is None:
            kw["head"] = head
        orig(self, *a, **kw)
        if getattr(self, "head", None) != head:
            raise ValueError("head '%s' is not in this model. Available heads: %s" % (head, ", ".join(map(str, self.available_heads))))
        print("[info] Head=%s (available: %s)" % (self.head, ", ".join(map(str, self.available_heads))), file=sys.__stdout__, flush=True)

    MACECalculator.__init__ = init


def _import_step(name, label=None):
    import importlib
    import io
    import warnings
    from contextlib import redirect_stderr, redirect_stdout

    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            mod = importlib.import_module(name)
    _say("%s %s ready, import took %.1f s" % (label or name, getattr(mod, "__version__", ""), time.time() - t))
    return mod


code = 0
try:
    head, rest = _take(sys.argv[1:], "--head")
    ppid, rest = _take(rest, "--parent-pid")
    if ppid and ppid.isdigit():
        threading.Thread(target=_watch_parent, args=(int(ppid),), daemon=True).start()
    try:
        faulthandler.dump_traceback_later(45, repeat=True, file=sys.__stdout__)
    except Exception:  # noqa: BLE001
        pass
    sys.stdout = _Out(sys.stdout)

    _say("Worker started (Python %s)" % sys.version.split()[0])
    _import_step("torch", "PyTorch")
    _import_step("ase", "ASE")
    _import_step("e3nn", "e3nn")
    # NOTE: do not import `mace` before vasp_mace.cli: importing mace sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1,
    # which vasp_mace.cli removes right after its own imports; mace must be imported afterwards (load_calc does).
    t_cli = time.time()
    from vasp_mace.cli import main

    _say("vasp-mace imported, took %.1f s" % (time.time() - t_cli))

    if head:
        t_m = time.time()
        _patch_head(head)
        _say("MACE imported, took %.1f s" % (time.time() - t_m))
    _say("Loading the model and starting the optimiser")
    sys.argv = ["vasp-mace"] + rest
    main()
except SystemExit as e:  # main() exits with 1 on error
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException as e:  # noqa: BLE001
    print("[error] %s: %s" % (type(e).__name__, e))
    code = 1
finally:
    try:
        faulthandler.cancel_dump_traceback_later()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)
