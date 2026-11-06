"""Browser end-to-end test of the web app (real Chromium against the real server).

Usage:  python tests/web/test_web.py
Env:    PK_TEST_SERVER_PYTHON  python that has the server + engine packages (default: this interpreter)
        PK_TEST_DATA_DIR       data dir for the server; put mace-mh-1.model in <dir>/mace-models to include the relaxor
Needs:  pip install playwright && playwright install chromium
"""
import json, os, shutil, socket, subprocess, sys, tempfile, time, urllib.request, urllib.error, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SERVER_PY = os.environ.get("PK_TEST_SERVER_PYTHON", sys.executable)
fails = 0


def ok(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails += 1


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def api(base, name, arg=None, sid="a" * 24):
    req = urllib.request.Request(base + "/api/call/" + name, data=json.dumps({"arg": arg}).encode(),
                                 headers={"Content-Type": "application/json", "X-Session": sid}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    tmp = tempfile.mkdtemp(prefix="pk_web_")
    data = os.environ.get("PK_TEST_DATA_DIR") or os.path.join(tmp, "data")
    case, struct = os.path.join(tmp, "case"), os.path.join(tmp, "struct")
    subprocess.check_call([SERVER_PY, os.path.join(ROOT, "tests", "parity", "make_synthetic_vasp.py"), case], stdout=subprocess.DEVNULL)
    subprocess.check_call([SERVER_PY, os.path.join(ROOT, "tests", "dos", "make_structures.py"), struct], stdout=subprocess.DEVNULL)
    with open(os.path.join(tmp, "INCAR"), "w") as f:
        f.write("SYSTEM = web test\nIBRION = 2\nNSW = 5\nISIF = 2\nEDIFFG = -0.05\n")
    with open(os.path.join(tmp, "KPOINTS"), "w") as f:
        f.write("Automatic\n0\nGamma\n1 1 1\n")
    shutil.copy(os.path.join(struct, "CONTCAR"), os.path.join(tmp, "POSCAR"))

    port = free_port(); base = "http://127.0.0.1:%d" % port
    env = dict(os.environ, PK_DATA_DIR=data, PK_PYTHON=SERVER_PY, PK_MAX_NSW="50")
    srv = subprocess.Popen([SERVER_PY, os.path.join(ROOT, "server", "app.py"), "--port", str(port)], env=env,
                           stdout=open(os.path.join(tmp, "server.log"), "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/api/info", timeout=1); break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("server did not start: " + open(os.path.join(tmp, "server.log")).read()[-2000:])

        # ---------------- API-level security checks
        st, b = api(base, "dosStatus", sid="short")
        ok(st == 400, "invalid session id rejected")
        st, b = api(base, "parseStructure", "/etc/passwd")
        ok(st == 400, "absolute server path rejected")
        st, b = api(base, "parseStructure", "~/../../etc/passwd")
        ok(st == 400, "path traversal rejected")
        st, b = api(base, "relaxRun", {"poscarPath": "~/x", "incarText": "NSW = 9999\nIBRION = 2\n", "modelPath": "@models/a.model"})
        ok(st == 400 and "limit" in b.get("error", ""), "NSW above the server limit rejected")
        st, b = api(base, "relaxRun", {"poscarPath": "~/x", "incarText": "NSW = 5\n", "modelPath": "~/inputs/x/evil.model"})
        ok(st == 403, "custom .model upload refused by default")
        st, b = api(base, "relaxCheckEnv")
        ok(st == 404, "no environment-check / arbitrary Python route on the server")
        st, b = api(base, "nope")
        ok(st == 404, "unknown call -> 404")
        req = urllib.request.Request(base + "/api/download?path=~/../../etc/passwd", headers={"X-Session": "b" * 24})
        try:
            urllib.request.urlopen(req); code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        ok(code == 400, "download outside the session folder rejected")

        # ---------------- cancel really stops a running relaxation (needs the MACE model)
        if os.path.isfile(os.path.join(data, "mace-models", "mace-mh-1.model")):
            import threading
            sid = "c" * 24
            data_poscar = open(os.path.join(tmp, "POSCAR"), "rb").read()
            up = urllib.request.Request(base + "/api/upload?name=POSCAR&group=cx", data=data_poscar, headers={"X-Session": sid}, method="POST")
            urllib.request.urlopen(up).read()
            box = {}
            t = threading.Thread(target=lambda: box.update(r=api(base, "relaxRun", {"poscarPath": "~/inputs/cx/POSCAR", "incarText": "NSW = 50\nIBRION = 2\nEDIFFG = -0.001\n",
                                                                                  "modelPath": "@models/mace-mh-1.model", "head": "omat_pbe", "device": "cpu", "dtype": "float32", "optimizer": "BFGS"}, sid=sid)))
            t.start(); time.sleep(12)
            api(base, "cancelJob", sid=sid)
            t.join(60)
            ok(not t.is_alive() and box.get("r", (0,))[0] == 409 and "CANCELLED" in box["r"][1].get("error", ""), "cancel stops the relaxation (HTTP 409 CANCELLED)")
            left = ""
            for _ in range(40):   # the worker notices the closed pipe and exits within a few seconds
                left = subprocess.run(["pgrep", "-f", r"^\S*python\S*\s.*relax_worker\.py"], capture_output=True, text=True).stdout.strip() if os.name == "posix" else ""
                if not left:
                    break
                time.sleep(0.5)
            ok(left == "", "no relax_worker process left behind " + left)
            st, b = api(base, "dosStatus", sid=sid)
            ok(st == 200 and b.get("available"), "engine restarts after a cancel")

        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            br = pw.chromium.launch(args=["--no-sandbox"])
            ctx = br.new_context(bypass_csp=True, accept_downloads=True, viewport={"width": 1500, "height": 950})
            page = ctx.new_page()
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(base)
            ok("Padarth Khoj" in page.title(), "page loads")
            page.click("#ddTeam > summary")
            t = page.inner_text("#ddTeam")
            ok("Prof. Ranjit Thapa" in t and "Mr. Arupjyoti Pathak" in t and "founder" in t.lower() and "developer" in t.lower(), "Team menu shows founder and developer")
            page.click("#ddLinks > summary")
            ok(page.get_attribute("#ddLinks a", "href") == "https://github.com/ranjitphy/padarthkhoj", "Links menu has the GitHub link")
            ok(page.evaluate("document.getElementById('ddTeam').open") is False, "opening Links closes Team")
            page.wait_for_function("document.getElementById('modelStatusText').textContent.includes('ready')", timeout=120000)
            ok(True, "ML models trained in the server engine")

            ok(page.evaluate("document.getElementById('sectionRelax').classList.contains('active')"), 'app opens on Structure Relaxor')
            ok(page.evaluate("[...document.querySelectorAll('.section-btn')].map(b=>b.dataset.section).join()") == 'relax,dos,activity', 'tab order: Relaxor, DOS Predictor, Activity Screening')
            # ---------------- Activity Screening
            page.click('#navActivity')
            with page.expect_file_chooser() as fc:
                page.click("#uploadBtn")
            fc.value.set_files([os.path.join(case, n) for n in ("DOSCAR", "CONTCAR", "OUTCAR")])
            page.wait_for_function("document.getElementById('fileDetails').textContent.includes('atoms')", timeout=60000)
            ok(True, "DOSCAR + CONTCAR + OUTCAR uploaded and header read: " + page.inner_text("#fileDetails").replace("\n", " | ")[:120])
            page.click("#runBtn")
            page.wait_for_function("window.__pk.state.records && window.__pk.state.records.length > 0", timeout=240000)
            ok(True, "analysis ran: %d sites" % page.evaluate("window.__pk.state.records.length"))
            with page.expect_download() as dl:
                page.click("#downloadExcelBtn")
            p = dl.value.path()
            ok(os.path.getsize(p) > 3000 and zipfile.is_zipfile(p), "Excel result downloaded (%d bytes)" % os.path.getsize(p))
            ok(not page.is_visible("#openFolderBtn"), "desktop-only 'Open folder' hidden in the web version")

            # ---------------- DOS Predictor
            page.click("#navDos")
            with page.expect_file_chooser() as fc:
                page.click("#dosOpenBtn")
            fc.value.set_files(os.path.join(struct, "CONTCAR"))
            page.wait_for_function("!document.getElementById('dosRunBtn').disabled", timeout=60000)
            page.click("#dosRunBtn")
            page.wait_for_function("window.__dos.state.result !== null", timeout=240000)
            ok(True, "PDOS predicted for %d atoms" % page.evaluate("window.__dos.state.result.atoms.length"))
            page.wait_for_function("!document.getElementById('dosExportBtn').disabled", timeout=30000)
            with page.expect_download(timeout=240000) as dl:
                page.click("#dosExportBtn")
            p = dl.value.path()
            z = zipfile.ZipFile(p)
            ok(len(z.namelist()) >= 40, "PDOS plots ZIP downloaded (%d images)" % len(z.namelist()))

            # ---------------- Structure Relaxor
            page.click("#navRelax")
            page.wait_for_function("document.getElementById('rxEngineNote').textContent.length > 0", timeout=60000)
            ok(not page.is_visible("#rxCustomModelBtn"), "custom model button hidden")
            ok(not page.is_visible("#rxPyRow"), "own-Python option hidden in the web version")
            cite = page.inner_text("#rxCite")
            ok("zenodo.19479246" in cite and "MACE-MH-1" in cite and "2510.25380" in cite, "citation visible (vasp-mace, MACE-MH-1)")
            ok(page.input_value("#rxModelSel") == "mace-mh-1", "MACE-MH-1 is the default model")
            ok("ASL" in page.inner_text("#rxLicense"), "ASL licence note shown")
            model_ok = page.evaluate("window.__relax.state.status && window.__relax.state.status.model_present")
            if not model_ok:
                print("SKIP relaxation run: mace-mh-1.model not present in " + data + "/mace-models")
            else:
                for btn, name in (("#rxPoscarBtn", "POSCAR"), ("#rxIncarBtn", "INCAR"), ("#rxKpointsBtn", "KPOINTS")):
                    with page.expect_file_chooser() as fc:
                        page.click(btn)
                    fc.value.set_files(os.path.join(tmp, name))
                page.wait_for_function("document.getElementById('rxPoscarName').textContent.includes('atoms')", timeout=60000)
                page.wait_for_function("!document.getElementById('rxRunBtn').disabled", timeout=10000)
                page.select_option("#rxDevice", "cpu")
                page.select_option("#rxDtype", "float32")
                ok(page.input_value("#rxHead") == "omat_pbe", "head defaults to omat_pbe")
                page.click("#rxRunBtn")
                page.wait_for_function("window.__relax.state.result !== null", timeout=300000)
                r = page.evaluate("(() => { const x = window.__relax.state.result; return [x.steps, x.initial_energy, x.final_energy]; })()")
                ok(r[2] < r[1], "MH-1 relaxation ran: %d steps, %.3f -> %.3f eV" % tuple(r))
                with page.expect_download() as dl:
                    page.click("#rxSaveContcar")
                ok(os.path.getsize(dl.value.path()) > 500 and dl.value.suggested_filename == "CONTCAR", "CONTCAR downloaded")
                with page.expect_download() as dl:
                    page.click("#rxDownloadRelaxed")
                ok(os.path.getsize(dl.value.path()) > 500 and dl.value.suggested_filename == "CONTCAR", "Download relaxed structure button (under the viewer) saves the CONTCAR")
                ok(page.evaluate("document.querySelectorAll('#rxDownloadRelaxed').length") == 1 and page.evaluate("!document.querySelector('#sectionDos #rxDownloadRelaxed') && !document.querySelector('#sectionActivity #rxDownloadRelaxed')"), "button only in the Relaxor section")
                with page.expect_download() as dl:
                    page.click("#rxSaveZip")
                names = zipfile.ZipFile(dl.value.path()).namelist()
                ok(any(n.endswith("CONTCAR") for n in names) and any(n.endswith("OUTCAR") for n in names), "outputs ZIP downloaded (%d files)" % len(names))
                page.click("#rxToDos")
                page.wait_for_function("document.getElementById('dosStructTitle').textContent.includes('C')", timeout=30000)
                ok(page.is_visible("#sectionDos"), "Send to DOS Predictor loads the relaxed CONTCAR")
            page.screenshot(path=os.path.join(tmp, "web_ui.png"))

            # ---------------- isolation between browser sessions
            sid1 = page.evaluate("sessionStorage.getItem('pk-session')")
            ctx2 = br.new_context(bypass_csp=True); p2 = ctx2.new_page(); p2.goto(base)
            sid2 = p2.evaluate("sessionStorage.getItem('pk-session')")
            ok(sid1 != sid2, "each browser tab gets its own session")
            st, b = api(base, "parseStructure", {"path": "~/inputs/anything/CONTCAR"}, sid=sid2)
            ok(st in (400, 500) or not b.get("success", False), "another session cannot read the first session's uploads")
            ok(not errs, "no page errors " + "; ".join(errs))
            br.close()
        print("screenshot:", os.path.join(tmp, "web_ui.png"))
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()
    print("ALL PASSED" if not fails else "%d FAILED" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
