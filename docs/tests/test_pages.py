"""Headless-browser test of the static site (everything runs in Pyodide, no server code).
usage: python tests/test_pages.py <site dir> <pyodide dir (full release, for offline use)> [synthetic-case dir] [structure dir]
The Pyodide release is served locally from /_pyodide/ (the real site loads it from the jsDelivr CDN)."""
import os, sys, threading, http.server, functools, tempfile, shutil, zipfile, time
from playwright.sync_api import sync_playwright

site, pyo = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
case, struct = sys.argv[3], sys.argv[4]
stage = tempfile.mkdtemp(prefix="pk_stage_")
for n in os.listdir(site):
    if n not in ("tests", "tools", ".github"):
        os.symlink(os.path.join(site, n), os.path.join(stage, n))
os.symlink(pyo, os.path.join(stage, "_pyodide"))

class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=stage))
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d/?pyodide=/_pyodide/" % srv.server_address[1]
fails = []
def ok(c, m):
    print(("PASS  " if c else "FAIL  ") + m, flush=True)
    if not c: fails.append(m)

with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--no-sandbox"])
    # --- 1. real CSP (no bypass): runtime starts, no policy violations
    ctx = br.new_context(accept_downloads=True)
    page = ctx.new_page(); msgs = []
    page.on("console", lambda m: msgs.append(m.text)); page.on("pageerror", lambda e: msgs.append(str(e)))
    t0 = time.time(); page.goto(base)
    page.wait_for_selector("#modelStatusText:has-text('ready')", timeout=600000)
    ok(True, "Python runtime + ML models ready in %.0f s (strict CSP active)" % (time.time() - t0))
    ok(not [m for m in msgs if "Content Security Policy" in m or "Refused" in m], "no CSP violations %s" % [m for m in msgs if "Content Security" in m][:2])
    ok(page.inner_text("#sectionNav").count("desktop only") == 1, "Relaxor tab marked desktop only")
    ctx.close()

    # --- 2. functional run
    ctx = br.new_context(accept_downloads=True, bypass_csp=True)
    page = ctx.new_page(); page.goto(base)
    page.wait_for_function("document.getElementById('modelStatusText').textContent.includes('ready')", timeout=600000)
    ok(page.evaluate("document.getElementById('sectionDos').classList.contains('active')"), "opens on the DOS Predictor tab")
    ok(page.evaluate("[...document.querySelectorAll('.section-btn')].map(b=>b.dataset.section).join()") == "relax,dos,activity", "tab order: Relaxor, DOS Predictor, Activity Screening")

    # Activity Screening
    page.click("#navActivity")
    with page.expect_file_chooser() as fc: page.click("#uploadBtn")
    fc.value.set_files([os.path.join(case, n) for n in ("DOSCAR", "CONTCAR", "OUTCAR")])
    page.wait_for_function("document.getElementById('fileDetails').textContent.includes('atoms')", timeout=120000)
    ok(True, "DOSCAR + CONTCAR + OUTCAR read in the browser: " + page.inner_text("#fileDetails").replace("\n", " | ")[:100])
    page.click("#runBtn")
    page.wait_for_function("window.__pk.state.records && window.__pk.state.records.length > 0", timeout=300000)
    n = page.evaluate("window.__pk.state.records.length")
    ok(n > 0, "Activity Screening ran in the browser: %d sites" % n)
    best = page.evaluate("window.__pk.state.records.filter(r=>r.Overpotential!=null).map(r=>r.Overpotential).sort((a,b)=>a-b)[0]")
    print("      lowest overpotential:", best)
    with page.expect_download() as dl: page.click("#downloadExcelBtn")
    p = dl.value.path()
    ok(os.path.getsize(p) > 3000 and zipfile.is_zipfile(p), "Excel result downloaded (%d bytes)" % os.path.getsize(p))
    ok(not page.is_visible("#openFolderBtn"), "desktop-only 'Open folder' hidden")

    # DOS Predictor
    page.click("#navDos")
    with page.expect_file_chooser() as fc: page.click("#dosOpenBtn")
    fc.value.set_files(os.path.join(struct, "CONTCAR"))
    page.wait_for_function("!document.getElementById('dosRunBtn').disabled", timeout=120000)
    t0 = time.time(); page.click("#dosRunBtn")
    page.wait_for_function("window.__dos.state.result !== null", timeout=900000)
    na = page.evaluate("window.__dos.state.result.atoms.length")
    ok(na > 0, "PDOS predicted in the browser for %d atoms in %.0f s" % (na, time.time() - t0))
    ok(page.evaluate("window.__dos.state.result.curves.length") == 8, "8 orbital curves returned")
    page.wait_for_function("!document.getElementById('dosExportBtn').disabled", timeout=60000)
    with page.expect_download(timeout=600000) as dl: page.click("#dosExportBtn")
    z = zipfile.ZipFile(dl.value.path())
    ok(len(z.namelist()) >= na, "PDOS plots ZIP downloaded (%d images)" % len(z.namelist()))
    page.screenshot(path=os.path.join(tempfile.gettempdir(), "pk_pages_dos.png"))

    # Relaxor tab = notice
    page.click("#navRelax")
    ok("not available in the browser" in page.inner_text("#sectionRelax"), "Relaxor tab explains it is desktop only")
    br.close()
shutil.rmtree(stage, ignore_errors=True)
print("ALL PASSED" if not fails else "FAILED: %s" % fails)
sys.exit(1 if fails else 0)
