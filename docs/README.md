# Padarth Khoj Online

Padarth Khoj in your browser, as a link. No installation, no server, no upload: the calculations run on **your own computer**
inside the web page (Python via [Pyodide](https://pyodide.org)), so your DOSCAR / structure files never leave it.

| Tool | In this web page |
|---|---|
| **Activity Screening** (OER / ORR overpotentials from DOSCAR + CONTCAR + OUTCAR, SVR / MLR) | yes, same Python code as the desktop app |
| **DOS Predictor** (PDOS of N-doped graphene structures, 2-stage graph network) | yes, the network is re-implemented in NumPy (checked against PyTorch, max difference about 5e-6) |
| **Structure Relaxor** (vasp-mace + MACE-MH-1) | no: needs PyTorch and GBs of software. Use the desktop app or the self-hosted web version |

The first visit downloads the Python runtime (about 40 MB from the jsDelivr CDN) and the models (about 13 MB); the browser caches them afterwards.
Speed: Activity Screening takes seconds. The DOS Predictor runs in WebAssembly, which is slower than the desktop app: about 1.5 minutes
for 48 atoms, roughly 3-4 minutes for 100 atoms. Keep the tab open while it runs.

## Put it online (GitHub Pages)

1. Create a new repository on github.com, for example `padarthkhoj-online` (Public).
2. Upload the **contents** of this folder (drag the files and folders in, including the hidden `.nojekyll`). Commit.
   - The web uploader accepts about 100 files per upload; this folder has fewer.
3. In the repository open **Settings -> Pages**. Under "Build and deployment" choose **Deploy from a branch**, branch **main**, folder **/ (root)**, and Save.
4. After one or two minutes the site is live at `https://<your-github-name>.github.io/<repository-name>/`, for example
   `https://ranjitphy.github.io/padarthkhoj-online/`. Share that link.

To put it into the existing `padarthkhoj` repository instead, upload everything into a folder named `docs` and choose **/docs** in step 3
(the link is then `https://ranjitphy.github.io/padarthkhoj/`). Do not upload into the root of a repository that already has an `index.html` or `README.md`: they would be overwritten.

## Run it locally (to test)

```
python -m http.server 8000
```
then open http://localhost:8000 (opening `index.html` by double-click does not work: browsers block web workers on `file://`).

## What is in this folder

| Path | Purpose |
|---|---|
| `index.html`, `styles.css`, `*.js` | the interface (same as the desktop / web app) |
| `static-shim.js`, `worker.js` | run the Python engine in a Web Worker, handle file uploads and downloads |
| `py/engine.py` | Activity Screening engine (identical to the desktop app) |
| `py/dos_predictor.py` | NumPy version of the PDOS networks, structure readers (POSCAR, CIF, XYZ, CC1) |
| `py/pk_browser.py` | glue between the page and the engine |
| `models/` | PDOS network weights (`stage1.npz`, `stage2.npz`) and configuration |
| `data/` | training tables `oer.csv`, `orr.csv` |
| `vendor/` | `openpyxl` + `et_xmlfile` wheels (Excel output) |
| `tools/export_weights.py` | converts the PyTorch checkpoints to the `.npz` files (needs torch) |
| `tests/` | `verify_numpy_port.py` (NumPy vs PyTorch) and `test_pages.py` (headless-browser test) |

## Notes

- Supported structure files for the DOS Predictor: POSCAR / CONTCAR / .vasp, CIF (P1 and symmetry operators), XYZ / extxyz, .cc1.
  Other formats that the desktop app reads through ASE (pdb, mol, sdf, ...) are not supported here.
- Needs a current Chrome, Edge, Firefox or Safari and an internet connection on the first visit.
- Results come from machine-learning models trained on the data in `data/` and `models/`; they are screening estimates, not DFT.
