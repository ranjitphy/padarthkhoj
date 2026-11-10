'use strict';
// Runs Python (Pyodide) off the page's main thread so the interface stays responsive during calculations.
// Everything stays in the browser: files the user opens are copied into this worker's in-memory folder only.
const params = new URL(self.location.href).searchParams;
const PYODIDE = params.get('pyodide') || 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const SITE = new URL('./', self.location.href).href;
let py = null;

const FILES = {
  py: ['engine.py', 'dos_predictor.py', 'pk_browser.py'],
  data: ['oer.csv', 'orr.csv'],
  models: ['model_config.json', 'metrics.json', 'energy_grid.npy', 'stage1.npz', 'stage2.npz']
};
const post = (m) => self.postMessage(m);
const status = (text, percent) => post({ type: 'status', text, percent });

async function init() {
  status('Loading the Python runtime...', 3);
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  status('Loading numpy, pandas, scikit-learn...', 15);
  await py.loadPackage(['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'micropip']);
  status('Loading the spreadsheet writer...', 55);
  const mp = py.pyimport('micropip');
  await mp.install([SITE + 'vendor/et_xmlfile-2.0.0-py3-none-any.whl', SITE + 'vendor/openpyxl-3.1.5-py2.py3-none-any.whl']);
  status('Loading Padarth Khoj models...', 70);
  for (const d of ['/pk', '/pk/py', '/pk/data', '/pk/models', '/sess']) { try { py.FS.mkdir(d); } catch (e) { /* exists */ } }
  let n = 0; const total = Object.values(FILES).reduce((a, l) => a + l.length, 0);
  for (const [dir, names] of Object.entries(FILES)) {
    for (const name of names) {
      const res = await fetch(SITE + dir + '/' + name);
      if (!res.ok) throw new Error('Cannot load ' + dir + '/' + name + ' (' + res.status + ')');
      py.FS.writeFile('/pk/' + dir + '/' + name, new Uint8Array(await res.arrayBuffer()));
      status('Loading Padarth Khoj models...', 70 + Math.round(25 * (++n) / total));
    }
  }
  py.runPython("import sys; sys.path.insert(0, '/pk/py'); import pk_browser");
  status('Ready', 100);
}

const ready = init().then(() => post({ type: 'ready' }), (e) => post({ type: 'fatal', error: String(e && e.message || e) }));

function ensureDir(path) {
  const parts = path.split('/').filter(Boolean); let cur = '';
  for (const p of parts) { cur += '/' + p; try { py.FS.mkdir(cur); } catch (e) { /* exists */ } }
}

self.onmessage = async (ev) => {
  const m = ev.data;
  await ready;
  if (!py) return post({ id: m.id, ok: false, error: 'The Python runtime could not be started.' });
  try {
    if (m.op === 'call') {
      const fn = py.globals.get('pk_browser').call;
      const out = fn(m.name, JSON.stringify(m.arg === undefined ? null : m.arg));
      post({ id: m.id, ok: true, result: JSON.parse(out) });
    } else if (m.op === 'write') {
      const dir = m.path.replace(/\/[^/]*$/, '');
      ensureDir(dir);
      py.FS.writeFile(m.path, new Uint8Array(m.data));
      post({ id: m.id, ok: true, result: { size: m.data.byteLength } });
    } else if (m.op === 'read') {
      const data = py.FS.readFile(m.path);
      const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      self.postMessage({ id: m.id, ok: true, result: buf }, [buf]);
    }
  } catch (e) { post({ id: m.id, ok: false, error: String(e && e.message || e) }); }
};
