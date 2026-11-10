'use strict';
// =========================================================================
// Padarth Khoj - browser bridge. Provides the same window.electronAPI the desktop app uses,
// but over HTTP: file pickers upload to the server, save dialogs become downloads.
// =========================================================================
(function () {
  const sidKey = 'pk-session';
  const rnd = () => {
    const b = new Uint8Array(16); crypto.getRandomValues(b);
    return Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  };
  let sid = null;
  try { sid = sessionStorage.getItem(sidKey); } catch (e) { /* ignore */ }
  if (!sid || !/^[A-Za-z0-9_-]{16,64}$/.test(sid)) { sid = rnd(); try { sessionStorage.setItem(sidKey, sid); } catch (e) { /* ignore */ } }

  const web = { info: null, activityGroup: null };
  const baseName = (p) => String(p || '').split(/[\\/]/).pop();

  // ---------------------------------------------------------------- toast
  let toastEl = null, toastTimer = null;
  function toast(text, ms) {
    if (!toastEl) {
      toastEl = document.createElement('div');
      toastEl.className = 'web-toast'; toastEl.setAttribute('role', 'status');
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = text; toastEl.hidden = false;
    clearTimeout(toastTimer);
    if (ms) toastTimer = setTimeout(() => { toastEl.hidden = true; }, ms);
  }
  const untoast = () => { if (toastEl) toastEl.hidden = true; };

  // ---------------------------------------------------------------- in-browser Python worker
  const wparams = new URLSearchParams(location.search);
  const WORKER_URL = 'worker.js' + (wparams.get('pyodide') ? '?pyodide=' + encodeURIComponent(wparams.get('pyodide')) : '');
  let worker = null, workerReady = null, nextId = 1;
  const pending = new Map();
  const progressCbs = new Set();
  const statusCbs = new Set();
  web.status = { text: 'Starting...', percent: 0, ready: false, error: null };

  function startWorker() {
    web.status = { text: 'Starting...', percent: 0, ready: false, error: null };
    worker = new Worker(WORKER_URL);
    workerReady = new Promise((resolve, reject) => {
      worker.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === 'status') { web.status = Object.assign(web.status, { text: m.text, percent: m.percent }); statusCbs.forEach((cb) => cb(web.status)); }
        else if (m.type === 'ready') { web.status.ready = true; statusCbs.forEach((cb) => cb(web.status)); resolve(); }
        else if (m.type === 'fatal') { web.status.error = m.error; statusCbs.forEach((cb) => cb(web.status)); reject(new Error(m.error)); }
        else if (m.type === 'progress') { progressCbs.forEach((cb) => { try { cb(m); } catch (e) { console.error(e); } }); }
        else if (m.id && pending.has(m.id)) {
          const p = pending.get(m.id); pending.delete(m.id);
          if (m.ok) p.resolve(m.result); else p.reject(new Error(m.error));
        }
      };
      worker.onerror = (e) => { web.status.error = e.message || 'worker error'; reject(new Error(web.status.error)); };
    });
    workerReady.catch(() => {});
    return workerReady;
  }
  startWorker();

  async function send(msg, transfer) {
    await workerReady;
    return new Promise((resolve, reject) => {
      const id = nextId++; pending.set(id, { resolve, reject });
      worker.postMessage(Object.assign({ id }, msg), transfer || []);
    });
  }
  const call = (name, arg) => send({ op: 'call', name, arg });

  // cancelling a running calculation = throwing the worker away and starting a fresh one
  async function cancelJob() {
    for (const p of pending.values()) p.reject(new Error('Cancelled'));
    pending.clear();
    worker.terminate();
    toast('Cancelled. Restarting the Python engine ...');
    startWorker();
    try { await workerReady; } finally { untoast(); }
    return { success: true, message: 'Current job terminated.' };
  }

  async function upload(file, group, name) {
    const buf = await file.arrayBuffer();
    const fname = (name || file.name).replace(/[^\w.\-+() ]/g, '_').slice(0, 120) || 'file';
    const g = group || ('u' + rnd().slice(0, 8));
    await send({ op: 'write', path: '/sess/' + g + '/' + fname, data: buf }, [buf]);
    return { group: g, path: '~/' + g + '/' + fname, name: fname, sizeBytes: file.size };
  }

  async function uploadAll(files, group) {
    const total = files.reduce((a, f) => a + f.size, 0);
    toast('Reading ' + files.map((f) => f.name).join(', ') + ' (' + (total / 1048576).toFixed(1) + ' MB) ...');
    try {
      const out = [];
      let g = group;                       // all files of one selection share a folder
      for (const f of files) { const r = await upload(f, g); g = g || r.group; out.push(r); }
      return out;
    } finally { untoast(); }
  }

  function download(blobOrUrl, filename) {
    const a = document.createElement('a');
    const isBlob = typeof blobOrUrl !== 'string';
    const url = isBlob ? URL.createObjectURL(blobOrUrl) : blobOrUrl;
    a.href = url; a.download = filename; a.style.display = 'none';
    document.body.appendChild(a); a.click(); a.remove();
    if (isBlob) setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  async function downloadServerFile(path, filename) {
    if (typeof path !== 'string' || !path.startsWith('~/') || path.includes('..')) throw new Error('Invalid file reference.');
    const buf = await send({ op: 'read', path: '/sess/' + path.slice(2) });
    download(new Blob([buf]), filename || baseName(path));
  }

  // ---------------------------------------------------------------- file pickers
  function pickFiles(multiple, accept) {
    return new Promise((resolve) => {
      const inp = document.createElement('input');
      inp.type = 'file'; inp.multiple = !!multiple; if (accept) inp.accept = accept;
      inp.style.display = 'none'; document.body.appendChild(inp);
      let done = false;
      const fin = (v) => { if (done) return; done = true; inp.remove(); resolve(v); };
      inp.addEventListener('change', () => fin(inp.files && inp.files.length ? Array.from(inp.files) : null));
      inp.addEventListener('cancel', () => fin(null));
      inp.click();
    });
  }

  const classify = (name) => {
    const n = name.toUpperCase();
    if (n.startsWith('INCAR')) return 'incar';
    if (n.startsWith('KPOINTS')) return 'kpoints';
    if (n.endsWith('.MODEL')) return 'model';
    if (n.startsWith('POSCAR') || n.startsWith('CONTCAR') || n.endsWith('.VASP')) return 'poscar';
    return null;
  };

  async function selectDoscar() {
    const files = await pickFiles(true);
    if (!files) return null;
    const main = files.find((f) => /^DOSCAR/i.test(f.name)) || files[0];
    try {
      const ups = await uploadAll(files, null);
      const group = ups[0].group;
      web.activityGroup = group;
      const m = ups[files.indexOf(main)];
      const others = files.filter((f) => f !== main).map((f) => f.name);
      return { path: m.path, name: m.name, directory: others.length ? `uploaded together with ${others.join(', ')}` : 'uploaded file', sizeBytes: m.sizeBytes };
    } catch (e) { toast(e.message, 8000); throw e; }
  }

  async function selectStructure(opts) {
    const files = await pickFiles(false);
    if (!files) return null;
    const target = opts && opts.target;
    try {
      const [u] = await uploadAll(files, target === 'activity' ? web.activityGroup : null);
      if (target === 'activity' && !web.activityGroup) web.activityGroup = u.group;
      return { path: u.path, name: u.name, directory: 'uploaded file' };
    } catch (e) { toast(e.message, 8000); throw e; }
  }

  async function selectRelaxInputs(kind) {
    if (kind === 'model' && !(web.info && web.info.allow_custom_model)) {
      toast('Custom model files are disabled on this server.', 6000);
      return null;
    }
    const files = await pickFiles(kind === 'all', kind === 'model' ? '.model' : '');
    if (!files) return null;
    const out = [];
    const group = 'rx' + rnd().slice(0, 8);
    for (const f of files) {
      const k = kind === 'all' ? classify(f.name) : kind;
      if (!k) { out.push({ path: f.name, name: f.name, kind: null }); continue; }
      const item = { path: f.name, name: f.name, directory: 'uploaded file', kind: k };
      try {
        if (k === 'incar' || k === 'kpoints') {
          if (f.size > 1048576) item.error = 'File is larger than 1 MB.';
          else {
            const text = await f.text();
            if (text.includes('\u0000')) item.error = 'This does not look like a text file.';
            else item.text = text;
          }
        }
        if (k !== 'incar' && !item.error) item.path = (await uploadAll([f], group))[0].path;
      } catch (e) { item.error = e.message; }
      out.push(item);
    }
    return out;
  }

  // ---------------------------------------------------------------- saving
  async function saveFile({ sourcePath, defaultName }) {
    try {
      await downloadServerFile(sourcePath, defaultName || baseName(sourcePath));
      return { success: true, savedPath: defaultName || baseName(sourcePath) + ' (downloads folder)' };
    } catch (e) { return { success: false, error: e.message }; }
  }
  async function savePng({ dataUrl, defaultName }) {
    if (!/^data:image\/png;base64,/.test(dataUrl || '')) return { success: false, error: 'Not a PNG data URL.' };
    download(dataUrl, defaultName || 'image.png');
    return { success: true, savedPath: (defaultName || 'image.png') + ' (downloads folder)' };
  }
  async function withDownload(name, arg) {
    const r = await call(name, arg);
    if (r && r.success && r.downloadPath) {
      const fn = baseName(r.downloadPath);
      await downloadServerFile(r.downloadPath, fn);
      r.savedPath = fn + ' (downloads folder)';
    }
    return r;
  }

  // ---------------------------------------------------------------- full screen
  const fsCbs = new Set();
  document.addEventListener('fullscreenchange', () => fsCbs.forEach((cb) => cb(!!document.fullscreenElement)));
  const EXTERNAL = new Set(['github.com', 'doi.org', 'pypi.org', 'zenodo.org', 'arxiv.org', 'huggingface.co']);

  const api = {
    initModels: () => call('initModels'),
    getEngineStatus: () => call('getEngineStatus'),
    cancelJob,
    parseVaspHeader: (p) => call('parseVaspHeader', p),
    parseStructure: (a) => call('parseStructure', a),
    runAnalysis: (o) => call('runAnalysis', o),
    dosStatus: () => call('dosStatus'),
    dosLoadStructure: (p) => call('dosLoadStructure', p),
    dosPredict: (p) => call('dosPredict', p),
    dosGetAtom: (i) => call('dosGetAtom', i),
    dosExport: (o) => withDownload('dosExport', o),
    relaxStatus: () => call('relaxStatus'),
    relaxDownloadModel: (o) => call('relaxDownloadModel', o),
    relaxRun: (o) => call('relaxRun', o),
    relaxExportZip: (o) => withDownload('relaxExportZip', o),
    selectRelaxInputs, selectDoscar, selectStructure,
    openExternal: async (url) => {
      try {
        const u = new URL(String(url));
        if (u.protocol !== 'https:' || !EXTERNAL.has(u.hostname)) return { success: false, error: 'Link not allowed' };
        window.open(u.toString(), '_blank', 'noopener,noreferrer');
        return { success: true };
      } catch (e) { return { success: false, error: e.message }; }
    },
    saveFile, savePng,
    openFolder: async () => ({ success: false, error: 'Not available in the web version' }),
    showInFolder: async () => ({ success: false, error: 'Not available in the web version' }),
    copyToClipboard: async (text) => {
      try { await navigator.clipboard.writeText(String(text)); }
      catch (e) {
        const t = document.createElement('textarea'); t.value = String(text); t.style.position = 'fixed'; t.style.opacity = '0';
        document.body.appendChild(t); t.select();
        const ok = document.execCommand && document.execCommand('copy'); t.remove();
        if (!ok) throw e;
      }
      return { success: true };
    },
    getAppInfo: async () => ({ version: web.info ? web.info.version : '', platform: 'web', arch: '', electron: null, engine: 'in-browser Python (Pyodide)' }),
    toggleFullScreen: async () => { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); return !!document.fullscreenElement; },
    setFullScreen: async (on) => {
      try { if (on && !document.fullscreenElement) await document.documentElement.requestFullscreen(); else if (!on && document.fullscreenElement) await document.exitFullscreen(); } catch (e) { /* denied */ }
      return !!document.fullscreenElement;
    },
    isFullScreen: async () => !!document.fullscreenElement,
    minimizeWindow: async () => {}, maximizeWindow: async () => {}, closeWindow: async () => {},
    setTheme: async (t) => t,
    onProgress: (cb) => { progressCbs.add(cb); return () => progressCbs.delete(cb); },
    onMenuAction: () => () => {},
    onFullScreenChange: (cb) => { fsCbs.add(cb); return () => fsCbs.delete(cb); }
  };
  window.electronAPI = api; window.padarthAPI = api; window.__web = web;

  // ---------------------------------------------------------------- page chrome
  document.addEventListener('DOMContentLoaded', async () => {
    document.body.classList.add('web');
    document.body.classList.add('no-custom-model');
    web.info = { version: window.PK_VERSION || '', allow_custom_model: false };
    // loading banner (first visit downloads the Python runtime, about 40 MB, then it is cached by the browser)
    const bar = document.createElement('div'); bar.className = 'load-banner'; bar.setAttribute('role', 'status');
    bar.innerHTML = '<span id="loadText">Starting...</span><div class="load-track"><div id="loadFill" class="load-fill"></div></div>';
    document.body.appendChild(bar);
    const upd = (st) => {
      document.getElementById('loadText').textContent = st.error ? 'Could not start the Python engine: ' + st.error : (st.ready ? 'Ready' : st.text);
      document.getElementById('loadFill').style.width = (st.ready ? 100 : st.percent || 0) + '%';
      if (st.ready) setTimeout(() => bar.remove(), 1200);
      if (st.error) bar.classList.add('load-error');
    };
    statusCbs.add(upd); upd(web.status);
    // only one header dropdown open at a time; close on outside click / Escape
    const dds = Array.from(document.querySelectorAll('.web-dd'));
    dds.forEach((d) => d.addEventListener('toggle', () => { if (d.open) dds.forEach((o) => { if (o !== d) o.open = false; }); }));
    document.addEventListener('click', (e) => { dds.forEach((d) => { if (!d.contains(e.target)) d.open = false; }); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') dds.forEach((d) => { d.open = false; }); });
  });
})();
