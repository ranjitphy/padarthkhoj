'use strict';
// =========================================================================
// Padarth Khoj - sections (Activity Screening / DOS Predictor / Structure Relaxor)
// and the DOS Predictor UI: structure -> PDOS of every atom (stage 1 + stage 2 networks)
// =========================================================================
(function () {
  const api = window.electronAPI;
  const $ = (id) => document.getElementById(id);
  const baseName = (p) => String(p || '').split(/[\\/]/).pop();

  // ------------------------------------------------------------ sections
  const SECTIONS = { activity: 'sectionActivity', dos: 'sectionDos', relax: 'sectionRelax' };
  let dosViewer = null;
  let dosStatusLoaded = false;

  function gotoSection(name) {
    if (!SECTIONS[name]) return;
    document.querySelectorAll('.section-btn').forEach((b) => {
      const on = b.dataset.section === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    for (const [k, id] of Object.entries(SECTIONS)) $(id).classList.toggle('active', k === name);
    if (name === 'dos') {
      loadDosStatus();
      requestAnimationFrame(() => { if (dosViewer) dosViewer.invalidate(); drawPlot(); });
    } else if (name === 'relax') {
      window.dispatchEvent(new Event('rx:shown'));
    } else if (window.__pk && window.__pk.viewer) {
      requestAnimationFrame(() => window.__pk.viewer.invalidate());
    }
  }
  document.querySelectorAll('.section-btn').forEach((b) => b.addEventListener('click', () => gotoSection(b.dataset.section)));

  // ------------------------------------------------------------ DOS state
  const S = {
    structure: null,   // {path, name}
    result: null,      // engine result of dos_predict
    energy: null,      // Float64Array(3000)
    labels: [], colors: [],
    cache: new Map(),  // atom number -> Float32Array[8]
    atom: 1,
    busy: false,
    available: null,
    hoverX: null
  };

  const el = {
    open: $('dosOpenBtn'), run: $('dosRunBtn'), runText: $('dosRunText'), prog: $('dosProgress'), progBar: $('dosProgressBar'),
    fileDetails: $('dosFileDetails'), engineNote: $('dosEngineNote'), metrics: $('dosMetrics'),
    picker: $('atomPicker'), prev: $('dosPrev'), next: $('dosNext'), num: $('dosAtomNum'), of: $('dosAtomOf'), tag: $('dosAtomTag'),
    stage: $('plotStage'), canvas: $('plotCanvas'), empty: $('plotEmpty'), tip: $('plotTip'),
    spinUp: $('spinUp'), spinDn: $('spinDn'), eMin: $('eMin'), eMax: $('eMax'), eFull: $('eFull'),
    table: $('dosTable'), tbody: $('dosTableBody'), tableEmpty: $('dosTableEmpty'), log: $('dosLog'),
    savePlot: $('dosSavePlot'), exportBtn: $('dosExportBtn'), exportFmt: $('dosExportFmt'), exportDpi: $('dosExportDpi'),
    structTitle: $('dosStructTitle'), vCanvas: $('dosViewerCanvas'), atomInfo: $('dosAtomInfo'), vTip: $('dosViewerTip'),
    optBonds: $('dosOptBonds'), optCell: $('dosOptCell'), optLabels: $('dosOptLabels'), optRep: $('dosOptRep'), optSize: $('dosOptSize'),
    saveStruct: $('dosSaveStruct')
  };
  const ctx = el.canvas.getContext('2d');
  const log = (t, replace) => { el.log.textContent = replace ? t : el.log.textContent + (el.log.textContent ? '\n' : '') + t; };
  const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? '-' : Number(v).toFixed(d));
  const mk = (tag, props, ...kids) => {
    const n = document.createElement(tag);
    if (props) for (const [k, v] of Object.entries(props)) { if (k === 'class') n.className = v; else n.setAttribute(k, v); }
    for (const k of kids) n.append(k instanceof Node ? k : document.createTextNode(String(k)));
    return n;
  };

  function showDosTab(id) {
    document.querySelectorAll('[data-dostab]').forEach((b) => b.classList.toggle('active', b.dataset.dostab === id));
    ['dosPlotView', 'dosTableView', 'dosLogView'].forEach((t) => $(t).classList.toggle('active', t === id));
    if (id === 'dosPlotView') requestAnimationFrame(drawPlot);
  }
  document.querySelectorAll('[data-dostab]').forEach((b) => b.addEventListener('click', () => showDosTab(b.dataset.dostab)));

  // ------------------------------------------------------------ status of the predictor
  async function loadDosStatus() {
    if (dosStatusLoaded || !api || !api.dosStatus) return;
    dosStatusLoaded = true;
    try {
      const st = await api.dosStatus();
      S.available = !!(st && st.available && st.models_present);
      if (!S.available) {
        const why = !st ? 'engine not reachable'
          : !st.models_present ? 'model files (engine/models) are missing'
            : `missing Python packages: ${(st.missing || []).join(', ')}`;
        el.engineNote.textContent = 'Predictor unavailable';
        el.engineNote.style.color = 'var(--bad)';
        el.fileDetails.querySelector('.file-subtext').textContent =
          `Cannot predict: ${why}. Run "npm run setup:python" (installs PyTorch, ASE, matplotlib) and restart the app.`;
        el.fileDetails.querySelector('.file-subtext').classList.add('warn');
        log(`DOS predictor unavailable: ${why}`);
      } else {
        el.engineNote.textContent = 'Model ready';
        el.engineNote.style.color = 'var(--good)';
      }
      const m = st && st.metrics && st.metrics.stage2_vs_raw;
      if (m) {
        const r2 = (k) => (m[k] ? m[k].R2.toFixed(2) : '-');
        el.metrics.textContent = `Held-out R² vs DFT (your training run, after stage 2): s ${r2('s_up')}, p_y ${r2('py_up')}, p_z ${r2('pz_up')}, p_x ${r2('px_up')} (spin up).`;
      }
    } catch (err) {
      dosStatusLoaded = false;
      log(`Could not query the predictor: ${err.message}`);
    }
    updateButtons();
  }

  function updateButtons() {
    el.run.disabled = !(S.structure && S.available !== false) || S.busy;
    el.open.disabled = S.busy;
    const has = !!S.result;
    el.savePlot.disabled = !has;
    el.exportBtn.disabled = !has || S.busy;
  }

  // ------------------------------------------------------------ choose + predict
  async function chooseStructure() {
    if (S.busy) return;
    try {
      const f = await api.selectStructure({ target: 'dos' });
      if (!f) return;
      gotoSection('dos');
      await loadStructureFile(f);
    } catch (err) { log(`File selection error: ${err.message}`); }
  }
  el.open.addEventListener('click', chooseStructure);

  async function loadStructureFile(f) {
    S.structure = null;          // Predict stays disabled until the file has been read
    S.result = null; S.cache.clear(); S.energy = null; S.atom = 1;
    resetOutputs();
    const sub = mk('span', { class: 'file-subtext' }, f.directory);
    el.fileDetails.replaceChildren(mk('span', { class: 'file-label' }, f.name), sub);
    log(`Structure selected: ${f.path}`, true);
    const s = await api.dosLoadStructure(f.path);
    if (!s || !s.success) {
      sub.textContent = (s && s.error) || 'Could not read this file.';
      sub.classList.add('warn');
      dosViewer.clear();
      el.structTitle.textContent = 'Structure';
      log(sub.textContent);
      updateButtons();
      return;
    }
    S.structure = f;
    const parts = [s.formula, `${s.natoms} atoms`, s.periodic ? 'periodic' : 'no unit cell'];
    sub.textContent = parts.join('  ·  ');
    if (s.warnings && s.warnings.length) {
      sub.classList.add('warn');
      sub.title = s.warnings.join('\n');
      log('Notes:\n  - ' + s.warnings.join('\n  - '));
    }
    dosViewer.setStructure(s);
    el.structTitle.textContent = `${s.formula}  (${s.name})`;
    updateButtons();
  }

  function resetOutputs() {
    el.picker.hidden = true;
    el.empty.hidden = false;
    el.table.hidden = true; el.tableEmpty.hidden = false; el.tbody.textContent = '';
    el.atomInfo.hidden = true;
    el.progBar.style.width = '0'; el.prog.hidden = true;
    drawPlot();
    updateButtons();
  }

  async function predict() {
    if (!S.structure || S.busy) return;
    S.busy = true;
    el.run.classList.add('loading');
    el.runText.textContent = 'Predicting...';
    el.prog.hidden = false; el.progBar.style.width = '2%';
    updateButtons();
    log('Predicting PDOS (stage 1 graph network, stage 2 sharpening network)...');
    try {
      const r = await api.dosPredict(S.structure.path);
      if (!r || !r.success) {
        log(`Prediction failed: ${(r && r.error) || 'unknown error'}`);
        showDosTab('dosLogView');
        return;
      }
      S.result = r;
      S.energy = Float64Array.from(r.energy);
      S.labels = r.labels; S.colors = r.colors;
      S.cache.clear();
      S.cache.set(1, r.curves.map((c) => Float32Array.from(c)));
      S.atom = 1;
      dosViewer.setStructure(r.structure);
      el.structTitle.textContent = `${r.structure.formula}  (${r.structure.name})`;
      renderTable();
      el.num.max = String(r.atoms.length);
      el.empty.hidden = true;
      el.picker.hidden = false;
      setAtom(1, true);
      log(`Done in ${r.elapsed_s} s: ${r.atoms.length} atoms. Showing atom 1; use the arrows or click an atom in the structure.`);
    } catch (err) {
      log(`Prediction error: ${err.message}`);
      showDosTab('dosLogView');
    } finally {
      S.busy = false;
      el.run.classList.remove('loading');
      el.runText.textContent = 'Predict PDOS';
      el.prog.hidden = true;
      updateButtons();
    }
  }
  el.run.addEventListener('click', predict);

  if (api && api.onProgress) {
    api.onProgress((p) => {
      if (!S.busy) return;
      if (typeof p.percent === 'number') el.progBar.style.width = Math.max(2, Math.min(100, p.percent)) + '%';
      if (p.stage === 'export') { if (typeof p.percent === 'number') el.exportBtn.textContent = `Rendering ${Math.round(p.percent)}%`; }
      else if (p.message) el.runText.textContent = p.message.length > 28 ? 'Predicting...' : p.message;
    });
  }

  // ------------------------------------------------------------ atom selection
  async function getCurves(n) {
    if (S.cache.has(n)) return S.cache.get(n);
    const r = await api.dosGetAtom(n);
    if (!r || !r.success) throw new Error((r && r.error) || 'could not get atom');
    const c = r.curves.map((x) => Float32Array.from(x));
    S.cache.set(n, c);
    return c;
  }

  async function setAtom(n, fromPredict) {
    if (!S.result) return;
    const total = S.result.atoms.length;
    n = Math.max(1, Math.min(total, Math.round(n) || 1));
    try { await getCurves(n); } catch (err) { log(`Atom ${n}: ${err.message}`); return; }
    S.atom = n;
    el.num.value = String(n);
    el.of.textContent = `of ${total}`;
    el.tag.firstElementChild.textContent = `${S.result.atoms[n - 1].symbol}${n}`;
    if (dosViewer.selected !== n) dosViewer.select(n, true);
    updateAtomInfo();
    highlightRow(n, !fromPredict);
    drawPlot();
  }
  el.prev.addEventListener('click', () => setAtom(S.atom - 1));
  el.next.addEventListener('click', () => setAtom(S.atom + 1));
  el.num.addEventListener('change', () => setAtom(parseInt(el.num.value, 10)));
  window.addEventListener('keydown', (e) => {
    if (!$('sectionDos').classList.contains('active') || !S.result) return;
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === 'ArrowLeft') setAtom(S.atom - 1);
    else if (e.key === 'ArrowRight') setAtom(S.atom + 1);
  });

  function updateAtomInfo() {
    el.atomInfo.textContent = '';
    if (!S.result || !dosViewer.struct) { el.atomInfo.hidden = true; return; }
    const a = S.result.atoms[S.atom - 1];
    const info = dosViewer.atomInfo(S.atom);
    el.atomInfo.append(mk('div', null, mk('b', null, `${a.symbol}${S.atom}`), `  (atom ${S.atom})`));
    if (info) el.atomInfo.append(mk('div', { class: 'muted' }, `x ${info.cart[0].toFixed(3)}  y ${info.cart[1].toFixed(3)}  z ${info.cart[2].toFixed(3)} Å`));
    el.atomInfo.append(mk('div', null, `p_z at E_F ${fmt(a.pz_fermi)}    p_z occ ${fmt(a.pz_occ)}`));
    el.atomInfo.hidden = false;
  }

  function renderTable() {
    el.tbody.textContent = '';
    for (const a of S.result.atoms) {
      const tr = mk('tr', { 'data-atom': a.index },
        mk('td', null, mk('span', { class: 'site-tag' }, `${a.symbol}${a.index}`)),
        ...['py_fermi', 'py_occ', 'pz_fermi', 'pz_occ', 'px_fermi', 'px_occ'].map((k) => mk('td', { class: 'num' }, fmt(a[k], 4))));
      tr.addEventListener('click', () => setAtom(a.index));
      el.tbody.append(tr);
    }
    el.table.hidden = false; el.tableEmpty.hidden = true;
  }
  function highlightRow(n, scroll) {
    el.tbody.querySelectorAll('tr.selected').forEach((r) => r.classList.remove('selected'));
    const row = el.tbody.querySelector(`tr[data-atom="${n}"]`);
    if (row) { row.classList.add('selected'); if (scroll) row.scrollIntoView({ block: 'nearest' }); }
  }

  // ------------------------------------------------------------ plot
  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function niceStep(range, target) {
    const raw = range / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
  }

  function enabledChannels() {
    const out = [];
    document.querySelectorAll('[data-orb]').forEach((cb) => {
      if (!cb.checked) return;
      const o = parseInt(cb.dataset.orb, 10);
      if (el.spinUp.checked) out.push(2 * o);
      if (el.spinDn.checked) out.push(2 * o + 1);
    });
    return out;
  }
  const window_ = () => {
    let lo = parseFloat(el.eMin.value), hi = parseFloat(el.eMax.value);
    if (!isFinite(lo)) lo = -10;
    if (!isFinite(hi)) hi = 5;
    if (hi - lo < 0.5) hi = lo + 0.5;
    return [Math.max(-25, lo), Math.min(10, hi)];
  };

  let plotGeom = null;
  function sizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, el.stage.clientWidth), h = Math.max(10, el.stage.clientHeight);
    // CSS size is set explicitly: the bitmap is w*dpr pixels, but it must always be shown at w x h CSS pixels
    // (otherwise, on scaled Windows displays, the plot is drawn dpr times too large and runs out of the window)
    el.canvas.style.width = w + 'px'; el.canvas.style.height = h + 'px';
    if (el.canvas.width !== Math.round(w * dpr) || el.canvas.height !== Math.round(h * dpr)) {
      el.canvas.width = Math.round(w * dpr); el.canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }

  function drawPlot() {
    const c2 = ctx;
    const [W, H] = sizeCanvas();
    const bg = cssVar('--panel', '#fff'), fg = cssVar('--text', '#111'), muted = cssVar('--muted', '#777'), line = cssVar('--line', '#ccc');
    c2.fillStyle = bg;
    c2.fillRect(0, 0, W, H);
    plotGeom = null;
    if (!S.result || !S.cache.has(S.atom)) return;
    const curves = S.cache.get(S.atom);
    const chans = enabledChannels();
    const [e0, e1] = window_();
    const L = 62, R = 16, B = 46;
    // legend layout first (it decides the top margin)
    c2.font = '12px "Segoe UI", system-ui, sans-serif';
    const names = ['s', 'p_y', 'p_z', 'p_x'];
    const legend = [];
    let lrow = 0, lcur = 0;
    for (const k of chans) {
      const label = names[k >> 1] + (k % 2 ? ' down' : ' up');
      const tw = c2.measureText(label).width + 38;
      if (lcur + tw > W - L - R && lcur > 0) { lrow++; lcur = 0; }
      legend.push({ k, label, x: L + lcur, row: lrow });
      lcur += tw;
    }
    const T = 32 + (lrow + 1) * 17;
    const pw = W - L - R, ph = H - T - B;
    if (pw < 40 || ph < 40) return;

    // y range from the visible window
    let ymin = 0, ymax = 0;
    const E = S.energy;
    let i0 = 0, i1 = E.length - 1;
    while (i0 < E.length - 1 && E[i0] < e0) i0++;
    while (i1 > 0 && E[i1] > e1) i1--;
    for (const k of chans) { const c = curves[k]; for (let i = i0; i <= i1; i++) { if (c[i] < ymin) ymin = c[i]; if (c[i] > ymax) ymax = c[i]; } }
    if (ymax - ymin < 1e-6) { ymax = 1e-3; ymin = -1e-3; }
    const pad = (ymax - ymin) * 0.07;
    ymin -= pad; ymax += pad;
    const X = (e) => L + ((e - e0) / (e1 - e0)) * pw;
    const Y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * ph;
    plotGeom = { L, T, pw, ph, e0, e1, ymin, ymax, X, Y };

    c2.font = '12px "Segoe UI", system-ui, sans-serif';
    c2.textBaseline = 'middle';
    // grid + ticks
    const xs = niceStep(e1 - e0, 8), ys = niceStep(ymax - ymin, 6);
    c2.strokeStyle = line; c2.lineWidth = 1; c2.fillStyle = muted;
    c2.textAlign = 'center';
    for (let x = Math.ceil(e0 / xs) * xs; x <= e1 + 1e-9; x += xs) {
      c2.globalAlpha = 0.45; c2.beginPath(); c2.moveTo(X(x), T); c2.lineTo(X(x), T + ph); c2.stroke(); c2.globalAlpha = 1;
      c2.fillText((Math.abs(x) < 1e-9 ? 0 : x).toFixed(xs < 1 ? 1 : 0), X(x), T + ph + 14);
    }
    c2.textAlign = 'right';
    for (let y = Math.ceil(ymin / ys) * ys; y <= ymax + 1e-9; y += ys) {
      c2.globalAlpha = 0.45; c2.beginPath(); c2.moveTo(L, Y(y)); c2.lineTo(L + pw, Y(y)); c2.stroke(); c2.globalAlpha = 1;
      c2.fillText((Math.abs(y) < ys * 1e-6 ? 0 : y).toFixed(ys < 0.01 ? 3 : ys < 0.1 ? 2 : ys < 1 ? 1 : 0), L - 7, Y(y));
    }
    // frame + zero line
    c2.strokeStyle = muted; c2.strokeRect(L, T, pw, ph);
    if (ymin < 0 && ymax > 0) { c2.beginPath(); c2.moveTo(L, Y(0)); c2.lineTo(L + pw, Y(0)); c2.stroke(); }
    // Fermi level
    if (e0 < 0 && e1 > 0) {
      c2.save(); c2.setLineDash([5, 4]); c2.strokeStyle = fg; c2.globalAlpha = 0.7;
      c2.beginPath(); c2.moveTo(X(0), T); c2.lineTo(X(0), T + ph); c2.stroke(); c2.restore();
    }
    // curves
    c2.save();
    c2.beginPath(); c2.rect(L, T, pw, ph); c2.clip();
    c2.lineWidth = 1.6; c2.lineJoin = 'round';
    for (const k of chans) {
      const c = curves[k];
      c2.strokeStyle = S.colors[k] || '#888';
      c2.beginPath();
      for (let i = Math.max(0, i0 - 1); i <= Math.min(E.length - 1, i1 + 1); i++) {
        const px = X(E[i]), py = Y(c[i]);
        if (i === Math.max(0, i0 - 1)) c2.moveTo(px, py); else c2.lineTo(px, py);
      }
      c2.stroke();
    }
    c2.restore();
    // axis labels
    c2.fillStyle = fg; c2.textAlign = 'center';
    c2.fillText('E − E_F (eV)', L + pw / 2, H - 12);
    c2.save(); c2.translate(16, T + ph / 2); c2.rotate(-Math.PI / 2); c2.fillText('PDOS (states/eV)', 0, 0); c2.restore();
    // title + legend
    const a = S.result.atoms[S.atom - 1];
    c2.textAlign = 'left'; c2.font = '600 13px "Segoe UI", system-ui, sans-serif';
    c2.fillText(`Atom ${S.atom} (${a.symbol})`, L, 14);
    c2.font = '12px "Segoe UI", system-ui, sans-serif';
    for (const g of legend) {
      const ly = 32 + g.row * 17 + 6;
      c2.strokeStyle = S.colors[g.k]; c2.lineWidth = 2.2;
      c2.beginPath(); c2.moveTo(g.x, ly); c2.lineTo(g.x + 16, ly); c2.stroke();
      c2.fillStyle = fg; c2.fillText(g.label, g.x + 21, ly);
    }
    // hover crosshair
    if (S.hoverX !== null) {
      const e = e0 + ((S.hoverX - L) / pw) * (e1 - e0);
      if (e >= e0 && e <= e1) {
        c2.save(); c2.strokeStyle = muted; c2.globalAlpha = 0.8;
        c2.beginPath(); c2.moveTo(S.hoverX, T); c2.lineTo(S.hoverX, T + ph); c2.stroke(); c2.restore();
      }
    }
  }

  function drawHoverTip(ev) {
    if (!plotGeom || !S.result) { el.tip.hidden = true; return; }
    const r = el.canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const g = plotGeom;
    if (x < g.L || x > g.L + g.pw || y < g.T || y > g.T + g.ph) { S.hoverX = null; el.tip.hidden = true; drawPlot(); return; }
    S.hoverX = x;
    const e = g.e0 + ((x - g.L) / g.pw) * (g.e1 - g.e0);
    let idx = 0, best = Infinity;
    for (let i = 0; i < S.energy.length; i++) { const d = Math.abs(S.energy[i] - e); if (d < best) { best = d; idx = i; } }
    const c = S.cache.get(S.atom);
    const names = ['s', 'p_y', 'p_z', 'p_x'];
    const lines = [`E = ${S.energy[idx].toFixed(2)} eV`];
    for (const k of enabledChannels()) lines.push(`${names[k >> 1]} ${k % 2 ? 'down' : 'up'}  ${c[k][idx].toFixed(4)}`);
    el.tip.textContent = lines.join('\n');
    el.tip.hidden = false;
    const tx = x + 14 + 140 > el.stage.clientWidth ? x - 150 : x + 14;
    el.tip.style.left = tx + 'px';
    el.tip.style.top = Math.max(4, Math.min(y + 12, el.stage.clientHeight - 20 - lines.length * 15)) + 'px';
    drawPlot();
  }
  el.canvas.addEventListener('mousemove', drawHoverTip);
  el.canvas.addEventListener('mouseleave', () => { S.hoverX = null; el.tip.hidden = true; drawPlot(); });

  document.querySelectorAll('[data-orb]').forEach((cb) => cb.addEventListener('change', () => drawPlot()));
  [el.spinUp, el.spinDn, el.eMin, el.eMax].forEach((n) => n.addEventListener('change', () => drawPlot()));
  el.eFull.addEventListener('click', () => { el.eMin.value = '-25'; el.eMax.value = '10'; drawPlot(); });
  new ResizeObserver(() => drawPlot()).observe(el.stage);

  el.savePlot.addEventListener('click', async () => {
    if (!S.result) return;
    S.hoverX = null;
    drawPlot();
    const a = S.result.atoms[S.atom - 1];
    const res = await api.savePng({ dataUrl: el.canvas.toDataURL('image/png'), defaultName: `PDOS_atom_${S.atom}_${a.symbol}.png` });
    if (res && res.success) log(`Saved plot: ${res.savedPath}`);
    else if (res && !res.canceled) log(`Could not save plot: ${res.error}`);
  });

  // ------------------------------------------------------------ export all atoms (zip)
  el.exportBtn.addEventListener('click', async () => {
    if (!S.result || S.busy) return;
    S.busy = true;
    el.exportBtn.textContent = 'Rendering...';
    el.prog.hidden = false; el.progBar.style.width = '2%';
    updateButtons();
    const [emin, emax] = window_();
    try {
      const res = await api.dosExport({
        format: el.exportFmt.value, dpi: parseInt(el.exportDpi.value, 10), emin, emax,
        orbitals: enabledChannels(), baseName: S.structure ? S.structure.name : 'structure'
      });
      if (res && res.success) log(`Saved ${res.count} plots (${el.exportFmt.value.toUpperCase()}) to ${res.savedPath}  (${(res.bytes / 1048576).toFixed(1)} MB)`);
      else if (res && !res.canceled) { log(`Export failed: ${res.error}`); showDosTab('dosLogView'); }
    } catch (err) {
      log(`Export error: ${err.message}`);
      showDosTab('dosLogView');
    } finally {
      S.busy = false;
      el.exportBtn.textContent = 'Download plots (ZIP)';
      el.prog.hidden = true;
      updateButtons();
    }
  });

  // ------------------------------------------------------------ structure viewer (second instance)
  function initViewer() {
    dosViewer = new window.StructureViewer(el.vCanvas);
    dosViewer.emptyHint = 'Open a structure file (CIF, POSCAR / CONTCAR, xyz, cc1 ...) to see it here';
    dosViewer.invalidate();
    dosViewer.onhover = (n, x, y) => {
      if (!n) { el.vTip.hidden = true; return; }
      const s = dosViewer.struct && dosViewer.struct.symbols ? dosViewer.struct.symbols[n - 1] : '';
      el.vTip.textContent = `${s}${n}`;
      el.vTip.style.left = x + 14 + 'px'; el.vTip.style.top = y + 14 + 'px';
      el.vTip.hidden = false;
    };
    dosViewer.onselect = (n) => { if (n && S.result) setAtom(n); else updateAtomInfo(); };
    document.querySelectorAll('[data-dview]').forEach((b) => b.addEventListener('click', () => dosViewer.resetView(b.dataset.dview)));
    el.optBonds.addEventListener('change', () => dosViewer.setOption('showBonds', el.optBonds.checked));
    el.optCell.addEventListener('change', () => dosViewer.setOption('showCell', el.optCell.checked));
    el.optLabels.addEventListener('change', () => dosViewer.setOption('labels', el.optLabels.value));
    el.optRep.addEventListener('change', () => dosViewer.setOption('rep', el.optRep.value.split(',').map(Number)));
    el.optSize.addEventListener('input', () => dosViewer.setOption('size', parseFloat(el.optSize.value)));
    el.saveStruct.addEventListener('click', async () => {
      if (!dosViewer.struct) return;
      const res = await api.savePng({ dataUrl: dosViewer.toDataURL(2), defaultName: (dosViewer.struct.name || 'structure') + '.png' });
      if (res && res.success) log(`Saved image: ${res.savedPath}`);
    });
  }

  // theme changes redraw the plot and the viewer
  new MutationObserver(() => { drawPlot(); if (dosViewer) dosViewer.refreshTheme(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ------------------------------------------------------------ menu + start
  if (api && api.onMenuAction) {
    api.onMenuAction((action, data) => {
      if (action === 'goto-section') gotoSection(data);
      else if (action === 'open-dos-structure') chooseStructure();
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    initViewer();
  });

  // the page opens on the DOS Predictor tab
  setTimeout(() => gotoSection('dos'), 0);

  window.__dos = { state: S, gotoSection, predict, setAtom, loadStructureFile, get viewer() { return dosViewer; } };
})();
