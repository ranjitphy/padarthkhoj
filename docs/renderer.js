'use strict';
// =========================================================================
// Padarth Khoj - renderer: UI state, engine calls, results table, 3D viewer glue
// =========================================================================
(function () {
  const api = window.electronAPI;
  const $ = (id) => document.getElementById(id);

  const HIGH_ACTIVITY_ETA = 0.90; // same threshold as the first beta

  const state = {
    activeAnalysisType: 'DFT', // 'DFT' | 'Machine Learning'
    uploadedFile: null,
    header: null,
    excelFilePath: null,
    csvFilePath: null,
    outputDir: null,
    isAnalyzing: false,
    records: [],
    summary: null,
    sort: { key: 'eta', dir: 1 },
    selectedAtom: null,
    modelsReady: false
  };

  // ------------------------------------------------------------ element refs
  const el = {
    badge: $('modelStatusBadge'), badgeText: $('modelStatusText'),
    dftCard: $('dftCard'), mlCard: $('mlCard'), selectDftBtn: $('selectDftBtn'), selectMlBtn: $('selectMlBtn'),
    method: $('methodSelect'), reaction: $('reactionSelect'), mlMethod: $('mlMethodSelect'),
    upload: $('uploadBtn'), structBtn: $('structBtn'), fileDetails: $('fileDetails'),
    run: $('runBtn'), runText: $('runBtnText'), cancel: $('cancelBtn'), progress: $('progress'), progressBar: $('progressBar'),
    consoleText: $('consoleText'), table: $('resultsTable'), tbody: $('resultsTableBody'), empty: $('emptyTableState'),
    tableTools: $('tableTools'), carbonOnly: $('carbonOnly'),
    pills: $('summaryPills'), pCount: $('summaryCount'), pBest: $('summaryBest'), pEta: $('summaryEta'),
    dlExcel: $('downloadExcelBtn'), dlCsv: $('downloadCsvBtn'), dlCsvLabel: $('downloadCsvLabel'), openFolder: $('openFolderBtn'),
    themeSwitch: $('themeSwitch'), fsSwitch: $('fullscreenSwitch'),
    canvas: $('viewerCanvas'), structTitle: $('structTitle'), colorEtaBtn: $('colorEtaBtn'),
    legend: $('legend'), legendBar: $('legendBar'), legendMin: $('legendMin'), legendMax: $('legendMax'),
    atomInfo: $('atomInfo'), tip: $('viewerTip'),
    optBonds: $('optBonds'), optCell: $('optCell'), optLabels: $('optLabels'), optRep: $('optRep'), optSize: $('optSize'), savePng: $('savePngBtn')
  };

  const viewer = new window.StructureViewer(el.canvas);

  // ------------------------------------------------------------ helpers
  function mk(tag, props, ...kids) {
    const n = document.createElement(tag);
    if (props) for (const [k, v] of Object.entries(props)) { if (k === 'class') n.className = v; else n.setAttribute(k, v); }
    for (const k of kids) n.append(k instanceof Node ? k : document.createTextNode(String(k)));
    return n;
  }
  const baseName = (p) => String(p || '').split(/[\\/]/).pop();
  const log = (text, replace) => { el.consoleText.textContent = replace ? text : el.consoleText.textContent + (el.consoleText.textContent ? '\n' : '') + text; };
  const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? '-' : Number(v).toFixed(d));

  function showTab(id) {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === id));
    document.querySelectorAll('.tab-pane').forEach((p) => p.classList.toggle('active', p.id === id));
  }

  function setBadge(kind, text, title) {
    el.badge.className = 'status-badge status-' + kind;
    el.badgeText.textContent = text;
    el.badge.title = title || '';
  }

  // ------------------------------------------------------------ theme + full screen
  function applyTheme(theme, persist) {
    document.documentElement.setAttribute('data-theme', theme);
    el.themeSwitch.checked = theme === 'light';
    if (persist) { try { localStorage.setItem('pk-theme', theme); } catch (e) { /* ignore */ } }
    if (api && api.setTheme) api.setTheme(theme);
    viewer.refreshTheme();
    refreshLegend();
  }
  function initTheme() {
    let t = 'dark';
    try { t = localStorage.getItem('pk-theme') || 'dark'; } catch (e) { /* ignore */ }
    applyTheme(t === 'light' ? 'light' : 'dark', false);
  }
  el.themeSwitch.addEventListener('change', () => applyTheme(el.themeSwitch.checked ? 'light' : 'dark', true));

  el.fsSwitch.addEventListener('change', () => { if (api) api.setFullScreen(el.fsSwitch.checked); });
  if (api && api.onFullScreenChange) api.onFullScreenChange((on) => { el.fsSwitch.checked = !!on; });
  if (api && api.isFullScreen) api.isFullScreen().then((on) => { el.fsSwitch.checked = !!on; }).catch(() => {});

  // ------------------------------------------------------------ DFT / ML mode cards
  function updateUiState() {
    const dft = state.activeAnalysisType === 'DFT';
    el.dftCard.classList.toggle('active', dft);
    el.mlCard.classList.toggle('active', !dft);
    el.method.disabled = !dft;
    el.mlMethod.disabled = dft ? false : false; // the ML regressor is used by both modes (as in the notebook)
    el.selectDftBtn.textContent = dft ? 'Active' : 'Activate DFT';
    el.selectMlBtn.textContent = dft ? 'Activate ML' : 'Active';
  }
  function setActiveAnalysis(type) { state.activeAnalysisType = type; updateUiState(); }
  el.selectDftBtn.addEventListener('click', () => setActiveAnalysis('DFT'));
  el.selectMlBtn.addEventListener('click', () => setActiveAnalysis('Machine Learning'));
  el.method.addEventListener('change', () => setActiveAnalysis('DFT'));
  el.mlMethod.addEventListener('change', () => { setActiveAnalysis('Machine Learning'); updateDownloadLabels(); });
  el.reaction.addEventListener('change', updateDownloadLabels);

  function updateDownloadLabels() {
    el.dlCsvLabel.textContent = state.csvFilePath ? baseName(state.csvFilePath) : `overpotentials${el.reaction.value}_${el.mlMethod.value}.csv`;
  }

  function resetResults() {
    state.excelFilePath = state.csvFilePath = state.outputDir = null;
    state.records = []; state.summary = null; state.selectedAtom = null;
    el.dlExcel.disabled = el.dlCsv.disabled = el.openFolder.disabled = true;
    el.pills.hidden = true; el.tableTools.hidden = true;
    el.table.hidden = true; el.empty.hidden = false;
    el.tbody.textContent = '';
    viewer.setResults(new Map(), null);
    viewer.setColorMode('element');
    setColorButtons('element');
    el.colorEtaBtn.disabled = true;
    refreshLegend();
    updateDownloadLabels();
  }

  // ------------------------------------------------------------ structure viewer glue
  function setColorButtons(mode) {
    document.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.color === mode));
  }
  document.querySelectorAll('.seg-btn').forEach((b) => b.addEventListener('click', () => {
    if (b.disabled) return;
    viewer.setColorMode(b.dataset.color);
    setColorButtons(b.dataset.color);
    refreshLegend();
  }));
  document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => viewer.resetView(b.dataset.view)));
  el.optBonds.addEventListener('change', () => viewer.setOption('showBonds', el.optBonds.checked));
  el.optCell.addEventListener('change', () => viewer.setOption('showCell', el.optCell.checked));
  el.optLabels.addEventListener('change', () => viewer.setOption('labels', el.optLabels.value));
  el.optRep.addEventListener('change', () => viewer.setOption('rep', el.optRep.value.split(',').map(Number)));
  el.optSize.addEventListener('input', () => viewer.setOption('size', parseFloat(el.optSize.value)));
  el.savePng.addEventListener('click', async () => {
    if (!viewer.struct) return;
    const res = await api.savePng({ dataUrl: viewer.toDataURL(2), defaultName: (viewer.struct.name || 'structure') + '.png' });
    if (res && res.success) log(`Saved image: ${res.savedPath}`);
    else if (res && !res.canceled) log(`Could not save image: ${res.error}`);
  });

  function refreshLegend() {
    const show = viewer.colorMode === 'eta' && state.records.length > 0;
    el.legend.hidden = !show;
    if (!show) return;
    const sc = viewer.getColorScale();
    el.legendBar.style.background = 'linear-gradient(90deg,' + sc.stops.map(([t, c]) => `${c} ${Math.round(t * 100)}%`).join(',') + ')';
    el.legendMin.textContent = sc.min.toFixed(2);
    el.legendMax.textContent = sc.max.toFixed(2);
  }

  function atomLabel(n) {
    const s = viewer.struct && viewer.struct.symbols ? viewer.struct.symbols[n - 1] : null;
    return s ? `${s}${n}` : `Atom ${n}`;
  }

  viewer.onhover = (n, x, y) => {
    if (!n) { el.tip.hidden = true; return; }
    const info = viewer.atomInfo(n);
    const r = info && info.result;
    el.tip.textContent = atomLabel(n) + (r && r.eta !== null && r.eta !== undefined ? `   η ${r.eta.toFixed(3)} V` : '');
    el.tip.style.left = x + 14 + 'px';
    el.tip.style.top = y + 14 + 'px';
    el.tip.hidden = false;
  };

  viewer.onselect = (n) => {
    state.selectedAtom = n;
    updateAtomInfo();
    highlightRow(n, true);
  };

  function updateAtomInfo() {
    const n = state.selectedAtom;
    el.atomInfo.textContent = '';
    if (!n) { el.atomInfo.hidden = true; return; }
    const info = viewer.atomInfo(n);
    if (!info) { el.atomInfo.hidden = true; return; }
    el.atomInfo.append(mk('div', null, mk('b', null, atomLabel(n)), `  (atom ${n} in the DOSCAR)`));
    el.atomInfo.append(mk('div', { class: 'muted' }, `x ${info.cart[0].toFixed(3)}  y ${info.cart[1].toFixed(3)}  z ${info.cart[2].toFixed(3)} Å`));
    if (info.result) {
      el.atomInfo.append(mk('div', null, `ΔG ${fmt(info.result.dg)} eV    η ${fmt(info.result.eta)} V`));
      if (info.result.partner) el.atomInfo.append(mk('div', { class: 'muted' }, `Dual site with ${atomLabel(info.result.partner)} (dashed ring)`));
    }
    el.atomInfo.hidden = false;
  }

  async function loadStructure(args, label) {
    try {
      const s = await api.parseStructure(args);
      if (!s || !s.success) {
        viewer.clear();
        el.structTitle.textContent = 'Structure';
        if (s && s.error) log(`Structure: ${s.error}`);
        return null;
      }
      viewer.setStructure(s);
      el.structTitle.textContent = `${s.formula}  (${s.name})`;
      if (label) log(label);
      return s;
    } catch (err) {
      log(`Structure error: ${err.message}`);
      return null;
    }
  }

  // ------------------------------------------------------------ upload
  async function chooseDoscar() {
    if (state.isAnalyzing) return;
    try {
      const fileInfo = await api.selectDoscar();
      if (!fileInfo) return;
      state.uploadedFile = fileInfo;
      state.header = null;
      resetResults();
      el.run.disabled = false;
      const sub = mk('span', { class: 'file-subtext' }, fileInfo.directory);
      el.fileDetails.replaceChildren(mk('span', { class: 'file-label' }, `${fileInfo.name}  (${(fileInfo.sizeBytes / 1048576).toFixed(2)} MB)`), sub);
      log(`File selected: ${fileInfo.path}`, true);

      let header = null;
      try { header = await api.parseVaspHeader(fileInfo.path); } catch (e) { header = { success: false, error: e.message }; }
      if (header && header.success) {
        state.header = header;
        const parts = [`${header.natoms} atoms`, `NEDOS ${header.nedos}`, `Eᴺ ${header.efermi.toFixed(4)} eV (${header.efermi_source})`, `ISPIN ${header.nspin}`];
        if (header.structure) parts.unshift(header.structure.formula);
        sub.textContent = parts.join('  ·  ');
        if (header.warnings && header.warnings.length) {
          sub.classList.add('warn');
          sub.title = header.warnings.join('\n');
          log('Notes:\n  - ' + header.warnings.join('\n  - '));
        }
        if (header.structure_path) await loadStructure({ path: header.structure_path });
        else { viewer.clear(); el.structTitle.textContent = 'Structure'; }
      } else {
        sub.textContent = (header && header.error) || 'Could not read the DOSCAR header.';
        sub.classList.add('warn');
        log(sub.textContent);
      }
    } catch (err) {
      log(`File selection error: ${err.message}`);
    }
  }
  el.upload.addEventListener('click', chooseDoscar);

  async function chooseStructure() {
    try {
      const f = await api.selectStructure({ target: 'activity' });
      if (!f) return;
      resetResults();
      const s = await loadStructure({ path: f.path }, `Structure loaded: ${f.path}`);
      if (s) log(`${s.natoms} atoms, ${s.formula}${s.periodic ? '' : ' (non-periodic)'}`);
    } catch (err) {
      log(`File selection error: ${err.message}`);
    }
  }
  el.structBtn.addEventListener('click', chooseStructure);

  // ------------------------------------------------------------ run analysis
  function setAnalyzing(on) {
    state.isAnalyzing = on;
    el.run.disabled = on || !state.uploadedFile;
    el.upload.disabled = on;
    el.structBtn.disabled = on;
    el.run.classList.toggle('loading', on);
    el.runText.textContent = on ? 'Analyzing...' : 'Run analysis';
    el.cancel.hidden = !on;
    el.progress.hidden = !on;
    el.progressBar.style.width = '0';
  }

  if (api && api.onProgress) {
    api.onProgress((p) => {
      if (!state.isAnalyzing) return;
      if (typeof p.percent === 'number') {
        el.progressBar.style.width = Math.max(2, Math.min(100, p.percent)) + '%';
        el.runText.textContent = `Analyzing ${Math.round(p.percent)}%`;
      }
    });
  }

  el.cancel.addEventListener('click', async () => {
    await api.cancelJob();
    log('Analysis cancelled. The engine restarts and re-trains its models on the next run.');
  });

  async function runAnalysis() {
    if (!state.uploadedFile || state.isAnalyzing) return;
    if (state.activeAnalysisType === 'DFT' && el.method.value === 'Quantum Espresso') {
      showTab('consoleView');
      log('Quantum Espresso analysis is not yet implemented. Please select VASP or switch to ML analysis.', true);
      return;
    }
    setAnalyzing(true);
    const reaction = el.reaction.value, mlMethod = el.mlMethod.value;
    log(`Running ${reaction} analysis with ${mlMethod}...\nSplitting DOSCAR and computing p_z descriptors.`, true);

    try {
      const res = await api.runAnalysis({
        doscarPath: state.uploadedFile.path, activeType: state.activeAnalysisType,
        method: el.method.value, reaction, mlMethod
      });
      if (!res || !res.success) {
        showTab('consoleView');
        log(`Analysis failed: ${res ? (res.error || 'unknown error') : 'no response from the engine'}`, true);
        return;
      }
      state.excelFilePath = res.excel_path;
      state.csvFilePath = res.csv_path;
      state.outputDir = res.output_dir;
      state.records = res.records || [];
      state.summary = res.summary;

      let text = res.output_str || 'Analysis complete.';
      if (res.notes && res.notes.length) text = 'Notes:\n  - ' + res.notes.join('\n  - ') + '\n\n' + text;
      text += `\n\nFiles written to ${res.output_dir}\n  ${baseName(res.excel_path)}\n  ${baseName(res.csv_path)}  (${res.summary.elapsed_s}s)`;
      log(text, true);

      // viewer: make sure the structure of this folder is shown, then colour by overpotential
      if (res.structure_path && (!viewer.struct || viewer.struct.path !== res.structure_path)) {
        await loadStructure({ path: res.structure_path });
      }
      renderTable();
      el.dlExcel.disabled = !state.excelFilePath;
      el.dlCsv.disabled = !state.csvFilePath;
      el.openFolder.disabled = !state.outputDir;
      updateDownloadLabels();
      showTab('tableView');
      el.colorEtaBtn.disabled = false;
      el.colorEtaBtn.title = '';
      viewer.setColorMode('eta');
      setColorButtons('eta');
      refreshLegend();
    } catch (err) {
      if (String(err.message || err).includes('CANCELLED')) {
        log('Analysis cancelled.', true);
      } else {
        showTab('consoleView');
        log(`Execution error: ${err.message || err}`, true);
      }
    } finally {
      setAnalyzing(false);
    }
  }
  el.run.addEventListener('click', runAnalysis);

  // ------------------------------------------------------------ results table
  function visibleRecords() {
    const hasEl = state.records.some((r) => r.Element);
    if (hasEl && el.carbonOnly.checked) return state.records.filter((r) => r.Element === 'C');
    return state.records;
  }

  function recDg(r) { return r.Max_DelGoh_minus_DelGo !== undefined ? r.Max_DelGoh_minus_DelGo : r.Del_Goh; }

  function renderTable() {
    const all = state.records;
    if (!all.length) { el.table.hidden = true; el.empty.hidden = false; el.pills.hidden = true; return; }
    const hasEl = all.some((r) => r.Element);
    el.tableTools.hidden = !(hasEl && all.some((r) => r.Element !== 'C'));

    // viewer data covers every analysed atom
    const map = new Map();
    for (const r of all) map.set(r.Atom_Number, { eta: r.Overpotential, dg: recDg(r), partner: r.Best_Partner || null });
    const rows = visibleRecords();
    const best = rows.reduce((b, r) => (b === null || r.Overpotential < b.Overpotential ? r : b), null);
    viewer.setResults(map, best ? best.Atom_Number : null);

    const key = state.sort.key, dir = state.sort.dir;
    const val = { atom: (r) => r.Atom_Number, dg: recDg, eta: (r) => r.Overpotential };
    const sorted = [...rows].sort((a, b) => (val[key](a) - val[key](b)) * dir);

    el.tbody.textContent = '';
    for (const r of sorted) {
      const n = r.Atom_Number;
      const isBest = best && n === best.Atom_Number;
      const tr = mk('tr', { 'data-atom': n });
      if (isBest) tr.classList.add('best-site-row');
      const sym = r.Element || (viewer.struct && viewer.struct.symbols ? viewer.struct.symbols[n - 1] : '');
      const site = mk('span', { class: 'site-tag' }, `${sym}${n}`);
      const cell0 = mk('td', null, site);
      if (r.Best_Partner) cell0.append(mk('span', { class: 'site-sub' }, `+ ${atomLabel(r.Best_Partner)}`));
      const badge = isBest ? mk('span', { class: 'badge best' }, 'Best site')
        : r.Overpotential < HIGH_ACTIVITY_ETA ? mk('span', { class: 'badge high' }, 'High')
          : mk('span', { class: 'badge' }, 'Moderate');
      tr.append(cell0, mk('td', { class: 'num' }, fmt(recDg(r))), mk('td', { class: 'num' }, fmt(r.Overpotential)), mk('td', null, badge));
      tr.addEventListener('click', () => { viewer.select(n === state.selectedAtom ? null : n); });
      el.tbody.append(tr);
    }

    el.empty.hidden = true;
    el.table.hidden = false;
    el.pCount.textContent = String(rows.length);
    el.pBest.textContent = best ? atomLabel(best.Atom_Number) : '-';
    el.pEta.textContent = best ? `${best.Overpotential.toFixed(4)} V` : '-';
    el.pills.hidden = false;
    highlightRow(state.selectedAtom, false);
  }

  function highlightRow(n, scroll) {
    el.tbody.querySelectorAll('tr.selected').forEach((r) => r.classList.remove('selected'));
    if (!n) return;
    const row = el.tbody.querySelector(`tr[data-atom="${n}"]`);
    if (row) {
      row.classList.add('selected');
      if (scroll) row.scrollIntoView({ block: 'nearest' });
    }
  }

  document.querySelectorAll('th.sortable').forEach((th) => th.addEventListener('click', () => {
    const k = th.dataset.sort;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : 1 };
    document.querySelectorAll('th.sortable').forEach((t) => { t.classList.remove('sorted', 'desc'); });
    th.classList.add('sorted');
    th.classList.toggle('desc', state.sort.dir < 0);
    renderTable();
  }));
  el.carbonOnly.addEventListener('change', renderTable);

  document.querySelectorAll('.tab-btn').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

  // ------------------------------------------------------------ exports
  el.dlExcel.addEventListener('click', async () => {
    if (!state.excelFilePath) return;
    const res = await api.saveFile({ sourcePath: state.excelFilePath, defaultName: 'dos_results.xlsx', extension: 'xlsx' });
    if (res.success) log(`Excel file exported to ${res.savedPath}`);
    else if (!res.canceled) log(`Download error: ${res.error}`);
  });
  el.dlCsv.addEventListener('click', async () => {
    if (!state.csvFilePath) return;
    const res = await api.saveFile({ sourcePath: state.csvFilePath, defaultName: baseName(state.csvFilePath), extension: 'csv' });
    if (res.success) log(`CSV file exported to ${res.savedPath}`);
    else if (!res.canceled) log(`Download error: ${res.error}`);
  });
  el.openFolder.addEventListener('click', () => { if (state.csvFilePath) api.showInFolder(state.csvFilePath); });

  // ------------------------------------------------------------ native menu
  if (api && api.onMenuAction) {
    api.onMenuAction((action) => {
      if (action === 'open-doscar') chooseDoscar();
      else if (action === 'open-structure') chooseStructure();
      else if (action === 'run-analysis') runAnalysis();
      else if (action === 'save-image') el.savePng.click();
      else if (action === 'toggle-theme') { el.themeSwitch.checked = !el.themeSwitch.checked; el.themeSwitch.dispatchEvent(new Event('change')); }
      else if (action === 'reset-view') viewer.resetView('iso');
    });
  }

  // ------------------------------------------------------------ start-up
  window.addEventListener('DOMContentLoaded', async () => {
    initTheme();
    updateUiState();
    updateDownloadLabels();
    log('Training machine learning models...', true);
    try {
      const res = await api.initModels();
      if (res && res.success) {
        state.modelsReady = true;
        const t = res.training || {};
        const tip = Object.keys(t).sort().map((k) => `${k.replace('_', ' ')}: n=${t[k].n_samples}, training R² ${t[k].r2_train.toFixed(3)}`).join('\n');
        setBadge('ready', 'Models ready: OER & ORR, SVR / MLR', tip);
        log('Model training complete. Application is ready.\n' + tip + '\n\nUpload a DOSCAR file to begin.', true);
      } else {
        setBadge('error', 'Engine not available', res && res.error);
        showTab('consoleView');
        log(`Engine problem:\n${res ? res.error : 'unknown'}`, true);
      }
    } catch (err) {
      setBadge('error', 'Engine error', err.message);
      showTab('consoleView');
      log(`Engine error: ${err.message}`, true);
    }
  });

  // test hook (used by the automated UI check only)
  window.__pk = { state, viewer, runAnalysis, chooseDoscar };
})();
