'use strict';
// =========================================================================
// Padarth Khoj - Structure Relaxor (vasp-mace + MACE). POSCAR + INCAR (+ KPOINTS) -> CONTCAR
// =========================================================================
(function () {
  const api = window.electronAPI;
  const $ = (id) => document.getElementById(id);
  const baseName = (p) => String(p || '').split(/[\\/]/).pop();
  const fmt = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? '-' : Number(v).toFixed(d));

  const BIBTEX = `@software{grau_crespo_vasp_mace,
  author    = {Grau-Crespo, Ricardo},
  title     = {vasp-mace: a VASP-style workflow interface for MACE machine-learning interatomic potentials},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19479246},
  url       = {https://github.com/rgraucrespo/vasp-mace}
}`;

  const S = {
    poscar: null, incar: null, kpoints: null,   // {path,name,directory,text?}
    modelKey: 'mace-mh-1', modelPath: null, customModel: false, customName: '',
    status: null, busy: false,
    result: null, which: 'initial',
    initialStruct: null
  };
  const el = {
    poscarName: $('rxPoscarName'), incarName: $('rxIncarName'), kpointsName: $('rxKpointsName'),
    modelSel: $('rxModelSel'), head: $('rxHead'), headField: $('rxHeadField'), headList: $('rxHeadList'), headInfo: $('rxHeadInfo'), license: $('rxLicense'), dtype: $('rxDtype'), python: $('rxPython'), pyBrowse: $('rxPyBrowse'), checkEnv: $('rxCheckEnv'), pyRow: $('rxPyRow'), phase: $('rxPhase'), modelState: $('rxModelState'), dl: $('rxDownloadBtn'), custom: $('rxCustomModelBtn'),
    engineNote: $('rxEngineNote'), device: $('rxDevice'), opt: $('rxOptimizer'),
    run: $('rxRunBtn'), runText: $('rxRunText'), cancel: $('rxCancelBtn'), prog: $('rxProgress'), progBar: $('rxProgressBar'),
    incarText: $('rxIncarText'), incarCheck: $('rxIncarCheck'), fix: $('rxFixBtn'),
    summary: $('rxSummary'), table: $('rxTable'), tbody: $('rxTableBody'), empty: $('rxResultEmpty'), log: $('rxLog'),
    pills: $('rxPills'), pStatus: $('rxPillStatus'), pSteps: $('rxPillSteps'), pDE: $('rxPillDE'),
    saveC: $('rxSaveContcar'), saveZip: $('rxSaveZip'), toDos: $('rxToDos'), openF: $('rxOpenFolder'),
    showI: $('rxShowInitial'), showF: $('rxShowFinal'), title: $('rxStructTitle'), vCanvas: $('rxViewerCanvas'), vTip: $('rxViewerTip'),
    optBonds: $('rxOptBonds'), optCell: $('rxOptCell'), optLabels: $('rxOptLabels'), optRep: $('rxOptRep'), optSize: $('rxOptSize'),
    saveStruct: $('rxSaveStruct'), dlRelaxed: $('rxDownloadRelaxed')
  };
  let viewer = null;
  const log = (t, replace) => { el.log.textContent = replace ? t : el.log.textContent + (el.log.textContent ? '\n' : '') + t; el.log.parentElement.scrollTop = 1e9; };

  // ------------------------------------------------------------ tabs
  function showTab(id) {
    document.querySelectorAll('[data-rxtab]').forEach((b) => b.classList.toggle('active', b.dataset.rxtab === id));
    ['rxIncarView', 'rxResultView', 'rxLogView'].forEach((t) => $(t).classList.toggle('active', t === id));
  }
  document.querySelectorAll('[data-rxtab]').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.rxtab)));

  // ------------------------------------------------------------ INCAR parsing + checks
  function parseIncar(text) {
    const tags = {};
    for (const raw of String(text || '').split(/\r?\n/)) {
      const line = raw.replace(/[#!].*$/, '');
      for (const part of line.split(';')) {
        const m = part.match(/^\s*([A-Za-z_0-9]+)\s*=\s*(.*?)\s*$/);
        if (m) tags[m[1].toUpperCase()] = m[2];
      }
    }
    return tags;
  }
  function checkIncar() {
    const text = el.incarText.value;
    const out = [];
    let fixable = false, blocking = false;
    if (!text.trim()) { el.incarCheck.hidden = true; el.fix.hidden = true; return { blocking: false, empty: true }; }
    const t = parseIncar(text);
    const num = (k) => (t[k] === undefined ? null : parseFloat(t[k]));
    const nsw = num('NSW'), ibr = num('IBRION'), isif = num('ISIF'), ediffg = num('EDIFFG'), images = num('IMAGES');
    if (images && images > 0) { out.push(['bad', 'IMAGES > 0 selects NEB, which is not supported here. Remove IMAGES.']); blocking = true; }
    if (ibr === 0) { out.push(['bad', 'IBRION = 0 is molecular dynamics, not a relaxation.']); blocking = fixable = true; }
    if (ibr === 5 || ibr === 6 || ibr === 7 || ibr === 8) { out.push(['bad', `IBRION = ${ibr} is a phonon calculation, not a relaxation.`]); blocking = fixable = true; }
    if (nsw === null || nsw === 0) { out.push(['bad', 'NSW is missing or 0 (single point). Set NSW to the maximum number of ionic steps, e.g. 200.']); blocking = fixable = true; }
    if (!blocking) {
      const mode = (ibr === null || ibr === -1) ? `optimizer ${el.opt.value}` : 'conjugate-gradient style relaxation';
      const cell = isif === null || isif === 2 ? 'ions only (cell fixed)' : isif === 3 ? 'ions + cell shape + volume' : isif === 4 ? 'ions + cell shape (volume fixed)' : isif === 7 ? 'cell volume only' : isif === 6 ? 'cell shape + volume (ions fixed)' : `ISIF = ${isif}`;
      const crit = ediffg === null ? 'default force criterion' : ediffg < 0 ? `forces < ${Math.abs(ediffg)} eV/Å` : `energy change < ${ediffg} eV`;
      out.push(['good', `Relaxation: ${nsw} steps max, ${mode}, ${cell}, stop at ${crit}.`]);
    }
    if (isif !== null && isif >= 3 && isif !== 2) out.push(['warn', 'Cell is allowed to change: the relaxed lattice is computed with the MACE stress.']);
    el.incarCheck.replaceChildren(...out.map(([c, m]) => { const d = document.createElement('div'); d.className = c; d.textContent = m; return d; }));
    el.incarCheck.hidden = !out.length;
    el.fix.hidden = !fixable;
    return { blocking };
  }
  el.incarText.addEventListener('input', () => { checkIncar(); updateButtons(); });
  el.opt.addEventListener('change', checkIncar);
  el.fix.addEventListener('click', () => {
    const keep = el.incarText.value.split(/\r?\n/).filter((l) => !/^\s*(NSW|IBRION)\s*=/i.test(l));
    keep.push('NSW = 200', 'IBRION = 2');
    el.incarText.value = keep.join('\n').replace(/^\n+/, '') + '\n';
    checkIncar(); updateButtons();
  });

  // ------------------------------------------------------------ inputs
  function setName(node, f, extra) {
    if (!f) { node.textContent = 'not selected'; node.className = 'rx-file-name'; node.title = ''; return; }
    node.textContent = f.name + (extra ? `  ·  ${extra}` : '');
    node.title = f.path;
    node.className = 'rx-file-name ok';
  }
  async function takePoscar(f) {
    S.poscar = null; S.initialStruct = null;
    setName(el.poscarName, null);
    const s = await api.parseStructure({ path: f.path });
    if (!s || !s.success) {
      el.poscarName.textContent = `${f.name}: ${(s && s.error) || 'cannot read this structure'}`;
      el.poscarName.className = 'rx-file-name warn';
      log(`POSCAR: ${(s && s.error) || 'cannot read'}`);
      updateButtons();
      return;
    }
    S.poscar = f; S.initialStruct = s;
    setName(el.poscarName, f, `${s.formula}, ${s.natoms} atoms`);
    clearResult();
    S.which = 'initial';
    showStruct('initial');
    updateButtons();
  }
  function takeIncar(f) {
    if (f.error) { el.incarName.textContent = `${f.name}: ${f.error}`; el.incarName.className = 'rx-file-name warn'; return; }
    S.incar = f;
    el.incarText.value = f.text || '';
    setName(el.incarName, f);
    checkIncar(); updateButtons();
    showTab('rxIncarView');
  }
  function takeKpoints(f) { S.kpoints = f; setName(el.kpointsName, f, 'copied, not used by MACE'); }

  async function ingest(list) {
    if (!list || !list.length) return;
    const notes = [];
    for (const f of list) {
      if (f.kind === 'poscar') await takePoscar(f);
      else if (f.kind === 'incar') takeIncar(f);
      else if (f.kind === 'kpoints') takeKpoints(f);
      else notes.push(f.name);
    }
    if (notes.length) log(`Not recognised as POSCAR/INCAR/KPOINTS (ignored): ${notes.join(', ')}`);
  }
  async function pick(kind) {
    if (S.busy) return;
    try {
      const list = await api.selectRelaxInputs(kind);
      if (list && list.length && kind !== 'all') list.forEach((f) => { f.kind = kind; });
      await ingest(list);
    } catch (err) { log(`File selection error: ${err.message}`); }
  }
  $('rxPoscarBtn').addEventListener('click', () => pick('poscar'));
  $('rxIncarBtn').addEventListener('click', () => pick('incar'));
  $('rxKpointsBtn').addEventListener('click', () => pick('kpoints'));
  $('rxAllBtn').addEventListener('click', () => pick('all'));

  // ------------------------------------------------------------ model
  async function loadStatus(force) {
    if (!api || !api.relaxStatus) return;
    if (S.status && !force) return;
    try {
      const st = await api.relaxStatus();
      S.status = st;
      if (!st || !st.success) throw new Error((st && st.error) || 'engine not reachable');
      if (!st.available) {
        el.engineNote.textContent = 'vasp-mace not installed';
        el.engineNote.style.color = 'var(--bad)';
        log(`Missing Python packages: ${(st.missing || []).join(', ')}. Run "npm run setup:python" and restart the app.`);
      } else { el.engineNote.textContent = 'vasp-mace ready'; el.engineNote.style.color = 'var(--good)'; }
      if (!S.customModel) {
        if (!st.models || !st.models[S.modelKey]) S.modelKey = st.default_model || 'mace-mh-1';
        S.modelPath = st.models && st.models[S.modelKey] && st.models[S.modelKey].present ? st.models[S.modelKey].path : null;
      }
      renderModel();
    } catch (err) { S.status = null; log(`Could not query the relaxor: ${err.message}`); }
    updateButtons();
  }
  const curInfo = () => (S.status && S.status.models && !S.customModel ? S.status.models[S.modelKey] : null);

  function fillModelSelect() {
    const models = (S.status && S.status.models) || {};
    const opts = Object.values(models).map((m) => {
      const o = document.createElement('option');
      o.value = m.key; o.textContent = m.label + (m.present ? '' : '  -  not downloaded');
      return o;
    });
    if (S.customModel) {
      const o = document.createElement('option');
      o.value = '__custom'; o.textContent = `Custom: ${S.customName}`;
      opts.push(o);
    }
    el.modelSel.replaceChildren(...opts);
    el.modelSel.value = S.customModel ? '__custom' : S.modelKey;
  }

  function renderHead() {
    const m = curInfo();
    const heads = m && m.heads ? m.heads : null;
    el.headList.replaceChildren(...(heads || []).map((h) => { const o = document.createElement('option'); o.value = h.name; o.label = h.info; return o; }));
    // single-head stock models take no head; multi-head and custom models show the field
    el.headField.hidden = !!(m && !m.heads);
    if (m && m.heads && !el.head.value.trim()) el.head.value = m.default_head || heads[0].name;
    if (S.customModel && !el.head.value.trim()) el.head.placeholder = 'head name (multi-head models only)';
    const cur = heads && heads.find((h) => h.name === el.head.value.trim());
    let txt = cur ? cur.info : (S.customModel ? 'Custom model: type the head name if it is a multi-head model, leave empty otherwise. A wrong name stops the run and lists the heads the file contains.' : '');
    if (cur && !/PBE/i.test(cur.info)) txt += '. Note: this head is not PBE-based, so compare with PBE/VASP results with care (the optional D3 correction of vasp-mace is parametrised for PBE).';
    el.headInfo.textContent = txt;
    el.headInfo.hidden = !txt;
  }
  el.head.addEventListener('input', renderHead);

  function renderModel() {
    const st = S.status || {};
    fillModelSelect();
    const m = curInfo();
    if (S.customModel) {
      el.modelState.textContent = 'custom model file';
      el.dl.hidden = true; el.license.textContent = ''; el.license.hidden = true;
    } else if (m) {
      el.modelState.textContent = m.present ? `ready (${m.size_mb} MB)` : 'not downloaded yet';
      el.dl.hidden = m.present || st.available === false;
      el.dl.textContent = `Download (${m.expected_mb} MB)`;
      el.license.textContent = /ASL/.test(m.license) ? `${m.label.split(' (')[0]} is released under the ${m.license}: check its terms before commercial use.` : '';
      el.license.hidden = !el.license.textContent;
    }
    renderHead();
  }
  el.modelSel.addEventListener('change', () => {
    if (S.busy) { renderModel(); return; }
    if (el.modelSel.value === '__custom') return;
    S.customModel = false; S.modelKey = el.modelSel.value;
    const m = curInfo();
    S.modelPath = m && m.present ? m.path : null;
    renderModel(); updateButtons();
  });
  el.dl.addEventListener('click', async () => {
    if (S.busy) return;
    const m = curInfo(); if (!m) return;
    S.busy = true; updateButtons();
    el.prog.hidden = false; el.progBar.style.width = '2%';
    el.dl.textContent = 'Downloading...';
    showTab('rxLogView'); log(`Downloading ${m.label.split(' (')[0]} (${m.expected_mb} MB) from the ACEsuit mace-foundations release...`);
    try {
      const r = await api.relaxDownloadModel({ model: m.key });
      if (!r || !r.success) throw new Error((r && r.error) || 'download failed');
      log('Model downloaded.');
      await loadStatus(true);
    } catch (err) { log(`Download failed: ${err.message}. Check the internet connection, or choose a .model file you already have.`); }
    finally { S.busy = false; el.prog.hidden = true; renderModel(); updateButtons(); }
  });
  el.custom.addEventListener('click', async () => {
    if (S.busy) return;
    const list = await api.selectRelaxInputs('model');
    if (!list || !list.length) return;
    S.customModel = true; el.head.value = ''; S.modelPath = list[0].path; S.customName = baseName(list[0].name || list[0].path); renderModel(); updateButtons();
  });


  // ------------------------------------------------------------ python environment (advanced)
  try { el.python.value = localStorage.getItem('pk-relax-python') || ''; if (el.python.value) el.pyRow.open = true; } catch (e) { /* ignore */ }
  el.python.addEventListener('change', () => { try { localStorage.setItem('pk-relax-python', el.python.value.trim()); } catch (e) { /* ignore */ } });
  el.pyBrowse.addEventListener('click', async () => {
    if (S.busy) return;
    const list = await api.selectRelaxInputs('python');
    if (!list || !list.length) return;
    el.python.value = list[0].path; el.python.dispatchEvent(new Event('change'));
  });
  el.checkEnv.addEventListener('click', async () => {
    if (S.busy || !api.relaxCheckEnv) return;
    S.busy = true; updateButtons();
    const py = el.python.value.trim();
    el.cancel.hidden = false; el.prog.hidden = false; el.progBar.style.width = '5%';
    showTab('rxLogView');
    log(`Checking the Python environment (${py || "the app's own Python"}). Each package is imported in turn; this stops after 4 minutes.`, true);
    try {
      const r = await api.relaxCheckEnv({ pythonPath: py || null });
      if (!r || !r.success) throw new Error((r && r.error) || 'check failed');
      const lines = [`Python ${r.info}`];
      for (const s of r.steps) lines.push(s.ok ? `  ok    ${s.name} ${s.version}  (${s.seconds} s)` : `  FAIL  ${s.name}: ${s.error}`);
      if (r.timed_out) {
        lines.push('', `STUCK: the import of "${r.stuck_on}" did not finish in ${r.total_s} s and was stopped.`, 'Python stack traces printed while it waited:', ...r.log.filter((l) => !/^(start|ok|fail|done|python) /.test(l)).slice(-30));
        lines.push('', 'Try: set the Python of an environment where "vasp-mace" works in your terminal (field above), or exclude the app\'s folder (engine\\.venv) from antivirus real-time scanning.');
      } else if (r.ok) lines.push('', `All packages import fine (${r.total_s} s in total).`);
      else lines.push('', 'Some packages are missing or broken: run "npm run setup:python" (app Python), or install them in the environment you entered: pip install vasp-mace.');
      log(lines.join('\n'));
    } catch (err) {
      log(/cancel/i.test(String(err.message)) ? 'Check cancelled.' : `Check failed: ${err.message}`);
    } finally {
      S.busy = false; el.cancel.hidden = true; el.prog.hidden = true; updateButtons();
    }
  });

  // ------------------------------------------------------------ run
  function updateButtons() {
    const ready = !!(S.poscar && el.incarText.value.trim() && S.modelPath && !S.busy && !(S.status && S.status.available === false));
    el.run.disabled = !ready;
    const blocked = checkIncarQuiet();
    if (blocked) el.run.disabled = true;
    const has = !!(S.result && S.result.success);
    el.dlRelaxed.disabled = !has || S.busy;
    el.saveC.disabled = !has || S.busy; el.saveZip.disabled = !has || S.busy;
    el.toDos.disabled = !has || S.busy; el.openF.disabled = !has;
    el.showF.disabled = !has;
    document.querySelectorAll('.rx-file-row .btn, #rxAllBtn, #rxCustomModelBtn, #rxPyBrowse, #rxCheckEnv').forEach((b) => { b.disabled = S.busy; });
  }
  function checkIncarQuiet() {
    const text = el.incarText.value; if (!text.trim()) return false;
    const t = parseIncar(text);
    const nsw = t.NSW === undefined ? null : parseFloat(t.NSW);
    const ibr = t.IBRION === undefined ? null : parseFloat(t.IBRION);
    const im = t.IMAGES === undefined ? 0 : parseFloat(t.IMAGES);
    return nsw === null || nsw === 0 || ibr === 0 || (ibr >= 5 && ibr <= 8) || im > 0;
  }

  function clearResult() {
    S.result = null;
    el.summary.hidden = true; el.table.hidden = true; el.empty.hidden = false; el.tbody.replaceChildren();
    el.pills.hidden = true; el.log.textContent = '';
  }

  const fmtT = (sec) => { sec = Math.round(sec); return sec >= 60 ? `${Math.floor(sec / 60)} min ${String(sec % 60).padStart(2, '0')} s` : `${sec} s`; };
  let timer = null, t0 = 0, phaseText = 'Starting...';
  function tick() {
    const sec = (Date.now() - t0) / 1000;
    el.runText.textContent = `Relaxing... ${fmtT(sec)}`;
    el.phase.textContent = phaseText + (sec > 240 && /Importing|Starting|Loading/.test(phaseText) ? '  Still not computing after ' + fmtT(sec) + ': see the Log tab (a stack trace is printed there every 45 s). Cancel, then use "Python environment (advanced)" > Check environment.' : '');
  }
  function phaseFrom(msg) {
    let m;
    if (/^\[load\] Worker started/.test(msg)) return 'Worker started. Importing PyTorch...';
    if (/^\[load\] PyTorch/.test(msg)) return 'PyTorch loaded. Importing ASE, e3nn and vasp-mace...';
    if (/^\[load\] vasp-mace imported/.test(msg)) return 'vasp-mace loaded. Importing MACE and loading the model...';
    if (/^\[load\] Loading the model/.test(msg)) return 'Loading the model and preparing the first step...';
    if (/^\[info\] Model=/.test(msg)) return 'Model loaded. Computing the first step (the first step is the slowest)...';
    if ((m = /^\[step\s+(\d+)\]\s+E=(\S+)\s+eV.*Fmax=(\S+)/.exec(msg))) return `Step ${m[1]} of at most ${S.runNsw}:  E = ${m[2]} eV,  Fmax = ${m[3]} eV/\u00c5`;
    return null;
  }

  async function run() {
    if (S.busy || el.run.disabled) return;
    S.busy = true; clearResult(); updateButtons();
    const head = el.headField.hidden ? '' : el.head.value.trim();
    S.runNsw = parseInt((parseIncar(el.incarText.value).NSW || '0'), 10) || '?';
    S.runInfo = { model: S.customModel ? S.customName : (curInfo() ? curInfo().label.split(' (')[0] : ''), head, dtype: el.dtype.value };
    el.cancel.hidden = false; el.prog.hidden = false; el.progBar.style.width = '2%';
    el.run.classList.add('running');
    t0 = Date.now(); phaseText = 'Starting the relaxation worker...'; tick(); el.phase.hidden = false;
    timer = setInterval(tick, 1000);
    showTab('rxLogView');
    log(`POSCAR: ${S.poscar.path}\nINCAR: edited copy of ${S.incar ? S.incar.path : '(typed)'}\nKPOINTS: ${S.kpoints ? S.kpoints.path + ' (copied only)' : 'none'}\nModel: ${S.modelPath}${head ? '  (head ' + head + ')' : ''}\nPython: ${el.python.value.trim() || "the app's own"}`, true);
    try {
      const r = await api.relaxRun({
        poscarPath: S.poscar.path,
        incarText: el.incarText.value,
        kpointsPath: S.kpoints ? S.kpoints.path : null,
        modelPath: S.modelPath, head,
        device: el.device.value, dtype: el.dtype.value, optimizer: el.opt.value,
        pythonPath: el.python.value.trim() || null
      });
      if (r && r.log && r.log.length) log(r.log.join('\n'));
      if (!r || !r.success) throw new Error((r && r.error) || 'relaxation failed');
      S.result = r;
      showResult(r);
      S.which = 'final'; showStruct('final');
      showTab('rxResultView');
    } catch (err) {
      const m = String(err.message || err);
      log(/cancel/i.test(m) ? 'Relaxation cancelled.' : `Error: ${m}`);
      el.pills.hidden = false; el.pStatus.innerHTML = '<b></b>'; el.pStatus.firstChild.textContent = /cancel/i.test(m) ? 'Cancelled' : 'Failed';
      el.pSteps.textContent = '-'; el.pDE.textContent = '-';
    } finally {
      clearInterval(timer); timer = null; el.phase.hidden = true;
      S.busy = false; el.cancel.hidden = true; el.prog.hidden = true;
      el.run.classList.remove('running'); el.runText.textContent = 'Relax structure';
      updateButtons();
    }
  }
  el.run.addEventListener('click', run);
  el.cancel.addEventListener('click', async () => { el.cancel.disabled = true; try { await api.cancelJob(); } finally { el.cancel.disabled = false; } });

  if (api && api.onProgress) {
    api.onProgress((p) => {
      if (!S.busy || !p) return;
      if (typeof p.percent === 'number') el.progBar.style.width = Math.max(2, Math.min(100, p.percent)) + '%';
      if (p.message) {
        if (p.stage === 'download') el.dl.textContent = p.message.length > 26 ? 'Downloading...' : p.message;
        else { const ph = phaseFrom(p.message); if (ph) { phaseText = ph; el.phase.textContent = ph; } }
        if (p.message.length > 2) log(p.message);
      }
    });
  }

  // ------------------------------------------------------------ results
  function showResult(r) {
    el.empty.hidden = true; el.table.hidden = false; el.summary.hidden = false;
    const dE = r.final_energy - r.initial_energy;
    el.pills.hidden = false;
    el.pStatus.firstElementChild.textContent = r.converged ? 'Converged' : 'Step limit reached';
    el.pSteps.textContent = `${r.steps} / ${r.nsw}`;
    el.pDE.textContent = `${dE >= 0 ? '+' : ''}${fmt(dE, 3)} eV`;
    const rows = [
      ['Status', r.converged ? 'converged' : 'NSW reached before the force criterion', r.converged ? 'good' : 'warn'],
      ['Ionic steps', `${r.steps} of ${r.nsw}`],
      ['Initial energy', `${fmt(r.initial_energy)} eV`],
      ['Final energy', `${fmt(r.final_energy)} eV`],
      ['Energy change', `${fmt(dE)} eV`],
      ['Final max force', `${fmt(r.final_fmax)} eV/Å`],
      ['Model', S.runInfo ? S.runInfo.model + (S.runInfo.head ? ` / ${S.runInfo.head}` : '') : '-'],
      ['Device', `${r.device}, ${r.dtype}`],
      ['Time', `${fmt(r.elapsed_s, 1)} s`]
    ];
    const kids = rows.map(([k, v, c]) => {
      const d = document.createElement('div');
      const a = document.createElement('span'); a.className = 'k'; a.textContent = k;
      const b = document.createElement('span'); if (c) b.className = c; b.textContent = v;
      d.append(a, b); return d;
    });
    const f = document.createElement('div'); f.className = 'files';
    f.textContent = `Output folder: ${r.run_dir}   (${(r.files || []).map((x) => x.name).join(', ')})` + (r.kpoints_copied ? '' : '');
    kids.push(f);
    el.summary.replaceChildren(...kids);
    const frag = document.createDocumentFragment();
    for (const h of r.history || []) {
      const tr = document.createElement('tr');
      [h.step, fmt(h.energy), fmt(h.dE), fmt(h.fmax)].forEach((v) => { const td = document.createElement('td'); td.className = 'num'; td.textContent = v; tr.append(td); });
      frag.append(tr);
    }
    el.tbody.replaceChildren(frag);
  }

  // ------------------------------------------------------------ viewer
  function showStruct(which) {
    S.which = which;
    document.querySelectorAll('[data-rxwhich]').forEach((b) => b.classList.toggle('active', b.dataset.rxwhich === which));
    const use = which === 'final' && S.result ? S.result.final : (S.initialStruct || (S.result && S.result.initial));
    if (!viewer) return;
    if (use && use.success !== false) {
      viewer.setStructure(use);
      el.title.textContent = `${use.formula || 'Structure'}  (${which === 'final' ? 'relaxed' : 'initial'})`;
    } else { viewer.clear(); el.title.textContent = 'Structure'; }
  }
  document.querySelectorAll('[data-rxwhich]').forEach((b) => b.addEventListener('click', () => { if (!b.disabled) showStruct(b.dataset.rxwhich); }));

  function initViewer() {
    viewer = new window.StructureViewer(el.vCanvas);
    viewer.emptyHint = 'Select a POSCAR to see the structure here';
    viewer.invalidate();
    viewer.onhover = (n, x, y) => {
      if (!n) { el.vTip.hidden = true; return; }
      const s = viewer.struct && viewer.struct.symbols ? viewer.struct.symbols[n - 1] : '';
      el.vTip.textContent = `${s}${n}`; el.vTip.style.left = x + 14 + 'px'; el.vTip.style.top = y + 14 + 'px'; el.vTip.hidden = false;
    };
    document.querySelectorAll('[data-rxview]').forEach((b) => b.addEventListener('click', () => viewer.resetView(b.dataset.rxview)));
    el.optBonds.addEventListener('change', () => viewer.setOption('showBonds', el.optBonds.checked));
    el.optCell.addEventListener('change', () => viewer.setOption('showCell', el.optCell.checked));
    el.optLabels.addEventListener('change', () => viewer.setOption('labels', el.optLabels.value));
    el.optRep.addEventListener('change', () => viewer.setOption('rep', el.optRep.value.split(',').map(Number)));
    el.optSize.addEventListener('input', () => viewer.setOption('size', parseFloat(el.optSize.value)));
    el.saveStruct.addEventListener('click', async () => {
      if (!viewer.struct) return;
      const res = await api.savePng({ dataUrl: viewer.toDataURL(2), defaultName: (viewer.struct.name || 'structure') + '.png' });
      if (res && res.success) log(`Saved image: ${res.savedPath}`);
    });
  }
  new MutationObserver(() => { if (viewer) viewer.refreshTheme(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ------------------------------------------------------------ outputs
  async function saveContcar() {
    if (!S.result) return;
    const res = await api.saveFile({ sourcePath: S.result.contcar, defaultName: 'CONTCAR', extension: '' });
    if (res && res.success) log(`Saved CONTCAR: ${res.savedPath}`);
  }
  el.saveC.addEventListener('click', saveContcar);
  el.dlRelaxed.addEventListener('click', saveContcar);
  el.saveZip.addEventListener('click', async () => {
    if (!S.result) return;
    const res = await api.relaxExportZip({ runDir: S.result.run_dir });
    if (res && res.success) log(`Saved outputs: ${res.savedPath}`);
    else if (res && !res.canceled) log(`Could not zip: ${res.error || 'unknown error'}`);
  });
  el.openF.addEventListener('click', () => { if (S.result) api.showInFolder(S.result.contcar); });
  el.toDos.addEventListener('click', async () => {
    if (!S.result || !window.__dos) return;
    window.__dos.gotoSection('dos');
    await window.__dos.loadStructureFile({ path: S.result.contcar, name: 'CONTCAR (relaxed)', directory: S.result.run_dir });
  });

  // ------------------------------------------------------------ citation
  document.querySelectorAll('[data-ext]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault(); if (api && api.openExternal) api.openExternal(a.dataset.ext);
  }));
  $('rxCopyBib').addEventListener('click', async () => {
    const b = $('rxCopyBib');
    try { await api.copyToClipboard(BIBTEX); b.textContent = 'Copied'; } catch (e) { b.textContent = 'Copy failed'; }
    setTimeout(() => { b.textContent = 'Copy BibTeX (vasp-mace)'; }, 1600);
  });

  // ------------------------------------------------------------ start
  window.addEventListener('DOMContentLoaded', () => { initViewer(); updateButtons(); loadStatus(); }); // Relaxor is the first tab
  window.addEventListener('rx:shown', () => { loadStatus(); requestAnimationFrame(() => viewer && viewer.invalidate()); });
  window.__relax = { state: S, ingest, run, loadStatus, get viewer() { return viewer; } };
})();
