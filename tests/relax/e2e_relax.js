'use strict';
// End-to-end: launches the real Electron app, loads POSCAR/INCAR/KPOINTS, relaxes with MACE, checks the UI.
// Usage: PADARTH_PYTHON=<python with vasp-mace> PADARTH_E2E_DIR=<dir with POSCAR INCAR KPOINTS> PADARTH_TEST_MODEL=<x.model> \
//        xvfb-run -a node tests/relax/e2e_relax.js
const path = require('path'); const fs = require('fs');
const { _electron: electron } = require('playwright');
const dir = process.env.PADARTH_E2E_DIR, model = process.env.PADARTH_TEST_MODEL;
const root = path.join(__dirname, '..', '..');
let fails = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
(async () => {
  const app = await electron.launch({ executablePath: path.join(root, 'node_modules', 'electron', 'dist', 'electron'), args: [root, '--no-sandbox'], env: { ...process.env } });
  const win = await app.firstWindow();
  const errs = []; win.on('pageerror', (e) => errs.push(e.message));
  await win.waitForSelector('#sectionActivity');
  await win.click('[data-section="relax"]');
  ok(await win.isVisible('#sectionRelax'), 'Relaxor section opens');
  ok((await win.textContent('#rxCite')).includes('10.5281/zenodo.19479246') && (await win.textContent('#rxCite')).includes('MACE'), 'citation visible in UI');
  ok(await win.isDisabled('#rxRunBtn'), 'Run disabled with no inputs');
  await win.waitForFunction(() => document.getElementById('rxEngineNote').textContent.length > 0, null, { timeout: 60000 });
  ok((await win.textContent('#rxEngineNote')).includes('vasp-mace ready'), 'engine reports vasp-mace ready');
  const rd = (n) => ({ path: path.join(dir, n), name: n, directory: dir, text: fs.readFileSync(path.join(dir, n), 'utf8') });
  await win.evaluate(async ([a, b, c]) => window.__relax.ingest([{ ...a, kind: 'poscar' }, { ...b, kind: 'incar' }, { ...c, kind: 'kpoints' }]), [rd('POSCAR'), rd('INCAR'), rd('KPOINTS')]);
  ok((await win.textContent('#rxPoscarName')).includes('atoms'), 'POSCAR parsed: ' + (await win.textContent('#rxPoscarName')));
  ok((await win.inputValue('#rxIncarText')).includes('NSW = 8'), 'INCAR shown in editor');
  ok((await win.textContent('#rxKpointsName')).includes('KPOINTS'), 'KPOINTS accepted');
  ok(await win.isDisabled('#rxRunBtn'), 'Run disabled until a model exists (or custom model chosen)');
  await win.evaluate((m) => { const S = window.__relax.state; S.customModel = true; S.modelPath = m; document.getElementById('rxCustomModelBtn').dispatchEvent(new Event('x')); }, model);
  await win.fill('#rxIncarText', (await win.inputValue('#rxIncarText')) + ''); // triggers updateButtons
  ok(await win.isEnabled('#rxRunBtn'), 'Run enabled with POSCAR + INCAR + model');
  // blocking INCAR
  await win.fill('#rxIncarText', 'NSW = 0\nIBRION = 2\n');
  ok(await win.isDisabled('#rxRunBtn') && !(await win.isHidden('#rxIncarCheck')), 'NSW=0 blocks Run and shows a message');
  await win.click('#rxFixBtn');
  ok((await win.inputValue('#rxIncarText')).includes('NSW = 200'), 'fix button sets relaxation defaults');
  await win.fill('#rxIncarText', fs.readFileSync(path.join(dir, 'INCAR'), 'utf8'));
  await win.selectOption('#rxDevice', 'cpu');
  await win.click('#rxRunBtn');
  ok(await win.isVisible('#rxCancelBtn'), 'Cancel button shown while running');
  await win.waitForFunction(() => !document.getElementById('rxProgress').hidden === false && window.__relax.state.result !== null, null, { timeout: 240000 });
  const r = await win.evaluate(() => { const x = window.__relax.state.result; return { s: x.success, steps: x.steps, e0: x.initial_energy, e1: x.final_energy }; });
  ok(r.s && r.e1 < r.e0, `relaxation ran: ${r.steps} steps, ${r.e0.toFixed(3)} -> ${r.e1.toFixed(3)} eV`);
  ok((await win.locator('#rxTableBody tr').count()) >= 2, 'results table filled');
  ok(await win.isEnabled('#rxSaveContcar') && await win.isEnabled('#rxToDos') && await win.isEnabled('#rxShowFinal'), 'result buttons enabled');
  ok((await win.textContent('#rxStructTitle')).includes('relaxed'), 'viewer shows relaxed structure');
  await win.click('#rxShowInitial');
  ok((await win.textContent('#rxStructTitle')).includes('initial'), 'viewer toggles to initial');
  const files = await win.evaluate(() => window.__relax.state.result.files.map((x) => x.name));
  ok(['CONTCAR', 'OSZICAR', 'OUTCAR', 'KPOINTS'].every((f) => files.includes(f)), 'outputs written: ' + files.join(','));
  await win.click('#rxToDos');
  await win.waitForTimeout(1500);
  ok(await win.isVisible('#sectionDos') && (await win.textContent('#dosStructTitle')).includes('C'), 'Send to DOS Predictor loads CONTCAR: ' + (await win.textContent('#dosStructTitle')));
  await win.screenshot({ path: path.join(dir, 'relax_ui.png') });
  await win.click('[data-section="relax"]');
  await win.screenshot({ path: path.join(dir, 'relax_ui2.png') });
  ok(errs.length === 0, 'no page errors ' + errs.join('|'));
  await app.close();
  console.log(fails ? `${fails} FAILED` : 'ALL PASSED');
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
