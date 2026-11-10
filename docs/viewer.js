'use strict';
// =========================================================================
// Padarth Khoj - 3D structure viewer (dependency free, Canvas 2D)
//   orthographic projection, depth-sorted shaded spheres + bonds, periodic cell,
//   supercell repeat, atom picking, colouring by element or by predicted overpotential
// =========================================================================
(function (global) {
  const EL = global.PK_ELEMENTS || {};
  const elInfo = (sym) => EL[sym] || [1.0, '#c06bff'];

  // ---- small colour helpers
  function hex2rgb(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function parseColor(c) {
    if (c[0] === '#') return hex2rgb(c);
    const m = /rgba?\(([^)]+)\)/.exec(c);
    if (m) return m[1].split(',').slice(0, 3).map((v) => parseFloat(v));
    return [128, 128, 128];
  }
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const css = (rgb, a) => (a === undefined || a >= 1
    ? `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`
    : `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},${a})`);
  const WHITE = [255, 255, 255];
  const BLACK = [0, 0, 0];

  // ---- 3x3 matrices (row major)
  const mmul = (A, B) => {
    const R = new Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) R[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    return R;
  };
  const rotX = (a) => { const c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, -s, 0, s, c]; };
  const rotY = (a) => { const c = Math.cos(a), s = Math.sin(a); return [c, 0, s, 0, 1, 0, -s, 0, c]; };
  const rotZ = (a) => { const c = Math.cos(a), s = Math.sin(a); return [c, -s, 0, s, c, 0, 0, 0, 1]; };
  function orthonormalize(R) {
    let a = [R[0], R[1], R[2]], b = [R[3], R[4], R[5]];
    const n = (v) => Math.hypot(v[0], v[1], v[2]);
    let l = n(a); a = a.map((x) => x / l);
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    b = [b[0] - d * a[0], b[1] - d * a[1], b[2] - d * a[2]];
    l = n(b); b = b.map((x) => x / l);
    const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    return [...a, ...b, ...c];
  }

  const VIEWS = {
    top: () => [1, 0, 0, 0, 1, 0, 0, 0, 1],          // looking down the c axis
    front: () => rotX(-Math.PI / 2),                  // looking along b
    side: () => [0, 1, 0, 0, 0, 1, 1, 0, 0],          // looking along a
    iso: () => mmul(rotX(-1.0), rotZ(0.45))
  };

  // low overpotential (best) -> yellow, high -> violet
  const ETA_STOPS = [
    [0.0, '#fde725'], [0.25, '#7ad151'], [0.5, '#22a884'], [0.75, '#2a788e'], [1.0, '#7e4aa8']
  ].map(([t, c]) => [t, hex2rgb(c)]);
  function etaColor(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < ETA_STOPS.length; i++) {
      if (t <= ETA_STOPS[i][0]) {
        const [t0, c0] = ETA_STOPS[i - 1], [t1, c1] = ETA_STOPS[i];
        return mix(c0, c1, (t - t0) / (t1 - t0));
      }
    }
    return ETA_STOPS[ETA_STOPS.length - 1][1];
  }

  class StructureViewer {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.struct = null;
      this.atoms = [];
      this.bonds = [];
      this.results = new Map();
      this.colorMode = 'element';
      this.opts = { showCell: true, showBonds: true, labels: 'none', size: 1.0, rep: [1, 1, 1] };
      this.R = VIEWS.iso();
      this.zoom = 1;
      this.pan = [0, 0];
      this.baseScale = 20;
      this.center = [0, 0, 0];
      this.selected = null;
      this.hover = null;
      this.hits = [];
      this.onselect = null;
      this.onhover = null;
      this.eMin = 0;
      this.eMax = 1;
      this.bestAtom = null;
      this.dpr = global.devicePixelRatio || 1;
      this._dirty = true;
      this._geom = null;
      this._theme();
      this._bind();
      this._ro = new ResizeObserver(() => this._resize());
      this._ro.observe(canvas.parentElement || canvas);
      this._resize();
      const loop = () => { if (this._dirty) { this._dirty = false; this._draw(); } requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }

    // ------------------------------------------------------------ public API
    refreshTheme() { this._theme(); this.invalidate(); }
    invalidate() { this._dirty = true; }

    setStructure(s) {
      this.struct = s && s.success !== false ? s : null;
      this.selected = null;
      this.hover = null;
      this._buildAtoms();
      this._computeBonds();
      this._geom = null;
      this.resetView('iso');
    }

    clear() { this.setStructure(null); this.results = new Map(); }

    /** map: atomNumber -> { eta, dg, partner } */
    setResults(map, bestAtom) {
      this.results = map || new Map();
      this.bestAtom = bestAtom || null;
      const etas = [...this.results.values()].map((r) => r.eta).filter((v) => v !== null && v !== undefined && isFinite(v));
      this.eMin = etas.length ? Math.min(...etas) : 0;
      this.eMax = etas.length ? Math.max(...etas) : 1;
      this.invalidate();
    }

    setColorMode(mode) { this.colorMode = mode; this.invalidate(); }
    getColorScale() { return { min: this.eMin, max: this.eMax, stops: ETA_STOPS.map(([t, c]) => [t, css(c)]) }; }

    setOption(name, value) {
      this.opts[name] = value;
      if (name === 'rep') { this._geom = null; this.fit(); }
      if (name === 'showCell') this.fit();
      this.invalidate();
    }

    select(n, quiet) {
      this.selected = n || null;
      this.invalidate();
      if (!quiet && this.onselect) this.onselect(this.selected);
    }

    resetView(preset) {
      this.R = (VIEWS[preset] || VIEWS.iso)();
      this.zoom = 1;
      this.fit();
    }

    toDataURL(scale) {
      const pr = Math.max(scale || 2, this.dpr);
      const old = this.dpr;
      this.dpr = pr;
      this._resize(true);
      this._draw();
      const url = this.canvas.toDataURL('image/png');
      this.dpr = old;
      this._resize();
      this._draw();
      return url;
    }

    atomInfo(n) {
      if (!this.struct || n < 1 || n > this.atoms.length) return null;
      const a = this.atoms[n - 1];
      return { n, sym: a.sym, cart: a.p.slice(), result: this.results.get(n) || null };
    }

    // ------------------------------------------------------------ geometry
    _buildAtoms() {
      const s = this.struct;
      this.atoms = [];
      if (!s) return;
      for (let i = 0; i < s.symbols.length; i++) {
        this.atoms.push({ n: i + 1, sym: s.symbols[i], p: s.cart[i].slice(), f: s.frac ? s.frac[i].slice() : null, r: elInfo(s.symbols[i])[0], col: hex2rgb(elInfo(s.symbols[i])[1]) });
      }
      // centre on the atoms
      const c = [0, 0, 0];
      for (const a of this.atoms) for (let k = 0; k < 3; k++) c[k] += a.p[k] / this.atoms.length;
      this.center = c;
    }

    _lattice() { return this.struct && this.struct.lattice ? this.struct.lattice : null; }

    _computeBonds() {
      this.bonds = [];
      const L = this._lattice();
      const at = this.atoms;
      const n = at.length;
      if (!n) return;
      const scale = 1.15;
      let maxR = 0;
      for (const a of at) maxR = Math.max(maxR, a.r);
      const maxCut = scale * 2 * maxR;
      let single = false;
      let inv = null;
      if (L) {
        const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        const V = Math.abs(dot(L[0], cross(L[1], L[2])));
        const h = [V / Math.hypot(...cross(L[1], L[2])), V / Math.hypot(...cross(L[2], L[0])), V / Math.hypot(...cross(L[0], L[1]))];
        single = Math.min(...h) > 2 * maxCut;
        inv = true;
      }
      const f = at.map((a) => a.f);
      const hasFrac = L && f.every((x) => x);
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const cut = scale * (at[i].r + at[j].r);
          if (hasFrac) {
            const df = [f[j][0] - f[i][0], f[j][1] - f[i][1], f[j][2] - f[i][2]];
            const base = df.map((x) => -Math.round(x));
            const range = single ? [[0, 0, 0]] : [].concat(...[-1, 0, 1].map((a) => [].concat(...[-1, 0, 1].map((b) => [-1, 0, 1].map((c) => [a, b, c])))));
            for (const sh of range) {
              const s = [base[0] + sh[0], base[1] + sh[1], base[2] + sh[2]];
              const d = [df[0] + s[0], df[1] + s[1], df[2] + s[2]];
              const x = d[0] * L[0][0] + d[1] * L[1][0] + d[2] * L[2][0];
              const y = d[0] * L[0][1] + d[1] * L[1][1] + d[2] * L[2][1];
              const z = d[0] * L[0][2] + d[1] * L[1][2] + d[2] * L[2][2];
              const dist = Math.hypot(x, y, z);
              if (dist < cut && dist > 0.4) this.bonds.push({ i, j, s });
            }
          } else {
            const dist = Math.hypot(at[i].p[0] - at[j].p[0], at[i].p[1] - at[j].p[1], at[i].p[2] - at[j].p[2]);
            if (dist < cut && dist > 0.4) this.bonds.push({ i, j, s: [0, 0, 0] });
          }
        }
      }
    }

    /** Atom instances and bond halves for the current supercell repeat (Cartesian, centred). */
    _buildGeom() {
      const L = this._lattice();
      const rep = L ? this.opts.rep : [1, 1, 1];
      const ctr = this.center.slice();
      // centre of the displayed block
      if (L) for (let k = 0; k < 3; k++) for (let d = 0; d < 3; d++) ctr[d] += ((rep[k] - 1) / 2) * L[k][d];
      const T = (R) => [R[0] * L[0][0] + R[1] * L[1][0] + R[2] * L[2][0], R[0] * L[0][1] + R[1] * L[1][1] + R[2] * L[2][1], R[0] * L[0][2] + R[1] * L[1][2] + R[2] * L[2][2]];
      const inst = [];
      const reps = [];
      for (let a = 0; a < rep[0]; a++) for (let b = 0; b < rep[1]; b++) for (let c = 0; c < rep[2]; c++) reps.push([a, b, c]);
      for (const R of reps) {
        const t = L ? T(R) : [0, 0, 0];
        for (const a of this.atoms) inst.push({ a, base: R[0] === 0 && R[1] === 0 && R[2] === 0, p: [a.p[0] + t[0] - ctr[0], a.p[1] + t[1] - ctr[1], a.p[2] + t[2] - ctr[2]] });
      }
      const halves = [];
      const inRange = (R) => R[0] >= 0 && R[0] < rep[0] && R[1] >= 0 && R[1] < rep[1] && R[2] >= 0 && R[2] < rep[2];
      const ext = [[-1, rep[0]], [-1, rep[1]], [-1, rep[2]]];
      for (const bd of this.bonds) {
        const A = this.atoms[bd.i], B = this.atoms[bd.j];
        for (let a = ext[0][0]; a <= ext[0][1]; a++) for (let b = ext[1][0]; b <= ext[1][1]; b++) for (let c = ext[2][0]; c <= ext[2][1]; c++) {
          const Ri = [a, b, c];
          const Rj = [a + bd.s[0], b + bd.s[1], c + bd.s[2]];
          const inI = inRange(Ri), inJ = inRange(Rj);
          if (!inI && !inJ) continue;
          const ti = L ? T(Ri) : [0, 0, 0];
          const tj = L ? T(Rj) : [0, 0, 0];
          const pi = [A.p[0] + ti[0] - ctr[0], A.p[1] + ti[1] - ctr[1], A.p[2] + ti[2] - ctr[2]];
          const pj = [B.p[0] + tj[0] - ctr[0], B.p[1] + tj[1] - ctr[1], B.p[2] + tj[2] - ctr[2]];
          const mid = [(pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2, (pi[2] + pj[2]) / 2];
          if (inI) halves.push({ p1: pi, p2: mid, a: A });
          if (inJ) halves.push({ p1: pj, p2: mid, a: B });
        }
      }
      let corners = null;
      if (L) {
        corners = [];
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) {
          const q = T([i * rep[0], j * rep[1], k * rep[2]]);
          corners.push([q[0] - ctr[0], q[1] - ctr[1], q[2] - ctr[2]]);
        }
      }
      this._geom = { inst, halves, corners, ctr, rep, T, L };
    }

    // ------------------------------------------------------------ view
    _proj(p) {
      const R = this.R;
      return [R[0] * p[0] + R[1] * p[1] + R[2] * p[2], R[3] * p[0] + R[4] * p[1] + R[5] * p[2], R[6] * p[0] + R[7] * p[1] + R[8] * p[2]];
    }

    fit() {
      if (!this.struct) { this._dirty = true; return; }
      if (!this._geom) this._buildGeom();
      const g = this._geom;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const add = (p, r) => {
        const v = this._proj(p);
        x0 = Math.min(x0, v[0] - r); x1 = Math.max(x1, v[0] + r);
        y0 = Math.min(y0, v[1] - r); y1 = Math.max(y1, v[1] + r);
      };
      for (const i of g.inst) add(i.p, i.a.r * 0.55);
      const w = this.cssW, h = this.cssH, m = 46;
      const sx = (w - 2 * m) / Math.max(x1 - x0, 1e-6), sy = (h - 2 * m) / Math.max(y1 - y0, 1e-6);
      this.baseScale = Math.max(2, Math.min(sx, sy));
      this.zoom = 1;
      this.pan = [-((x0 + x1) / 2) * this.baseScale, ((y0 + y1) / 2) * this.baseScale];
      this._dirty = true;
    }

    _theme() {
      const cs = getComputedStyle(this.canvas);
      const g = (n, d) => (cs.getPropertyValue(n).trim() || d);
      this.th = {
        bg: g('--viewer-bg', '#0d0f16'),
        fg: g('--viewer-fg', '#e7e9f2'),
        muted: g('--viewer-muted', '#8b90a7'),
        accent: g('--accent', '#ff5a1f'),
        gold: g('--viewer-gold', '#ffc83d')
      };
    }

    _resize(force) {
      const par = this.canvas.parentElement || this.canvas;
      const w = Math.max(10, par.clientWidth), h = Math.max(10, par.clientHeight);
      if (!force && w === this.cssW && h === this.cssH && this._lastDpr === this.dpr) return;
      const first = !this.cssW;
      this.cssW = w; this.cssH = h; this._lastDpr = this.dpr;
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      this.canvas.width = Math.round(w * this.dpr);
      this.canvas.height = Math.round(h * this.dpr);
      if (first && this.struct) this.fit();
      this._dirty = true;
    }

    // ------------------------------------------------------------ drawing
    _atomColor(a) {
      const bg = parseColor(this.th.bg);
      if (this.colorMode === 'eta') {
        const r = this.results.get(a.n);
        if (r && r.eta !== null && r.eta !== undefined) {
          const t = this.eMax > this.eMin ? (r.eta - this.eMin) / (this.eMax - this.eMin) : 0;
          return { rgb: etaColor(t), dim: false };
        }
        return { rgb: mix(a.col, bg, 0.62), dim: true };
      }
      return { rgb: a.col, dim: false };
    }

    _draw() {
      const ctx = this.ctx;
      const W = this.cssW, H = this.cssH;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.fillStyle = this.th.bg;
      ctx.fillRect(0, 0, W, H);
      this.hits = [];
      if (!this.struct) {
        ctx.fillStyle = this.th.muted;
        ctx.font = '14px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No structure loaded', W / 2, H / 2 - 8);
        ctx.font = '12px system-ui, sans-serif';
        ctx.fillText(this.emptyHint || 'Upload a DOSCAR (CONTCAR in the same folder) or open a POSCAR / CONTCAR / xyz file', W / 2, H / 2 + 14);
        return;
      }
      if (!this._geom) this._buildGeom();
      const g = this._geom;
      const sc = this.baseScale * this.zoom;
      const cx = W / 2 + this.pan[0] * this.zoom, cy = H / 2 + this.pan[1] * this.zoom;
      const screen = (v) => [cx + v[0] * sc, cy - v[1] * sc];
      const bgRGB = parseColor(this.th.bg);
      const fgRGB = parseColor(this.th.fg);

      // depth range for cueing
      let zmin = Infinity, zmax = -Infinity;
      const items = [];
      for (const i of g.inst) {
        const v = this._proj(i.p);
        i.v = v;
        zmin = Math.min(zmin, v[2]); zmax = Math.max(zmax, v[2]);
      }
      const zr = Math.max(zmax - zmin, 1e-6);
      const cue = (z) => 0.62 + 0.38 * ((z - zmin) / zr);

      // unit cell
      if (g.corners && this.opts.showCell) {
        const idx = (i, j, k) => i * 4 + j * 2 + k;
        const edges = [];
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { edges.push([idx(i, j, 0), idx(i, j, 1)]); edges.push([idx(i, 0, j), idx(i, 1, j)]); edges.push([idx(0, i, j), idx(1, i, j)]); }
        ctx.lineWidth = 1;
        ctx.strokeStyle = css(mix(parseColor(this.th.muted), bgRGB, 0.35));
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (const [a, b] of edges) {
          const p = screen(this._proj(g.corners[a])), q = screen(this._proj(g.corners[b]));
          ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // bonds
      const bw = Math.max(1.5, sc * 0.13 * Math.min(1.4, this.opts.size));
      if (this.opts.showBonds) {
        for (const h of g.halves) {
          const v1 = this._proj(h.p1), v2 = this._proj(h.p2);
          const col = this._atomColor(h.a);
          items.push({ z: (v1[2] + v2[2]) / 2 - 0.001, kind: 'bond', p: screen(v1), q: screen(v2), col: col.rgb, zc: cue((v1[2] + v2[2]) / 2), dim: col.dim });
        }
      }
      // atoms
      for (const i of g.inst) {
        const col = this._atomColor(i.a);
        const r = Math.max(3, i.a.r * 0.55 * this.opts.size * sc);
        items.push({ z: i.v[2], kind: 'atom', inst: i, p: screen(i.v), r, col: col.rgb, dim: col.dim, zc: cue(i.v[2]) });
      }
      items.sort((a, b) => a.z - b.z);

      ctx.lineCap = 'round';
      for (const it of items) {
        if (it.kind === 'bond') {
          const c = mix(BLACK, it.col, 0.55 + 0.35 * it.zc);
          ctx.strokeStyle = css(it.dim ? mix(c, bgRGB, 0.3) : c);
          ctx.lineWidth = bw;
          ctx.beginPath(); ctx.moveTo(it.p[0], it.p[1]); ctx.lineTo(it.q[0], it.q[1]); ctx.stroke();
        } else {
          this._drawAtom(ctx, it, bgRGB);
        }
      }

      // overlays: rings / labels
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const labelMode = this.opts.labels;
      for (const it of items) {
        if (it.kind !== 'atom') continue;
        const a = it.inst.a;
        const isSel = this.selected === a.n, isHover = this.hover === a.n, isBest = this.bestAtom === a.n;
        const selRes = this.selected ? this.results.get(this.selected) : null;
        const isPartner = selRes && selRes.partner === a.n;
        if (isBest || isSel || isHover || isPartner) {
          ctx.lineWidth = isSel ? 3 : 2;
          ctx.strokeStyle = isSel ? this.th.accent : isBest ? this.th.gold : isPartner ? this.th.accent : this.th.fg;
          ctx.setLineDash(isPartner && !isSel ? [4, 3] : []);
          ctx.beginPath(); ctx.arc(it.p[0], it.p[1], it.r + 3.5, 0, Math.PI * 2); ctx.stroke();
          ctx.setLineDash([]);
        }
        let text = null;
        if (labelMode === 'index') text = String(a.n);
        else if (labelMode === 'eta') { const r = this.results.get(a.n); if (r && r.eta !== null && r.eta !== undefined) text = r.eta.toFixed(2); }
        else if (labelMode === 'element') text = a.sym + a.n;
        if (isSel || isBest) { const r = this.results.get(a.n); if (!text) text = r && r.eta !== null && r.eta !== undefined ? `${a.n}  ${r.eta.toFixed(3)} V` : String(a.n); }
        if (text && it.r > 4 && it.inst.base) {
          const fs = Math.max(10, Math.min(13, it.r * 0.9));
          ctx.font = `600 ${fs}px system-ui, "Segoe UI", sans-serif`;
          ctx.lineWidth = 3;
          ctx.strokeStyle = css(bgRGB, 0.85);
          ctx.strokeText(text, it.p[0], it.p[1] - it.r - 9);
          ctx.fillStyle = css(fgRGB);
          ctx.fillText(text, it.p[0], it.p[1] - it.r - 9);
        }
      }

      this._drawAxes(ctx, g);
    }

    _drawAtom(ctx, it, bgRGB) {
      const { p, r } = it;
      const base = mix(mix(BLACK, it.col, 0.55 + 0.45 * it.zc), bgRGB, it.dim ? 0.15 : 0);
      const grad = ctx.createRadialGradient(p[0] - r * 0.35, p[1] - r * 0.38, r * 0.08, p[0], p[1], r);
      grad.addColorStop(0, css(mix(base, WHITE, 0.55)));
      grad.addColorStop(0.5, css(base));
      grad.addColorStop(1, css(mix(base, BLACK, 0.5)));
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill();
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = css(mix(base, BLACK, 0.65), 0.9);
      ctx.stroke();
      this.hits.push({ n: it.inst.a.n, x: p[0], y: p[1], r });
    }

    _drawAxes(ctx, g) {
      if (!g.L) return;
      const ox = 38, oy = this.cssH - 38, len = 24;
      const cols = ['#ff6b5a', '#59d36b', '#5aa7ff'];
      const names = ['a', 'b', 'c'];
      const dirs = g.L.map((v, k) => {
        const n = Math.hypot(...v);
        const w = this._proj([v[0] / n, v[1] / n, v[2] / n]);
        return { k, w };
      }).sort((x, y) => x.w[2] - y.w[2]);
      ctx.lineWidth = 2;
      ctx.font = '600 11px system-ui, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const d of dirs) {
        const ex = ox + d.w[0] * len, ey = oy - d.w[1] * len;
        ctx.strokeStyle = cols[d.k];
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.fillStyle = cols[d.k];
        ctx.fillText(names[d.k], ox + d.w[0] * (len + 9), oy - d.w[1] * (len + 9));
      }
    }

    // ------------------------------------------------------------ interaction
    _pick(x, y) {
      for (let i = this.hits.length - 1; i >= 0; i--) {
        const h = this.hits[i];
        if ((x - h.x) ** 2 + (y - h.y) ** 2 <= (h.r + 2) ** 2) return h.n;
      }
      return null;
    }

    _bind() {
      const c = this.canvas;
      let drag = null;
      const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      c.addEventListener('contextmenu', (e) => e.preventDefault());
      c.addEventListener('pointerdown', (e) => {
        c.setPointerCapture(e.pointerId);
        const [x, y] = pos(e);
        drag = { x, y, x0: x, y0: y, moved: false, pan: e.shiftKey || e.button === 2 || e.button === 1 };
      });
      c.addEventListener('pointermove', (e) => {
        const [x, y] = pos(e);
        if (drag) {
          const dx = x - drag.x, dy = y - drag.y;
          if (Math.hypot(x - drag.x0, y - drag.y0) > 3) drag.moved = true;
          if (drag.moved) {
            if (drag.pan) { this.pan[0] += dx / this.zoom; this.pan[1] += dy / this.zoom; }
            else { this.R = orthonormalize(mmul(mmul(rotX(dy * 0.009), rotY(dx * 0.009)), this.R)); }
            this._dirty = true;
          }
          drag.x = x; drag.y = y;
        } else {
          const n = this._pick(x, y);
          if (n !== this.hover) {
            this.hover = n;
            c.style.cursor = n ? 'pointer' : 'grab';
            this._dirty = true;
            if (this.onhover) this.onhover(n, e.clientX, e.clientY);
          } else if (n && this.onhover) {
            this.onhover(n, e.clientX, e.clientY);
          }
        }
      });
      const up = (e) => {
        if (!drag) return;
        if (!drag.moved) {
          const [x, y] = pos(e);
          const n = this._pick(x, y);
          this.select(n === this.selected ? null : n);
        }
        drag = null;
      };
      c.addEventListener('pointerup', up);
      c.addEventListener('pointercancel', () => { drag = null; });
      c.addEventListener('pointerleave', () => {
        if (this.hover !== null) { this.hover = null; this._dirty = true; if (this.onhover) this.onhover(null); }
      });
      c.addEventListener('wheel', (e) => {
        e.preventDefault();
        const [x, y] = pos(e);
        const f = Math.exp(-e.deltaY * 0.0015);
        const nz = Math.max(0.2, Math.min(12, this.zoom * f));
        // keep the point under the cursor fixed
        const mx = x - this.cssW / 2, my = y - this.cssH / 2;
        this.pan[0] += mx / nz - mx / this.zoom;
        this.pan[1] += my / nz - my / this.zoom;
        this.zoom = nz;
        this._dirty = true;
      }, { passive: false });
      c.addEventListener('dblclick', () => this.fit());
      c.style.cursor = 'grab';
      c.tabIndex = 0;
      c.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') this.select(null);
        else if (e.key === '+' || e.key === '=') { this.zoom = Math.min(12, this.zoom * 1.15); this._dirty = true; }
        else if (e.key === '-') { this.zoom = Math.max(0.2, this.zoom / 1.15); this._dirty = true; }
        else if (e.key === 'ArrowLeft') { this.R = orthonormalize(mmul(rotY(-0.08), this.R)); this._dirty = true; }
        else if (e.key === 'ArrowRight') { this.R = orthonormalize(mmul(rotY(0.08), this.R)); this._dirty = true; }
        else if (e.key === 'ArrowUp') { this.R = orthonormalize(mmul(rotX(-0.08), this.R)); this._dirty = true; }
        else if (e.key === 'ArrowDown') { this.R = orthonormalize(mmul(rotX(0.08), this.R)); this._dirty = true; }
      });
    }
  }

  global.StructureViewer = StructureViewer;
})(window);
