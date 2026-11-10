"""Compare the NumPy predictor with the original PyTorch one on real structures. Needs torch + ase (desktop engine env).
usage: python tests/verify_numpy_port.py <desktop engine dir> <structure files...>"""
import sys, os, time, importlib.util
import numpy as np
eng = sys.argv[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
torchdp = load("torchdp", os.path.join(eng, "dos_predictor.py"))
npdp = load("npdp", os.path.join(os.path.dirname(__file__), "..", "py", "dos_predictor.py"))
tp = torchdp.get_predictor(os.path.join(eng, "models"))
npp = npdp.get_predictor(os.path.join(os.path.dirname(__file__), "..", "models"))
bad = 0
for f in sys.argv[2:]:
    a_t = torchdp.read_structure(f); a_n = npdp.read_structure(f)
    assert a_t.get_chemical_symbols() == a_n.get_chemical_symbols(), "symbols differ"
    assert np.allclose(a_t.get_positions(), a_n.get_positions(), atol=1e-6), "positions differ"
    g1 = torchdp.build_graph(a_t); g2 = npdp.build_graph(a_n)
    assert np.allclose(g1[0], g2[0], atol=1e-4), "node features differ (%g)" % np.abs(g1[0] - g2[0]).max()
    assert g1[1].shape == g2[1].shape and g1[3].shape == g2[3].shape, "edge counts differ"
    t0 = time.time(); y1 = tp.predict(a_t); t1 = time.time(); y2 = npp.predict(a_n); t2 = time.time()
    err = np.abs(y1 - y2).max(); ref = np.abs(y1).max()
    ok = err < 2e-3 * max(ref, 1e-6) + 1e-5
    print("%-28s atoms=%3d  max|diff|=%.2e (peak %.3f)  torch %.1fs numpy %.1fs  %s" % (os.path.basename(f), len(a_t), err, ref, t1 - t0, t2 - t1, "OK" if ok else "MISMATCH"))
    bad += (not ok)
sys.exit(1 if bad else 0)
