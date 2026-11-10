"""Export the PyTorch weights to NumPy .npz (BatchNorm folded into the convolutions) for the in-browser predictor.
Run once with the desktop app's Python (needs torch):  python tools/export_weights.py <engine/models dir> <out dir>"""
import sys, numpy as np, torch
src, out = sys.argv[1], sys.argv[2]
s1 = torch.load(src + "/stage1_best.pt", map_location="cpu")
np.savez_compressed(out + "/stage1.npz", **{k: v.numpy().astype(np.float32) for k, v in s1.items()})
s2 = torch.load(src + "/stage2_best.pt", map_location="cpu")
# fold every "<prefix>.<i>.weight / <j>" BatchNorm that follows a conv inside ResBlock1D.net: keys net.0 (conv) net.1 (bn) net.3 (conv) net.4 (bn)
res = {}
done = set()
for k in list(s2):
    if k.endswith("net.1.running_mean") or k.endswith("net.4.running_mean"):
        bn = k[:-len("running_mean")]            # e.g. enc1.2.net.1.
        ci = bn[:-2] + ("0." if bn.endswith("net.1.") else "3.")
        w, b = s2[ci + "weight"], s2[ci + "bias"]
        g, be, rm, rv = s2[bn + "weight"], s2[bn + "bias"], s2[bn + "running_mean"], s2[bn + "running_var"]
        f = g / torch.sqrt(rv + 1e-5)
        res[ci + "weight"] = (w * f.view(-1, 1, 1)).numpy()
        res[ci + "bias"] = ((b - rm) * f + be).numpy()
        done.update({ci + "weight", ci + "bias", bn + "weight", bn + "bias", bn + "running_mean", bn + "running_var", bn + "num_batches_tracked"})
for k, v in s2.items():
    if k not in done and k not in res:
        res[k] = v.numpy()
np.savez_compressed(out + "/stage2.npz", **{k: v.astype(np.float32) for k, v in res.items()})
print(len(s1), "stage-1 tensors,", len(res), "stage-2 tensors")
