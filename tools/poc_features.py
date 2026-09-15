"""Does model/features.py produce the same values as the navsim builder?

It is a reimplementation, so it is only worth having if it matches.
$NAVSIM_PY writes the reference and $TT_PY compares against it -- two
interpreters over the same tokens.
"""
import argparse, gzip, os, pathlib, pickle, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CACHE = EXP / "data_cache_navtest"


def tokens(n):
    fs = sorted(CACHE.glob("*/*/sparsedrive_feature.gz"))[:n]
    return [(f.parent.name, f) for f in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ref", "check"], required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    out = EXP / "feat_ref.pt"

    if a.mode == "ref":       # $NAVSIM_PY
        sys.path.insert(0, os.environ["NAVSIM_DEVKIT_ROOT"])
        import reference; reference.install()
        from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder
        sys.path.insert(0, str(ROOT / "test"))
        from test_forward_smoke import build_config
        b = SparseDriveFeatureBuilder(build_config())
        ref = {}
        for tok, f in tokens(a.n):
            with gzip.open(f, "rb") as fh:
                feat = pickle.load(fh)
            r, _, _ = b.pipeline(dict(feat), {}, tok, test_mode=True)
            c = r["camera_feature"]
            ref[tok] = dict(imgs=c["imgs"].float(), proj=c["projection_mat"].float(),
                            iwh=torch.as_tensor(c["image_wh"].copy()).float(),
                            status=r["status_feature"].float())
        torch.save(ref, out)
        print(f"  wrote the reference for {len(ref)} tokens")
        return 0

    ref = torch.load(out, map_location="cpu", weights_only=False)
    from model.features import build
    print(f"  {'token':<18} {'imgs':>12} {'proj':>12} {'iwh':>8} {'status':>8}")
    ok = True
    for tok, f in tokens(a.n):
        with gzip.open(f, "rb") as fh:
            feat = pickle.load(fh)
        imgs, status, proj, iwh = build(feat)
        r = ref[tok]
        ds = [float((imgs - r["imgs"]).abs().max()), float((proj - r["proj"]).abs().max()),
              float((iwh - r["iwh"]).abs().max()), float((status - r["status"]).abs().max())]
        ok &= max(ds) < 1e-4
        print(f"  {tok:<18} " + " ".join(f"{d:12.2e}" if i < 2 else f"{d:8.2e}"
                                          for i, d in enumerate(ds)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
