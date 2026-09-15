"""PDM-score trajectories produced on device.

Runs in $NAVSIM_PY. Reuses navsim's scorer unchanged -- the only difference
from scripts/eval_navtest_v1.sh is that the trajectories come from a file
instead of from a forward pass, so the score is comparable to the 0.9222 that
script produced for the PyTorch reference.
"""

import argparse
import dataclasses
import os
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

DEVKIT = pathlib.Path(os.environ["NAVSIM_DEVKIT_ROOT"])
EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
sys.path.insert(0, str(DEVKIT))

# navsim_v1's Trajectory, not navsim.common's -- pdm_score is annotated on the
# former and rejects the latter with a TypeError.
from navsim.navsim_v1.common.dataclasses import Trajectory  # noqa: E402
from navsim.navsim_v1.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.navsim_v1.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer  # noqa: E402
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--metric-cache", default=str(EXP / "metric_cache_navtestv1"))
    ap.add_argument("--out", default=str(EXP / "tt_navtest_v1.csv"))
    a = ap.parse_args()

    traj = torch.load(a.traj, map_location="cpu", weights_only=False)
    print(f"  loaded {len(traj)} trajectories")

    with initialize_config_dir(config_dir=str(
            DEVKIT / "navsim/planning/script/config/pdm_scoring"), version_base=None):
        cfg = compose(config_name="default_run_pdm_score_fast_v1",
                      overrides=["experiment_name=tt_score", "train_test_split=navtest"])
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    mcl = MetricCacheLoader(pathlib.Path(a.metric_cache))
    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    rows: List[Dict[str, Any]] = []
    toks = [t for t in traj if t in mcl.tokens]
    print(f"  {len(toks)} tokens in common with the metric cache")
    for i, tok in enumerate(toks):
        mc = mcl.get_from_token(tok)
        t = Trajectory(traj[tok].numpy().astype(np.float32), sampling)
        try:
            s = pdm_score(metric_cache=mc, model_trajectory=t,
                          future_sampling=simulator.proposal_sampling,
                          simulator=simulator, scorer=scorer)
            # pdm_score returns PDMResults, a dataclass
            row = dataclasses.asdict(s) if dataclasses.is_dataclass(s) else dict(s)
            row["token"] = tok
            row.setdefault("valid", True)
            rows.append(row)
        except Exception as e:
            rows.append({"token": tok, "valid": False, "score": float("nan"),
                         "err": type(e).__name__})
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(toks)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    valid = int(df["valid"].sum()) if "valid" in df else len(df)
    if valid == 0:
        print(f"\n  !! 0 valid of {len(df)} -- every score failed")
        if "err" in df:
            print("     exceptions:", df["err"].value_counts().to_dict())
        return 1
    num = df.select_dtypes("number")
    print()
    print(f"  scored {len(df)}   valid {int(df.get('valid', pd.Series([True]*len(df))).sum())}")
    for c in num.columns:
        if c not in ("Unnamed: 0",):
            print(f"    {c:34s} {num[c].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
