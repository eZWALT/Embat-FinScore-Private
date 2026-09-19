"""Checks for the score forecast: no look-ahead, valid fans, drivers that add up, short trails and gaps.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m analysis.monitor.forecast_test

Uses the committed forecast_params.json and synthetic score panels, so it needs no data.
"""
from __future__ import annotations

import json

import numpy as np

from . import forecast as fc


def _panel(seed: int = 1, n: int = 60, T: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S = np.clip(65 + np.cumsum(rng.normal(0, 4, (n, T)), axis=1) * 0.5 + rng.normal(0, 3, (n, T)), 0, 100)
    for i in range(n):
        S[i, : rng.integers(0, 10)] = np.nan
    return S


def main() -> int:
    params = json.loads(fc.PARAMS_PATH.read_text(encoding="utf8"))
    centre, typical = params["level_centre"], params["typical_change_scale"]
    S = _panel()
    F = fc.build_features(S, centre, typical)

    # 1. no look-ahead: features at month t are the same whether or not later months exist
    for t in (8, 13, 20):
        Ft = fc.build_features(S[:, : t + 1], centre, typical)
        for name in fc.FEATURES:
            assert np.allclose(Ft[name], F[name][:, : t + 1], equal_nan=True), f"feature {name} at t={t} depends on later months"
    print("ok  features use only months up to the origin")

    # 2. the fan for a truncated panel equals the full-panel fan at that origin (same company, same origin)
    t = 15
    full = fc.forecast_rows(S[:, : t + 1], params)
    cut = fc.forecast_rows(np.concatenate([S[:, : t + 1], np.full((S.shape[0], 5), np.nan)], axis=1), params)
    for a, b in zip(full, cut):
        assert (a is None) == (b is None)
        if a:
            assert all(abs(p["median"] - q["median"]) < 1e-9 for p, q in zip(a["points"], b["points"])), "trailing empty months change the fan"
    print("ok  trailing empty months do not change the fan")

    # 3. fans are ordered, inside 0-100, and drivers add up to the 3-month median change (before clipping)
    rows = fc.forecast_rows(S, params)
    n_fans = 0
    for i, r in enumerate(rows):
        if r is None:
            assert np.isfinite(S[i]).sum() < fc.MIN_HISTORY
            continue
        n_fans += 1
        for p in r["points"]:
            v = [p["lo80"], p["lo50"], p["median"], p["hi50"], p["hi80"]]
            assert v == sorted(v) and 0 <= v[0] and v[-1] <= 100, f"company {i}: fan not ordered or outside 0-100"
        d = r["drivers"]
        if d:
            assert abs(sum(d["parts"].values()) - d["expected_change"]) < 1e-9
            m3 = r["points"][2]["median"]
            assert abs(np.clip(r["naive_last"] + d["expected_change"], 0, 100) - m3) < 1e-6, "drivers do not reproduce the 3-month median"
    assert n_fans > 0
    print(f"ok  {n_fans} fans ordered inside 0-100, drivers add up to the 3-month median")

    # 4. short trail gives no forecast; an interior gap falls back to the naive fan instead of failing
    short = np.full((1, 24), np.nan)
    short[0, -3:] = [60, 61, 62]
    assert fc.forecast_rows(short, params) == [None]
    gap = np.full((1, 24), np.nan)
    gap[0, 8:] = np.linspace(60, 70, 16)
    gap[0, 17] = np.nan
    r = fc.forecast_rows(gap, params)[0]
    assert r is not None and all(p["median"] == p["median"] for p in r["points"])
    print("ok  short trail -> None, gap in the trail -> a fan")

    # 5. the pull: far below its own average -> expected change up, far above -> down, holding level and history equal
    base = np.tile(np.array([70.0, 72, 68, 71, 70, 69, 72, 70, 71, 70, 69, 71]), (2, 1))
    base[0, -1], base[1, -1] = 40.0, 95.0
    low, high = fc.forecast_rows(np.concatenate([np.full((2, 12), np.nan), base], axis=1), params)
    assert low["drivers"]["parts"]["own_average"] > 0 > high["drivers"]["parts"]["own_average"], "own-average pull has the wrong sign"
    assert low["points"][5]["median"] > 40.0 and high["points"][5]["median"] < 95.0, "6-month median does not move toward the own average"
    print("ok  the fan is pulled toward the company's own average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
