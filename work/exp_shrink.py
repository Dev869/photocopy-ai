"""Experiment: shrink scene-smoothed kNN medians toward an era prior.

pred = alpha * knn_scene_median + (1 - alpha) * era_prior
era_prior = per-slider median over the 5 weddings nearest in time (excl. holdout).
Tune alpha on 2025-01-04, validate on 2024-12-30.
"""
import statistics, sys

from predict import load_index
from scenes import predict_scene_smoothed

KEYS = ("Exposure2012 Temperature Contrast2012 Highlights2012 Shadows2012 "
        "Whites2012 Blacks2012 Vibrance Saturation").split()


def era_prior(recs, ids, holdout_date, n_weddings=5):
    dates = sorted({(recs[x].get("capture_date") or "")[:10] for x in ids} - {""})
    dates = [d for d in dates if d != holdout_date]
    near = sorted(dates, key=lambda d: abs((int(d[:4]) * 372 + int(d[5:7]) * 31 + int(d[8:10]))
                                           - (int(holdout_date[:4]) * 372 + int(holdout_date[5:7]) * 31
                                              + int(holdout_date[8:10]))))[:n_weddings]
    pool = [recs[x] for x in ids if (recs[x].get("capture_date") or "")[:10] in near]
    return {k: statistics.median(float(r["settings"][k]) for r in pool if k in r["settings"])
            for k in KEYS}


def evaluate(date, alphas):
    idx = load_index()
    ids, recs = idx["ids"], idx["recs"]
    hold = [i for i, x in enumerate(ids) if (recs[x].get("capture_date") or "").startswith(date)]
    hold_ids = {ids[i] for i in hold}
    _per, smoothed, _sc = predict_scene_smoothed(idx, hold, exclude=hold_ids)
    prior = era_prior(recs, [x for x in ids if x not in hold_ids], date)
    for a in alphas:
        errs = {k: [] for k in KEYS}
        for i in hold:
            truth = recs[ids[i]]["settings"]
            for k in KEYS:
                if k in truth and k in smoothed.get(i, {}):
                    pred = a * smoothed[i][k] + (1 - a) * prior[k]
                    errs[k].append(abs(float(truth[k]) - pred))
        line = "  ".join(f"{k.replace('2012','')} {statistics.mean(v):.2f}"
                         for k, v in errs.items() if v)
        print(f"alpha={a}: {line}")


if __name__ == "__main__":
    for date, alphas in (("2025-01-04", (1.0, 0.7, 0.5, 0.3, 0.0)),
                         ("2024-12-30", (1.0, 0.5,))):
        print(f"== {date} ==")
        evaluate(date, alphas)
