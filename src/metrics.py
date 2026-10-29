"""Evaluation metrics for next-move forecasts."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)


def error_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> dict:
    """MAE / RMSE in price units, plus the same in pips (1 pip = 0.0001)."""
    yt, yp = np.ravel(y_true_price), np.ravel(y_pred_price)
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"mae": mae, "rmse": rmse, "mae_pips": mae / 1e-4, "rmse_pips": rmse / 1e-4}


def next_move_metrics(actual_ret: np.ndarray, score: np.ndarray) -> dict:
    """Directional metrics for predicting the next real 1-minute move.

    `actual_ret` is the realised next-minute return; `score` is the model's
    predicted return (or any score whose sign is the predicted direction).
    Minutes with no price change are excluded, and their share is reported.
    """
    actual_ret, score = np.ravel(actual_ret), np.ravel(score)
    moved = actual_ret != 0
    t = (actual_ret[moved] > 0).astype(int)
    s = score[moved]
    p = (s > 0).astype(int)
    out = {
        "directional_accuracy": accuracy_score(t, p),
        "cohen_kappa": cohen_kappa_score(t, p),
        "precision_up": precision_score(t, p, zero_division=0),
        "recall_up": recall_score(t, p, zero_division=0),
        "f1_up": f1_score(t, p, zero_division=0),
        "share_up_predicted": float(p.mean()),
        "share_up_actual": float(t.mean()),
        "share_unchanged_minutes": float(1 - moved.mean()),
        "n": int(moved.sum()),
    }
    out["roc_auc"] = (roc_auc_score(t, s) if len(np.unique(s)) > 2 else float("nan"))
    # 95% interval for accuracy under a binomial approximation
    se = np.sqrt(out["directional_accuracy"] * (1 - out["directional_accuracy"]) / out["n"])
    out["accuracy_ci95"] = [out["directional_accuracy"] - 1.96 * se,
                            out["directional_accuracy"] + 1.96 * se]
    return out
