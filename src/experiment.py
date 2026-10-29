"""Can anything predict the next 1-minute EUR/USD move?

Usage:
    python -m src.experiment --data data/DAT_XLSX_EURUSD_M1_2023.xlsx

Evaluation design:
  * Target: the next real 1-minute log return.
  * Inputs at time t use only bars up to t, so no future information leaks in.
  * Scale-free, stationary features; the scaler is fitted on the training period only.
  * Outliers are clipped rather than deleted, so every window covers consecutive bars;
    targets more than 5 minutes after the last bar (weekends, holidays) are excluded.
  * Train / validation / test split by time (70 / 10 / 20).
  * Every model is compared with naive baselines on the same test minutes.
  * ARIMA(5,1,0) makes one-step forecasts with fresh data, like the LSTM.
  * Errors in price units and pips, Cohen's kappa, ROC-AUC from continuous scores.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import build_features, load_histdata
from .metrics import error_metrics, next_move_metrics

LOOK_BACK = 60
CLIP = 8.0  # clip standardised features at +/- 8 standard deviations
MAX_GAP_SECONDS = 300  # HistData omits minutes without ticks; skip weekend/holiday gaps


def prepare(df: pd.DataFrame):
    feats = build_features(df)
    close = df["Close"]
    target = np.log(close.shift(-1) / close)            # return from bar t to the next bar
    next_gap = (df.index.to_series().shift(-1) - df.index.to_series()).dt.total_seconds()
    valid = feats.notna().all(axis=1) & target.notna() & (next_gap <= MAX_GAP_SECONDS)
    return feats, target, close, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()
    t0 = time.time()

    df = load_histdata(args.data)
    feats, target, close, valid = prepare(df)
    n = len(df)
    first_ok = int(np.argmax(feats.notna().all(axis=1).to_numpy())) + LOOK_BACK
    rows = np.where(valid.to_numpy() & (np.arange(n) >= first_ok))[0]

    cut_train, cut_val = rows[int(len(rows) * 0.7)], rows[int(len(rows) * 0.8)]
    train_rows = rows[rows < cut_train]
    val_rows = rows[(rows >= cut_train) & (rows < cut_val)]
    test_rows = rows[rows >= cut_val]

    # Standardise with training-period statistics only, then clip extremes
    F = feats.to_numpy(dtype="float64")
    scaler = StandardScaler().fit(F[train_rows])
    Fs = np.clip(scaler.transform(np.nan_to_num(F)), -CLIP, CLIP).astype("float32")
    y = target.to_numpy()
    y_std = y[train_rows].std()
    c = close.to_numpy()

    actual = y[test_rows]
    next_close, last_close = c[test_rows + 1], c[test_rows]
    res = {
        "rows_loaded": n,
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "test_period": [str(df.index[test_rows[0]]), str(df.index[test_rows[-1]])],
        "mean_abs_move_pips_test": float(np.mean(np.abs(next_close - last_close)) / 1e-4),
        "models": {},
    }

    def record(name, score, pred_ret=None):
        entry = next_move_metrics(actual, score) if score is not None else {}
        if pred_ret is not None:
            entry.update(error_metrics(next_close, last_close * np.exp(pred_ret)))
        res["models"][name] = entry

    # --- Naive baselines ---------------------------------------------------------
    last_ret = y[test_rows - 1]  # the return that just happened (t-1 -> t)
    record("Random walk (next price = last price)", None, np.zeros_like(actual))  # errors only
    record("Momentum (repeat last move)", last_ret)
    record("Reversal (opposite of last move)", -last_ret)
    up_share = float((y[train_rows] > 0).mean() / max((y[train_rows] != 0).mean(), 1e-9))
    record("Majority class from training", np.full_like(actual, 1.0 if up_share > 0.5 else -1.0))

    # --- Linear model on the latest feature row ----------------------------------
    ridge = Ridge(alpha=1.0).fit(Fs[train_rows], y[train_rows] / y_std)
    record("Ridge regression (same features)", ridge.predict(Fs[test_rows]),
           ridge.predict(Fs[test_rows]) * y_std)

    # --- ARIMA(5,1,0), one-step-ahead with fresh data ------------------------------
    from statsmodels.tsa.arima.model import ARIMA
    t1 = time.time()
    span_train = np.log(c[first_ok:cut_train])
    arima = ARIMA(span_train, order=(5, 1, 0)).fit()
    full = arima.apply(np.log(c[first_ok:])).fittedvalues   # one-step predictions of log price
    # fittedvalues[k] predicts log price at position k from data up to k-1
    pred_next_log = full[test_rows + 1 - first_ok]
    arima_ret = pred_next_log - np.log(last_close)
    record("ARIMA(5,1,0) one-step", arima_ret, arima_ret)
    res["arima_params"] = dict(zip(arima.model.param_names, map(float, arima.params)))
    res["arima_seconds"] = round(time.time() - t1, 1)

    # --- LSTM ----------------------------------------------------------------------
    if not args.skip_lstm:
        import tensorflow as tf
        from tensorflow import keras
        from numpy.lib.stride_tricks import sliding_window_view

        win = sliding_window_view(Fs, LOOK_BACK, axis=0)  # win[j] covers rows j..j+59
        def windows(r):                                 # window ending at row r (inclusive)
            return np.ascontiguousarray(win[r - LOOK_BACK + 1].transpose(0, 2, 1))
        Xtr, Xva, Xte = windows(train_rows), windows(val_rows), windows(test_rows)
        ytr, yva = (y[train_rows] / y_std).astype("float32"), (y[val_rows] / y_std).astype("float32")

        seed_scores, seed_logs = [], []
        for seed in args.seeds:
            t2 = time.time()
            tf.keras.utils.set_random_seed(seed)
            model = keras.Sequential([
                keras.Input(shape=Xtr.shape[1:]),
                keras.layers.LSTM(args.units),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ])
            model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber())
            hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs,
                             batch_size=256, verbose=2,
                             callbacks=[keras.callbacks.EarlyStopping(
                                 monitor="val_loss", patience=3, restore_best_weights=True)])
            s = model.predict(Xte, batch_size=2048, verbose=0).ravel()
            seed_scores.append(s)
            record(f"LSTM seed {seed}", s, s * y_std)
            seed_logs.append({"seed": seed, "epochs": len(hist.history["loss"]),
                              "best_val_loss": float(min(hist.history["val_loss"])),
                              "seconds": round(time.time() - t2, 1)})
        ens = np.mean(seed_scores, axis=0)
        record("LSTM (average of seeds)", ens, ens * y_std)
        res["lstm_training"] = seed_logs
        np.save(Path(args.out).with_suffix(".lstm_scores.npy"), ens)

    np.save(Path(args.out).with_suffix(".test_actual.npy"), actual)
    res["total_seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
