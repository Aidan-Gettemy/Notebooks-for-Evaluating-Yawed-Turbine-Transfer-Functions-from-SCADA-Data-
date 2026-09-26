"""
Unified cross-validation harness for NTF model comparison.
(Merged: replaces both cv_harness.py and cv_harness_updated.py, and absorbs the
time_series_kfold splitter that used to live in updated_ntf_code.ipynb.)

  time_series_kfold -> blocked (contiguous-in-time) k-fold train/validation splits

Linear regression, gradient boosting, and the quantile-loss cosine NTF model are
all scored through the SAME code:
  clean_fold      -> NaN policy (drop NaN target OR any feature), applied identically
  cv_evaluate     -> per-fold + mean RMSE and R2 for any make_predictor callback
Selection metric is RMSE (the paper's criterion); R2 is reported alongside.

The cosine model (make_cosine_quantile_predictor) fits/predicts in ratio space
internally but converts to/from raw power using a consensus-power input column,
so its RMSE/R2 land in the exact same power units as the LR/GB models.

GB hyperparameter tuning (tune_xgb_quantile) uses the additive-boosting trick to
sweep num_boost_round from a single fit per (subsample, depth). After tuning, build
a make_predictor for the winning config and run it through cv_evaluate so the GB and
LR final numbers come from identical scoring code.
"""

import numpy as np
import itertools


# --------------------------------------------------------------------------- #
# Blocked time-series folds
# --------------------------------------------------------------------------- #
def time_series_kfold(df, folds, *, time_col="ts"):
    """Split df into `folds` contiguous time blocks. Each block is used once as
    the validation fold; the remaining rows form the training fold.
    Returns (train_folds, valid_folds), two lists of DataFrames.

    Rows past folds * (len(df) // folds) (fewer than `folds` rows) are never in
    a validation block, same as before."""
    import pandas as pd

    df = df.sort_values(by=time_col).reset_index(drop=True)
    fold_size = len(df) // folds

    train_folds, valid_folds = [], []
    for f in range(folds):
        start_idx = f * fold_size
        end_idx = start_idx + fold_size
        valid_folds.append(df.iloc[start_idx:end_idx])
        train_folds.append(pd.concat([df.iloc[:start_idx], df.iloc[end_idx:]])
                           .reset_index(drop=True))
    return train_folds, valid_folds


# --------------------------------------------------------------------------- #
# Shared: fold cleaning + metrics + generic CV
# --------------------------------------------------------------------------- #
def clean_fold(train_n, test_n, inputs, output):
    """Extract arrays and drop rows with NaN in target OR any feature.
    Identical policy for every model so comparisons are on the same rows."""
    Xtr = train_n[inputs].to_numpy(float); ytr = train_n[output].to_numpy(float)
    Xte = test_n[inputs].to_numpy(float);  yte = test_n[output].to_numpy(float)
    mtr = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
    mte = ~np.isnan(yte) & ~np.isnan(Xte).any(axis=1)
    return Xtr[mtr], ytr[mtr], Xte[mte], yte[mte]


def _rmse(err):
    return float(np.sqrt(np.mean(err**2)))


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _wrap(err, circular):
    """Wrap error to [-180,180] for circular targets (windvane); else identity."""
    return (err + 180) % 360 - 180 if circular else err


def cv_evaluate(train_splits, test_splits, inputs, output, make_predictor,
                *, circular=False, verbose=True):
    """Blocked k-fold evaluation of ANY model.

    make_predictor(X_train, y_train) -> predict(X) -> y_pred.
    Returns dict with per-fold and mean RMSE and R2, plus per-fold n_test.
    This is the single scoring path used by both LR and the final GB model.
    """
    rmses, r2s, ns = [], [], []
    for fold_id, (tr, te) in enumerate(zip(train_splits, test_splits)):
        Xtr, ytr, Xte, yte = clean_fold(tr, te, inputs, output)
        if len(ytr) == 0 or len(yte) == 0:
            rmses.append(np.nan); r2s.append(np.nan); ns.append(0); continue

        predict = make_predictor(Xtr, ytr)
        yp = predict(Xte)

        err = _wrap(yp - yte, circular)
        rmses.append(_rmse(err))
        r2s.append(_r2(yte, yp))
        ns.append(len(yte))
        if verbose:
            print(f"Fold {fold_id}: RMSE = {rmses[-1]:.4f}  R2 = {r2s[-1]:.4f}  (n_test={ns[-1]})")

    out = {"rmse_folds": rmses, "r2_folds": r2s, "n_test": ns,
           "rmse": float(np.nanmean(rmses)), "r2": float(np.nanmean(r2s))}
    if verbose:
        print(f"\nMean RMSE: {out['rmse']:.4f} | Mean R2: {out['r2']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Model factories (make_predictor callbacks)
# --------------------------------------------------------------------------- #
def lr_median_predictor(Xtr, ytr):
    """50%-quantile (median) linear model, matching the paper's tuning criterion."""
    from sklearn.linear_model import QuantileRegressor
    m = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(Xtr, ytr)
    return m.predict


def lr_mean_predictor(Xtr, ytr):
    """Ordinary least squares (conditional mean), if you want the mean model."""
    from sklearn.linear_model import LinearRegression
    m = LinearRegression().fit(Xtr, ytr)
    return m.predict


def make_cosine_quantile_predictor(inputs, windvane_col, ref_col, *,
                                   quantile=0.5, degrees=True, vane_limit_deg=90.0,
                                   ratio_numerator="output", fit_log=None,
                                   consensus_col=None):
    """Median (or any quantile) cosine-law NTF power model, as a make_predictor
    callback -- scored through the SAME cv_evaluate/clean_fold path as LR/GB.

    The cosine law is a model for the ratio P_gamma/P_0 = beta0 + |cos(u)|^alpha
    (yawed power over unyawed power; <= ~1 and FALLING with |yaw|). The factory
    fits in ratio space and converts back to the units of `output`, so RMSE/R2
    are directly comparable with the LR/GB runs.

    ref_col is the other power column (must be in `inputs`). Which one is the
    numerator depends on what `output` is:

      ratio_numerator="output" (default; the original behaviour)
          output = measured/yawed power, ref_col = consensus power
          fit  ratio = y / ref ;  predict  y = f(u) * ref

      ratio_numerator="ref"
          output = consensus/unyawed power, ref_col = measured/yawed power
          fit  ratio = ref / y ;  predict  y = ref / f(u)
          -> use this for the NTF setup in updated_ntf_code.ipynb
             (output="consensus_power_shifted", ref_col="power"), so the model
             is fit to power / consensus, the same ratio as the full-data fit
             and the figure. With "output" in that setup the law is fit to
             consensus / power, which rises with yaw and which |cos u|^alpha
             cannot follow.

    Parameters
    ----------
    inputs : list[str]
        The SAME `inputs` list passed to cv_evaluate -- used only to look up
        which column of Xtr/Xte is the windvane and which is ref_col.
    degrees : bool
        True if the windvane column is in degrees.
    vane_limit_deg : float
        Training rows with |windvane| >= this are dropped before fitting. Test
        rows are NOT filtered, so every fold is scored on the same rows as LR/GB.
    fit_log : list or None
        If given, each fold's fitted (alpha, beta0, tau, n_train) dict is appended.
    consensus_col : str, deprecated
        Old name for ref_col (kept so the old keyword call still works).
    """
    from scipy.optimize import minimize
    from ntf_fit import cosine_model, quantile_loss

    if consensus_col is not None:
        ref_col = consensus_col
    if ratio_numerator not in ("output", "ref"):
        raise ValueError("ratio_numerator must be 'output' or 'ref'")

    vane_idx = inputs.index(windvane_col)
    ref_idx = inputs.index(ref_col)
    vane_limit = np.deg2rad(vane_limit_deg) if degrees else vane_limit_deg

    def _factory(Xtr, ytr):
        vane_tr = Xtr[:, vane_idx]
        ref_tr = Xtr[:, ref_idx]
        u_tr = np.deg2rad(vane_tr) if degrees else vane_tr
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_tr = ytr / ref_tr if ratio_numerator == "output" else ref_tr / ytr

        good = np.isfinite(u_tr) & np.isfinite(ratio_tr) & (np.abs(u_tr) < vane_limit)
        res = minimize(
            quantile_loss, x0=[1.0, 0.0], args=(u_tr[good], ratio_tr[good], quantile),
            bounds=[(0, None), (None, None)],
        )
        alpha, beta0 = res.x
        if fit_log is not None:
            fit_log.append({"alpha": alpha, "beta0": beta0, "tau": quantile,
                            "n_train": int(good.sum())})

        def _predict(Xte):
            vane_te = Xte[:, vane_idx]
            ref_te = Xte[:, ref_idx]
            u_te = np.deg2rad(vane_te) if degrees else vane_te
            ratio_pred = cosine_model((alpha, beta0), u_te)
            if ratio_numerator == "output":
                return ratio_pred * ref_te
            return ref_te / ratio_pred

        return _predict

    return _factory


def make_xgb_predictor(params, num_round):
    """Build a make_predictor for a FIXED xgb quantile config, so the final GB model
    is scored by the same cv_evaluate as LR."""
    import xgboost as xgb

    def _factory(Xtr, ytr):
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        bst = xgb.train(params, dtrain, num_boost_round=num_round)
        def _predict(Xte):
            return bst.predict(xgb.DMatrix(Xte))
        return _predict
    return _factory


# --------------------------------------------------------------------------- #
# GB tuning with additive-boosting round sweep (selection on RMSE)
# --------------------------------------------------------------------------- #
def tune_xgb_quantile(train_splits, test_splits, inputs, output, *,
                      round_grid, subsample_grid, max_depth_grid,
                      learning_rate=0.3,
                      quantile=0.5, device="cpu", circular=False, verbose=True):
    """Grid search. For each (subsample, depth): fit ONE booster per fold to
    max(round_grid), then read validation RMSE at each round via iteration_range.
    Selection = lowest mean-over-folds RMSE. Returns (results, best).

    Cleans folds with the SAME clean_fold policy as cv_evaluate, so tuning and the
    LR comparison see identical rows.
    """
    import xgboost as xgb

    # clean once, build DMatrices once
    fold_mats = []
    for tr, te in zip(train_splits, test_splits):
        Xtr, ytr, Xte, yte = clean_fold(tr, te, inputs, output)
        fold_mats.append({
            "dtrain": xgb.DMatrix(Xtr, label=ytr),
            "dvalid": xgb.DMatrix(Xte),
            "yvalid": yte,
        })

    max_round = max(round_grid)
    checkpoints = sorted(round_grid)
    results = []

    for subsample, max_depth in itertools.product(subsample_grid, max_depth_grid):
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": quantile,
            "device": device,
            "subsample": subsample,
            "max_depth": max_depth,
            "eta": learning_rate,
        }
        # one fit per fold to the max rounds
        boosters = [xgb.train(params, fm["dtrain"], num_boost_round=max_round)
                    for fm in fold_mats]

        # read each checkpoint from the same boosters (first r trees == r-round model)
        for r in checkpoints:
            fold_rmses = []
            for fm, bst in zip(fold_mats, boosters):
                yp = bst.predict(fm["dvalid"], iteration_range=(0, r))
                err = _wrap(yp - fm["yvalid"], circular)
                fold_rmses.append(_rmse(err))
            results.append({
                "num_round": r, "subsample": subsample, "max_depth": max_depth,
                "params": params, "rmse": float(np.mean(fold_rmses)),
                "rmse_folds": fold_rmses,
            })
            if verbose:
                print(f"sub={subsample:<6} depth={max_depth} round={r:4d}  "
                      f"RMSE={results[-1]['rmse']:.4f}")

    best = min(results, key=lambda d: d["rmse"])
    return results, best
