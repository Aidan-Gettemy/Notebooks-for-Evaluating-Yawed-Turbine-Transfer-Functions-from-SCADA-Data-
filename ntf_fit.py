"""
Final NTF stage: fit models on ALL steered data (no holdout) at three quantile
levels (0.05, 0.50, 0.95) using the CV-selected hyperparameters, then evaluate on
an input grid to produce NTF curves + 90% prediction envelopes for plotting.

Two model backends:
  fit_ntf_xgb : gradient boosting, multi-quantile in ONE fit (quantile_alpha list)
                -> guarantees non-crossing quantiles (0.05 <= 0.50 <= 0.95).
  fit_ntf_lr  : linear quantile regression, one model per quantile.

predict_grid : evaluate a fitted NTF over a sweep of one feature with the others
               held fixed (e.g. sweep measured windvane; hold TI at dataset mean;
               one curve per power level) -- reproduces the paper's Fig 10/11/13.
to_ratio_grid: turn a predicted-QoI grid into a measured/predicted ratio grid.

Cosine-law baseline (single home for code that used to be defined four times):
  cosine_model, quantile_loss : P_gamma/P_0 = [beta0 +] |cos(u)|^alpha, pinball loss
  fit_cosine_quantiles        : full-data fit at several quantiles
  build_cosine_grid           : predict_grid-shaped table for plot_ntf_figure
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data prep: pull the steered rows, drop NaN target/feature (same policy as CV)
# --------------------------------------------------------------------------- #
def steered_xy(df, inputs, output, *, steer_col="targetwindvane", steer_thresh=0.1):
    """Return (X, y, kept_df) for rows where |steer_col| > steer_thresh, dropping
    rows with NaN in target or any feature (matches the CV harness's clean_fold)."""
    steered = df[df[steer_col].abs() > steer_thresh]
    X = steered[inputs].to_numpy(float)
    y = steered[output].to_numpy(float)
    m = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    return X[m], y[m], steered.iloc[m]


# --------------------------------------------------------------------------- #
# Final GB fit: all three quantiles in a single model (non-crossing)
# --------------------------------------------------------------------------- #
def fit_ntf_xgb(X, y, *, params, num_round, quantiles=(0.05, 0.50, 0.95)):
    """Fit ONE xgb model that outputs all quantiles simultaneously via a
    quantile_alpha list. Returns (booster, quantiles). Predicting yields an
    (n, n_quantiles) array with columns in `quantiles` order.

    Uses the CV-selected params (subsample, max_depth, eta) but overrides
    quantile_alpha with the full list. num_round is the CV-selected best.
    """
    import xgboost as xgb
    p = dict(params)
    p["objective"] = "reg:quantileerror"
    p["quantile_alpha"] = list(quantiles)     # list -> multi-quantile, non-crossing
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(p, dtrain, num_boost_round=num_round)
    return bst, tuple(quantiles)


def predict_xgb(bst, X):
    """Predict all quantiles. Returns (n, n_quantiles) array."""
    import xgboost as xgb
    out = bst.predict(xgb.DMatrix(X))
    return out if out.ndim == 2 else out[:, None]


# --------------------------------------------------------------------------- #
# Final LR fit: one linear quantile model per quantile
# --------------------------------------------------------------------------- #
def fit_ntf_lr(X, y, *, quantiles=(0.05, 0.50, 0.95), alpha=0.0):
    """Fit one QuantileRegressor per quantile. Returns (models_dict, quantiles).
    Linear quantiles CAN cross; we sort predictions at eval time to enforce order."""
    from sklearn.linear_model import QuantileRegressor
    models = {}
    for q in quantiles:
        models[q] = QuantileRegressor(quantile=q, alpha=alpha, solver="highs").fit(X, y)
    return models, tuple(quantiles)


def predict_lr(models, quantiles, X):
    """Predict each quantile; sort across quantiles per-row to prevent crossing."""
    cols = np.column_stack([models[q].predict(X) for q in quantiles])
    cols.sort(axis=1)                          # enforce monotone quantiles
    return cols


# --------------------------------------------------------------------------- #
# Grid prediction for plotting the NTF
# --------------------------------------------------------------------------- #
def predict_grid(predict_fn, inputs, *, sweep, sweep_values, held, series=None):
    """Evaluate a fitted NTF over a sweep of one feature.

    predict_fn   : callable X -> (n, n_quantiles) array (predict_xgb or predict_lr
                   wrapped to capture the model/quantiles).
    inputs       : the feature-name order the model was trained on.
    sweep        : name of the feature to vary along the x-axis (e.g. 'measured_windvane').
    sweep_values : 1-D array of x values to evaluate.
    held         : dict {feature: fixed value} for the non-swept, non-series features
                   (e.g. {'measured_ti': 0.09}).
    series       : optional (feature_name, [values]) to produce one curve per value
                   (e.g. ('measured_power', [500,1000,1500,2000])). If None, one curve.

    Returns a long dataframe: sweep value, series value (if any), and one column per
    quantile (q0.05, q0.5, q0.95).
    """
    def build_X(series_val=None):
        n = len(sweep_values)
        cols = []
        for f in inputs:
            if f == sweep:
                cols.append(np.asarray(sweep_values, float))
            elif series is not None and f == series[0]:
                cols.append(np.full(n, series_val, float))
            elif f in held:
                cols.append(np.full(n, held[f], float))
            else:
                raise ValueError(f"feature '{f}' not covered by sweep/series/held")
        return np.column_stack(cols)

    rows = []
    series_iter = series[1] if series is not None else [None]
    for sval in series_iter:
        X = build_X(sval)
        P = predict_fn(X)                      # (n, nq)
        nq = P.shape[1]
        rec = {sweep: np.asarray(sweep_values, float)}
        if series is not None:
            rec[series[0]] = np.full(len(sweep_values), sval, float)
        for j in range(nq):
            rec[f"q{j}"] = P[:, j]             # caller maps q0->0.05 etc by quantiles order
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Ratio grid for windspeed / power NTFs (paper Section 4.3, Fig 11 & 13)
# --------------------------------------------------------------------------- #
def to_ratio_grid(grid, *, series_col, quantiles=(0.05, 0.50, 0.95)):
    """Convert a predicted-QoI grid into a RATIO grid:

        ratio(vane) = (fixed measured QoI = series level) / (predicted-unyawed QoI)

    The numerator is the series level itself (e.g. windspeed held at 7.5 m/s), held
    constant across the vane sweep; the denominator is the NTF's predicted unyawed
    QoI at each vane angle and each quantile. Because dividing by the prediction
    inverts quantile order (larger prediction -> smaller ratio), the quantile
    columns are re-sorted so q0 <= q1 <= q2 remains the low/median/high envelope.

    Expects `grid` from predict_grid with columns: sweep feature, series_col, q0/q1/q2.
    Returns a grid of identical shape with q0/q1/q2 now holding the RATIO quantiles.
    """
    g = grid.copy()
    qcols = [f"q{i}" for i in range(len(quantiles))]
    num = g[series_col].to_numpy(float)                    # numerator = held QoI level
    denom = g[qcols].to_numpy(float)                       # (n, nq) predicted quantiles
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num[:, None] / denom                       # broadcast numerator over quantiles
    ratio.sort(axis=1)                                     # re-order low->high after inversion
    for j, c in enumerate(qcols):
        g[c] = ratio[:, j]
    return g


# --------------------------------------------------------------------------- #
# Cosine-law baseline (paper's comparison model for the power NTF)
# --------------------------------------------------------------------------- #
def cosine_model(params, u):
    """Cosine-law power ratio P_gamma/P_0, u in RADIANS.

    params = (alpha, beta0) -> beta0 + |cos(u)|^alpha   (offset form, current)
    params = alpha (scalar) -> |cos(u)|^alpha           (original one-parameter form)
    """
    p = np.atleast_1d(np.asarray(params, dtype=float))
    alpha = p[0]
    beta0 = p[1] if p.size > 1 else 0.0
    return beta0 + np.abs(np.cos(u)) ** alpha


def quantile_loss(params, u, y, tau):
    """Pinball (quantile) loss of cosine_model at quantile tau."""
    resid = y - cosine_model(params, u)
    return np.sum(np.maximum(tau * resid, (tau - 1) * resid))


def fit_cosine_quantiles(vane_deg, ratio, *, quantiles=(0.05, 0.50, 0.95),
                         with_offset=True, vane_limit_deg=90.0):
    """Fit the cosine law to (vane, P_gamma/P_0 ratio) data at each quantile.

    vane_deg      : wind vane in DEGREES (converted to radians internally).
    ratio         : measured power / consensus (unyawed) power.
    with_offset   : True  -> fit (alpha, beta0), x0=[1, 0], alpha >= 0, beta0 free
                    False -> fit alpha only,     x0=1,      alpha >= 0
    vane_limit_deg: rows with |vane| >= this are dropped (the old |u| < pi/2 guard).

    Returns {tau: params}, where params is (alpha, beta0) with offset or a float
    alpha without, ready for cosine_model / build_cosine_grid.
    """
    from scipy.optimize import minimize

    u = np.deg2rad(np.asarray(vane_deg, dtype=float))
    y = np.asarray(ratio, dtype=float)
    good = np.isfinite(u) & np.isfinite(y) & (np.abs(u) < np.deg2rad(vane_limit_deg))
    u, y = u[good], y[good]

    fits = {}
    for tau in quantiles:
        if with_offset:
            res = minimize(quantile_loss, x0=[1.0, 0.0], args=(u, y, tau),
                           bounds=[(0, None), (None, None)])  # alpha >= 0; beta0 free
            fits[tau] = tuple(res.x)
        else:
            res = minimize(quantile_loss, x0=1.0, args=(u, y, tau), bounds=[(0, None)])
            fits[tau] = float(res.x[0])
    return fits


def build_cosine_grid(fits, sweep_values, series_col, series_values,
                      sweep_col="windvane", quantile_order=(0.05, 0.50, 0.95)):
    """Predict_grid-shaped table from already-fitted cosine-law quantile params.
    Broadcasts the same q0/q1/q2 curve across every series level, since this
    model only depends on the sweep angle. Values are already ratios, so do NOT
    pass the result through to_ratio_grid."""
    u = np.deg2rad(sweep_values)
    preds = {f"q{i}": cosine_model(fits[tau], u) for i, tau in enumerate(quantile_order)}

    return pd.concat(
        [pd.DataFrame({sweep_col: sweep_values, series_col: lvl, **preds})
         for lvl in series_values],
        ignore_index=True,
    )
