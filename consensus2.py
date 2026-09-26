"""
Consensus QoI estimation built for the LONG-format corrected_df:
    one row per (turbine, ts), columns include
    winddir, windspeed, windvane, windvane_cp, power, nacdir, targetwindvane, ti,
    plus filtering features (circ_dif_mean, circ_dif_std, ratio_std_comp, farm_z_wd,
    screened, seg, seg_offset, n_breaks, state).

Distances come from a locations table: columns [turbine_id, x_D, y_D] in rotor
diameters.

By default, no min-useful threshold: a consensus is produced wherever at least one
neighbor survives the masks (weight sum > 0); otherwise NaN. Pass min_neighbors=k to
estimate_consensus (also threaded through sweep/append_consensus) to instead require
at least k usable neighbors within the method's required neighbor set (e.g. the
nearest-k set for invdist) at that timestamp, else the estimate is NaN. We rely on
data volume + the four-slice error report to choose a method.

Post-consensus calibration (moved here from updated_ntf_code.ipynb):
    shift_consensus_windvane_by_turbine -> per-turbine static windvane offset
    calibrate_ratio_by_turbine          -> per-turbine median-regression ratio
                                           calibration for windspeed / power
    median_regression                   -> IRLS L1 (median) line fit used above

Methods:
    'global'   -> equal weights                       (param ignored)
    'gaussian' -> exp(-d^2 / (2 sigma^2))              (param = sigma, in D)
    'invdist'  -> 1/(d + dist_offset), nearest-k only  (param = k cluster size; 'all' = no
                                                         cap; dist_offset default 0.0 = classic
                                                         IDW; dist_offset > 0, in D, gives a
                                                         "damped"/modified-Shepard IDW that caps
                                                         any one neighbor's weight at 1/dist_offset
                                                         instead of letting 1/d diverge as d -> 0)

Raw IDW (dist_offset=0) has no free bandwidth parameter -- the relative weight
between any two neighbors is fixed purely by their distance ratio, so whichever
turbine happens to be nearest tends to dominate the average regardless of the
neighbor-count cutoff k. This shows up as a spurious RMSE advantage (the dominant
neighbor's noise correlates with the target's own noise at short range) and a low
effective sample size (see _effective_sample_size / the `ess` diagnostic). Use
`gaussian` (bandwidth sigma is a free, physically-settable parameter) or `invdist`
with dist_offset > 0 (bounds -- but doesn't eliminate -- the near-neighbor weight)
if you want the weighting to reflect genuine multi-turbine averaging.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# One-time preparation: pivot long -> wide matrices, build distance table
# --------------------------------------------------------------------------- #
def prepare(corrected_df, locations, *, vane_col="windvane_cp"):
    """Pivot the long frame into ts x turbine matrices for each field we need, and
    build a turbine distance lookup from the locations table. Returns a dict of
    matrices (all sharing the same ts index and turbine columns) plus metadata.

    Doing this once keeps the per-target estimation a cheap slice + weighted sum."""
    df = corrected_df.copy()
    # ensure ts is datetime and sorted
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["ts", "turbine"])

    # fields we pivot (value columns aggregated or read per target)
    fields = {
        "winddir": "winddir",
        "windspeed": "windspeed",
        "windvane": vane_col,            # corrected vane by default
        "windvane_meas": "windvane",     # raw measured vane (for filter 3 + reporting)
        "power": "power",
        "nacdir": "nacdir",
        "targetwindvane": "targetwindvane",
        "ti": "ti",
    }
    mats = {}
    for name, col in fields.items():
        if col in df.columns:
            mats[name] = df.pivot(index="ts", columns="turbine", values=col)

    # a canonical turbine column order + ts index (from any matrix)
    any_mat = next(iter(mats.values()))
    turbines = list(any_mat.columns)
    ts_index = any_mat.index

    # reindex every matrix to the same turbine order (safety)
    for k in mats:
        mats[k] = mats[k].reindex(columns=turbines)

    # distance lookup: dict[target] -> Series over `turbines` of distance in D
    loc_ids = locations["turbine_id"]
    if loc_ids.duplicated().any():
        dups = sorted(loc_ids[loc_ids.duplicated()].unique().tolist())
        raise ValueError(f"locations has duplicate turbine_id rows: {dups}. "
                         "Deduplicate before calling prepare().")

    # coerce id dtype to match the pivot's turbine column dtype so reindex aligns.
    # (pivot columns come from corrected_df['turbine']; if locations.turbine_id is a
    #  different int width or object, reindex silently yields NaN -> NaN distances ->
    #  gaussian/invdist weights all NaN while 'global' still works. This is the bug.)
    loc = locations.copy()
    try:
        loc["turbine_id"] = loc["turbine_id"].astype(type(turbines[0]))
    except (ValueError, TypeError):
        loc["turbine_id"] = loc["turbine_id"].astype("int64")
        turbines = [int(t) for t in turbines]

    loc = loc.set_index("turbine_id")[["x_D", "y_D"]]
    coords = loc.reindex(turbines)                    # align to matrix column order

    missing = [t for t, bad in zip(turbines, coords.isna().any(axis=1).to_numpy()) if bad]
    if missing:
        raise ValueError(
            f"{len(missing)} turbine(s) in corrected_df have no coordinates in locations: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}. "
            "Distance-based methods (gaussian, invdist) need every turbine's x_D/y_D. "
            "Add them to the locations table, or restrict corrected_df to located turbines."
        )

    XY = coords.to_numpy(dtype=float)                 # (Nturb, 2), now guaranteed finite
    dist = {}
    for j, t in enumerate(turbines):
        d = np.sqrt(((XY - XY[j]) ** 2).sum(axis=1))  # distance from t to every turbine
        dist[t] = pd.Series(d, index=turbines)

    return {
        "mats": mats,
        "turbines": turbines,
        "ts": ts_index,
        "dist": dist,
        "vane_col": vane_col,
    }


# --------------------------------------------------------------------------- #
# Weight builders
# --------------------------------------------------------------------------- #
def _base_weights(distances, method, param, dist_offset=0.0):
    """distances: 1-D array over turbines (target's own entry is 0). Returns base
    weights over turbines; the target's self-weight is zeroed by the caller.

    dist_offset (invdist only): d0 >= 0, in the same distance units as `distances`
    (rotor diameters). Weight becomes 1/(d + d0) instead of 1/d -- a "damped" or
    "modified Shepard" IDW. d0=0 (default) reproduces classic IDW exactly. A
    positive d0 removes the 1/d singularity and caps the maximum weight any single
    neighbor can receive at 1/d0, which bounds how much one very close turbine can
    dominate the average (see project notes on IDW nearest-neighbor domination).
    Distance ordering for the nearest-k cluster cutoff is unaffected by d0, since
    adding a constant to every distance doesn't change their rank order."""
    d = np.asarray(distances, dtype=float)
    if method == "global":
        w = np.ones_like(d)
    elif method == "gaussian":
        if param is None:
            raise ValueError("gaussian needs param=sigma")
        sigma = float(param)
        w = np.exp(-(d * d) / (2.0 * sigma * sigma))
    elif method == "invdist":
        dd = d + float(dist_offset) if dist_offset else d
        with np.errstate(divide="ignore"):
            w = 1.0 / dd
        w[~np.isfinite(w)] = 0.0                      # d==0, d0==0 (self) -> 0
        # cluster: keep only nearest-k by true distance (d0 doesn't change rank order)
        if param is not None and param != "all":
            k = int(param)
            order = np.argsort(d)                     # ascending; self (d=0) first
            keep = np.zeros_like(d, dtype=bool)
            # skip the self entry (distance 0) when counting k neighbors
            nbr_order = [i for i in order if d[i] > 0]
            keep[nbr_order[:k]] = True
            w = w * keep.astype(float)
    else:
        raise ValueError("method must be 'global', 'gaussian', or 'invdist'")
    return w


def _effective_sample_size(w):
    """Kish's effective sample size for a weight vector: (sum w)^2 / sum(w^2).
    Equals the true neighbor count for an equal-weight average and collapses
    toward 1 as weight concentrates on a single neighbor -- a direct, interpretable
    measure of how much genuine multi-turbine averaging a weighting scheme is
    actually doing, independent of RMSE/bias/corr against measured."""
    w = np.asarray(w, dtype=float)
    s1 = w.sum()
    s2 = (w * w).sum()
    if s2 == 0:
        return np.nan
    return float((s1 * s1) / s2)


# --------------------------------------------------------------------------- #
# Segment-aware temporal smoothing (variance reduction of the reference)
# --------------------------------------------------------------------------- #
def _segment_roll(ts_index, values, *, window, gap, min_periods, how="mean"):
    """Rolling smooth of a single turbine's time series that NEVER bridges a time
    gap larger than `gap`. Filtering leaves disjoint estimates in time; each
    contiguous run (gaps <= `gap` bridged, gaps > `gap` split) is smoothed
    independently so a window can't average across a removed period.
    Returns an array aligned with the ORIGINAL ts_index order."""
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(ts_index))
    order = np.argsort(s.index.values)
    s_sorted = s.iloc[order]

    valid = s_sorted.notna()
    out_sorted = np.full(len(s_sorted), np.nan)
    if valid.any():
        sv = s_sorted[valid]
        dt = sv.index.to_series().diff()
        seg = (dt > pd.Timedelta(gap)).cumsum()
        rolled = np.full(len(sv), np.nan)
        for _, idx in seg.groupby(seg).groups.items():
            sub = sv.loc[idx]
            r = sub.rolling(window, min_periods=min_periods)
            r = r.median() if how == "median" else r.mean()
            pos = [sv.index.get_loc(i) for i in idx]
            rolled[pos] = r.values
        out_sorted[np.where(valid.to_numpy())[0]] = rolled

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return out_sorted[inv]


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #
def estimate_consensus(prep, *, qoi, target, method, param=None, dist_offset=0.0,
                       valid_turbines=None,
                       steer_thresh=0.1, meas_vane_thresh=None,
                       min_neighbors=None,
                       smooth=False, smooth_window="10min", smooth_gap="5min",
                       smooth_min_periods=3, smooth_how="mean",
                       keep_raw_consensus=False):
    """Consensus estimate of `qoi` for `target` at every ts.

    (1) qoi: 'winddir' | 'windvane' | 'windspeed' | 'power'
    (2) steer_thresh: exclude a neighbor at times where |its targetwindvane| > this
    (3) meas_vane_thresh: exclude a neighbor at times where |its measured windvane| > this
    (4) valid_turbines: only these contribute; all others weight 0 (None = all)
    (5) target: turbine to estimate for; its own value never contributes
    (6) method, (7) param, dist_offset: see _base_weights. dist_offset only
        applies to method='invdist' -- weight becomes 1/(d + dist_offset) instead
        of 1/d, which bounds how much a single very close neighbor can dominate.
    (8) min_neighbors: minimum number of USABLE neighbors required, out of the
        target's required neighbor set, for a ts to get an estimate.

        The "required neighbor set" is whichever turbines end up with a nonzero
        base spatial weight for this target under (method, param, valid_turbines) --
        e.g. for invdist with param=k that's exactly the nearest-k turbines; for
        global/gaussian (or invdist param='all') it's every eligible turbine. At
        each ts we count how many turbines in that set actually survive the
        per-ts usability masks (value present, not steered, not high measured-yaw).
        If that count is < min_neighbors, the ts is left NaN even if weight sum > 0.
        None (default) keeps the old behavior: any ts with weight sum > 0 (i.e. at
        least 1 usable neighbor) gets an estimate.

    Temporal smoothing (lowers reference variance -> reduces the regression-to-the-
    mean artifact in the NTF; circular QoIs are smoothed on sin/cos, not degrees):
      smooth            : if True, segment-aware rolling-smooth the consensus.
      smooth_window     : time-based window, e.g. '10min'.
      smooth_gap        : gaps larger than this are NOT bridged (respects filtering
                          holes); gaps <= this are smoothed across. e.g. '5min'.
      smooth_min_periods: min points in a window to emit a value (else NaN).
      smooth_how        : 'mean' or 'median' central estimator.
      keep_raw_consensus: if True, also keep unsmoothed `consensus_{qoi}_raw`.

    Returns a per-ts dataframe of the target's own measured quantities + filtering
    features + `n_usable_neighbors` and `ess_neighbors` diagnostic columns + the
    consensus column `consensus_{qoi}` (smoothed if smooth=True). `ess_neighbors`
    is Kish's effective sample size (see _effective_sample_size) of the per-ts
    weight vector actually used -- how many turbines-worth of independent
    averaging the estimate reflects, vs. n_usable_neighbors which just counts how
    many had nonzero weight. out.attrs['base_ess'] gives the ts-invariant ESS of
    the required neighbor set's base weights alone (ignoring usability), i.e. the
    "nominal" concentration of the (method, param, dist_offset) choice itself.
    """
    mats = prep["mats"]
    turbines = prep["turbines"]
    ts_index = prep["ts"]
    target = int(target)
    if target not in turbines:
        raise ValueError(f"target {target} not in pivoted turbines")

    circular = qoi in ("winddir", "windvane")

    # For the windvane QoI we DON'T average neighbor vanes directly (their vanes are
    # relative to their OWN nacelles -> mixed reference frames). Instead we build a
    # consensus wind DIRECTION from neighbors (a global, common-frame quantity) and
    # then take the circular difference against the TARGET's nacelle direction to get
    # the target's relative vane. So the neighbor value matrix used for aggregation is
    # winddir even when qoi == 'windvane'.
    agg_field = "winddir" if qoi == "windvane" else qoi

    V = mats[agg_field].to_numpy(dtype=float)         # (T, N) neighbor values to aggregate
    tcol = turbines.index(target)

    # --- base spatial weights over turbines ------------------------------- #
    d = prep["dist"][target].to_numpy(dtype=float)
    w = _base_weights(d, method, param, dist_offset=dist_offset)  # (N,)
    w[tcol] = 0.0                                      # never include self (5)

    # valid-turbine restriction (4)
    if valid_turbines is not None:
        vt = set(int(x) for x in valid_turbines)
        valid_mask = np.array([t in vt for t in turbines], dtype=float)
        w = w * valid_mask

    # --- per-(ts,neighbor) usability masks -------------------------------- #
    usable = ~np.isnan(V)                             # value present

    if steer_thresh is not None and "targetwindvane" in mats:
        ST = mats["targetwindvane"].to_numpy(dtype=float)
        usable &= ~(np.abs(ST) > steer_thresh)        # (2) drop steered neighbors
    if meas_vane_thresh is not None and "windvane_meas" in mats:
        MV = mats["windvane_meas"].to_numpy(dtype=float)
        usable &= ~(np.abs(MV) > meas_vane_thresh)    # (3) drop high measured-yaw neighbors

    # --- weighted aggregation --------------------------------------------- #
    W = np.broadcast_to(w, V.shape).copy()
    W[~usable] = 0.0
    wsum = W.sum(axis=1)

    # required neighbor set = turbines with nonzero base weight (fixed per target,
    # doesn't vary by ts): the nearest-k for invdist(k), all eligible turbines for
    # global/gaussian/invdist('all'), intersected with valid_turbines if given.
    required_set = w > 0                               # (N,), ts-invariant
    n_usable = (usable & required_set[None, :]).sum(axis=1)  # (T,) per-ts count
    base_ess = _effective_sample_size(w)               # ts-invariant nominal ESS

    ok = wsum > 0                                      # at least one neighbor survives
    if min_neighbors is not None:
        ok &= (n_usable >= int(min_neighbors))         # (8) require k usable in the set

    # per-ts effective sample size of the weights actually used (post usability
    # masking) -- how many turbines-worth of independent averaging this ts reflects
    with np.errstate(invalid="ignore", divide="ignore"):
        ess = np.where(wsum > 0, (wsum ** 2) / np.sum(W ** 2, axis=1), np.nan)

    consensus = np.full(V.shape[0], np.nan)
    if circular:
        rad = np.deg2rad(np.where(usable, V, 0.0))
        S = np.nansum(np.sin(rad) * W, axis=1)
        C = np.nansum(np.cos(rad) * W, axis=1)
        cm = np.rad2deg(np.arctan2(S, C))             # consensus wind direction (deg)
        if qoi == "windvane":
            # target relative vane = circular diff (consensus winddir - target nacdir)
            nac = mats["nacdir"].to_numpy(dtype=float)[:, tcol]
            vane = (cm - nac + 180.0) % 360.0 - 180.0
            consensus[ok] = vane[ok]
        else:
            consensus[ok] = cm[ok]
    else:
        num = np.nansum(np.where(usable, V, 0.0) * W, axis=1)
        consensus[ok] = num[ok] / wsum[ok]

    # --- optional temporal smoothing (segment-aware) ---------------------- #
    consensus_raw = consensus.copy()
    if smooth:
        if circular:
            # smooth the unit vector, not degrees (avoids the +-180 wrap bug)
            ang = np.deg2rad(consensus)
            sin_s = _segment_roll(ts_index, np.sin(ang), window=smooth_window,
                                  gap=smooth_gap, min_periods=smooth_min_periods, how="mean")
            cos_s = _segment_roll(ts_index, np.cos(ang), window=smooth_window,
                                  gap=smooth_gap, min_periods=smooth_min_periods, how="mean")
            consensus = np.rad2deg(np.arctan2(sin_s, cos_s))
        else:
            consensus = _segment_roll(ts_index, consensus, window=smooth_window,
                                      gap=smooth_gap, min_periods=smooth_min_periods,
                                      how=smooth_how)

    # --- assemble the target's own row-wise quantities -------------------- #
    def col(name):
        return mats[name].to_numpy(dtype=float)[:, tcol] if name in mats else np.full(V.shape[0], np.nan)

    out = pd.DataFrame(index=ts_index)
    out["ts"] = ts_index
    out["turbine"] = target
    # measured value of the QoI at the target: for vane use the target's own
    # (corrected) measured vane; for others use that QoI's own measurement.
    meas_field = "windvane" if qoi == "windvane" else agg_field
    out[f"measured_{qoi}"] = col(meas_field)
    out["measured_windvane"] = col("windvane")        # corrected vane at target
    out["measured_windvane_raw"] = col("windvane_meas")
    out["targetwindvane"] = col("targetwindvane")     # the set/target vane (steering command)
    out["measured_winddir"] = col("winddir")
    out["measured_windspeed"] = col("windspeed")
    out["measured_power"] = col("power")
    out["nacdir"] = col("nacdir")
    out["ti"] = col("ti")
    out["n_usable_neighbors"] = n_usable               # diagnostic: usable count in the required set
    out["ess_neighbors"] = ess                          # diagnostic: per-ts effective sample size
    out[f"consensus_{qoi}"] = consensus
    if smooth and keep_raw_consensus:
        out[f"consensus_{qoi}_raw"] = consensus_raw

    out.attrs.update(qoi=qoi, target=target, method=method, param=param,
                     dist_offset=dist_offset,
                     steer_thresh=steer_thresh, meas_vane_thresh=meas_vane_thresh,
                     valid_turbines=None if valid_turbines is None else sorted(map(int, valid_turbines)),
                     min_neighbors=min_neighbors, required_set_size=int(required_set.sum()),
                     base_ess=base_ess,
                     smooth=smooth, smooth_window=smooth_window, smooth_gap=smooth_gap,
                     smooth_min_periods=smooth_min_periods, smooth_how=smooth_how)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Correlation with the target's own (individual) signal
# --------------------------------------------------------------------------- #
def _linear_corr(a, b):
    """Pearson correlation over the pairwise-finite entries of a, b. NaN if fewer
    than 2 usable pairs or either series is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    a, b = a[m], b[m]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _circular_corr(a_deg, b_deg):
    """Jammalamadaka-Sarma circular-circular correlation coefficient for two
    angle series given in degrees. Ranges [-1, 1] like Pearson's r, but respects
    the wraparound (0 == 360) so it doesn't get confused by a target's vane/
    winddir crossing the +-180 boundary. NaN if fewer than 2 usable pairs."""
    a = np.deg2rad(np.asarray(a_deg, dtype=float))
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    a, b = a[m], b[m]
    abar = np.arctan2(np.sin(a).sum(), np.cos(a).sum())
    bbar = np.arctan2(np.sin(b).sum(), np.cos(b).sum())
    sa = np.sin(a - abar)
    sb = np.sin(b - bbar)
    den = np.sqrt(np.sum(sa * sa) * np.sum(sb * sb))
    if den == 0:
        return np.nan
    return float(np.sum(sa * sb) / den)


# --------------------------------------------------------------------------- #
# Five-slice error report for one estimate
# --------------------------------------------------------------------------- #
def score_slices(est, qoi):
    """mean error, RMSE, std of error, and correlation with the target's own
    ("individual") measured signal, for consensus vs measured, on five slices:
       a |targetvane|<0.1 ; b |targetvane|>=0.1 ; c |measured vane|<0.1 ;
       d >=0.1 ; e_all: every valid row, no vane restriction.

    `corr` (and its magnitude `abs_corr`) is the correlation between the
    consensus estimate and the target's OWN measured_{qoi} -- circular-circular
    correlation for winddir/windvane, Pearson otherwise. This is a diagnostic
    for how much of the target's individual/idiosyncratic signal (rather than
    an independent neighbor-based estimate) is leaking into the consensus: a
    method dominated by one very close neighbor will tend to correlate more
    strongly with the target's own signal than a genuinely averaged one, for
    reasons unrelated to accuracy (see project notes on regression dilution).
    Prefer `e_all` (unconditioned on vane) when using this as a selection
    criterion, since the leakage this flags isn't specific to the steered vs.
    unsteered condition.

    `ess` is the mean per-ts effective sample size (Kish's ESS, see
    _effective_sample_size) within the slice -- how many turbines-worth of
    genuine averaging the estimate reflects on average. A useful cross-check
    against corr/abs_corr and RMSE when comparing weighting schemes: e.g. two
    configs with similar RMSE but very different `ess` tells you one is doing
    real multi-turbine averaging and the other is riding on ~1 neighbor.
    """
    circular = qoi in ("winddir", "windvane")
    pred = est[f"consensus_{qoi}"].to_numpy(float)
    meas = est[f"measured_{qoi}"].to_numpy(float)
    setv = est["targetwindvane"].to_numpy(float)
    mwv = est["measured_windvane"].to_numpy(float)
    ess = est["ess_neighbors"].to_numpy(float) if "ess_neighbors" in est.columns \
        else np.full(len(est), np.nan)

    err = pred - meas
    if circular:
        err = (err + 180.0) % 360.0 - 180.0
    base = ~np.isnan(err)

    slices = {
        "a_targetvane_lt_0.1": base & (np.abs(setv) < 0.1),
        "b_targetvane_ge_0.1": base & (np.abs(setv) >= 0.1),
        "c_measvane_lt_0.1": base & (np.abs(mwv) < 0.1),
        "d_measvane_ge_0.1": base & (np.abs(mwv) >= 0.1),
        "e_all": base,
    }
    corr_fn = _circular_corr if circular else _linear_corr
    rows = []
    for name, m in slices.items():
        e = err[m]
        if e.size:
            bias = float(np.mean(e))          # bias = mean error
            std = float(np.std(e))            # std of error around its own mean
            var = std * std                   # variance term
            rmse = float(np.sqrt(np.mean(e**2)))
            corr = corr_fn(pred[m], meas[m])  # correlation w/ target's individual signal
            ess_slice = ess[m]
            rows.append({
                "slice": name, "n": int(e.size),
                "mean_err": bias,             # = bias
                "rmse": rmse,
                "std_err": std,
                "bias": bias,                 # explicit, = mean_err
                "variance": var,              # = std_err**2
                "rmse_check": float(np.sqrt(bias * bias + var)),  # == rmse up to fp
                "corr": corr,
                "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                "ess": float(np.nanmean(ess_slice)) if np.any(np.isfinite(ess_slice)) else np.nan,
            })
        else:
            rows.append({
                "slice": name, "n": 0, "mean_err": np.nan, "rmse": np.nan,
                "std_err": np.nan, "bias": np.nan, "variance": np.nan,
                "rmse_check": np.nan, "corr": np.nan, "abs_corr": np.nan, "ess": np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Outer sweep: method x param x target, with per-turbine and averaged reports
# --------------------------------------------------------------------------- #
def sweep(prep, *, qoi, target_turbines, method_grid, dist_offset=0.0,
          valid_turbines=None, steer_thresh=0.1, meas_vane_thresh=None,
          min_neighbors=None,
          smooth=False, smooth_window="10min", smooth_gap="5min",
          smooth_min_periods=3, smooth_how="mean",
          selection_slice="a_targetvane_lt_0.1", selection_metric="rmse",
          verbose=True):
    """For each (method, param, target): estimate + five-slice score.

    dist_offset : see estimate_consensus / _base_weights; applied to every
        'invdist' entry in method_grid for this call. To compare several d0
        candidates, call sweep() once per candidate and compare the resulting
        `averaged` tables (rmse, abs_corr, ess together) -- e.g. sweep with
        dist_offset=0 (classic IDW) vs. dist_offset=2 (in rotor diameters) to
        quantify how much a given offset raises `ess` and how much (if any)
        RMSE cost that comes with.

    min_neighbors : see estimate_consensus (8); passed through unchanged so the
        sweep scores each (method, param) under the same min-usable-neighbors rule.

    selection_metric : which averaged column to minimize when picking `best` --
        'rmse' (default, backward compatible) or 'abs_corr' to instead pick the
        (method, param) with the LEAST correlation between the consensus and the
        target's own individual measured signal (see score_slices). Raw RMSE is
        biased toward whichever config averages the least (see project notes on
        regression dilution / IDW nearest-neighbor domination); abs_corr directly
        targets independence from the target's own idiosyncratic signal instead.
        When using 'abs_corr' you'll usually also want selection_slice='e_all',
        since the leakage it flags isn't specific to the steered/unsteered slices.
        Note: minimizing correlation alone doesn't guarantee physical validity --
        e.g. a degenerate, near-constant, or otherwise nonsensical consensus can
        also score a low correlation. Inspect `averaged` (rmse, bias, and abs_corr
        together) rather than trusting `best` blindly.

    Returns (per_turbine, averaged, best):
      per_turbine : long rows (method,param,target,slice, n,mean_err,rmse,std_err,corr,ess)
      averaged    : mean over targets, per (method,param,slice)
      best        : the (method,param) minimizing averaged `selection_metric`
                    on `selection_slice`
    """
    recs = []
    for method, params in method_grid:
        for param in params:
            for t in target_turbines:
                est = estimate_consensus(prep, qoi=qoi, target=t, method=method, param=param,
                                         dist_offset=dist_offset,
                                         valid_turbines=valid_turbines,
                                         steer_thresh=steer_thresh, meas_vane_thresh=meas_vane_thresh,
                                         min_neighbors=min_neighbors,
                                         smooth=smooth, smooth_window=smooth_window,
                                         smooth_gap=smooth_gap, smooth_min_periods=smooth_min_periods,
                                         smooth_how=smooth_how)
                sc = score_slices(est, qoi)
                sc["method"] = method
                sc["param"] = "none" if param is None else str(param)
                sc["target"] = t
                recs.append(sc)
            if verbose:
                print(f"  done {qoi} {method} param={param}")
    per_turbine = pd.concat(recs, ignore_index=True)

    # average across targets, per method/param/slice. corr/abs_corr are averaged
    # with a plain mean across targets (not a Fisher z-transform) -- fine as a
    # relative ranking signal across the method_grid, which is all `best` uses it
    # for; treat the absolute averaged-corr number as approximate.
    averaged = (per_turbine
                .groupby(["method", "param", "slice"], as_index=False)
                .agg(mean_err=("mean_err", "mean"),
                     rmse=("rmse", "mean"),
                     std_err=("std_err", "mean"),
                     bias=("bias", "mean"),
                     variance=("variance", "mean"),
                     corr=("corr", "mean"),
                     abs_corr=("abs_corr", "mean"),
                     ess=("ess", "mean"),
                     n_targets=("target", "nunique")))

    # decide best method/param on the selection slice by averaged selection_metric
    if selection_metric not in ("rmse", "abs_corr", "mean_err", "std_err", "variance"):
        raise ValueError(
            f"selection_metric={selection_metric!r} not recognized; expected one of "
            "'rmse', 'abs_corr', 'mean_err', 'std_err', 'variance'."
        )
    sel = averaged[averaged["slice"] == selection_slice].dropna(subset=[selection_metric])
    best = sel.loc[[sel[selection_metric].idxmin()]].reset_index(drop=True) if len(sel) else pd.DataFrame()
    return per_turbine, averaged, best


# --------------------------------------------------------------------------- #
# Apply winning method per QoI and append consensus columns onto corrected_df
# --------------------------------------------------------------------------- #
def append_consensus(corrected_df, prep, winners, *, target_turbines,
                     dist_offset=0.0,
                     steer_thresh=0.1, meas_vane_thresh=None,
                     valid_turbines=None, min_neighbors=None,
                     smooth=False, smooth_window="10min", smooth_gap="5min",
                     smooth_min_periods=3, smooth_how="mean",
                     verbose=True):
    """Run the chosen consensus method for each QoI over each target turbine and
    merge the predictions back onto corrected_df by (turbine, ts).

    Parameters
    ----------
    corrected_df : long frame, one row per (turbine, ts).
    prep         : output of prepare().
    winners      : dict {qoi: (method, param)} or {qoi: (method, param, dist_offset)},
                   e.g.
                   {'windvane': ('invdist', 5),
                    'windspeed': ('invdist', 5, 2.0),   # dist_offset=2.0 D for this qoi
                    'power': ('gaussian', 5.0)}
                   A 2-tuple falls back to the top-level `dist_offset` argument.
    target_turbines : which turbines to produce consensus for. Others get NaN.
    dist_offset : default dist_offset (see estimate_consensus / _base_weights) for
        any qoi in `winners` given as a 2-tuple; ignored for method != 'invdist'.
    min_neighbors : see estimate_consensus (8); applied to every qoi in `winners`.
        A ts with fewer than this many usable neighbors in the required set is NaN.

    Returns a copy of corrected_df with columns consensus_{qoi} appended (one per
    qoi in `winners`), aligned on (turbine, ts). Rows for turbines not in
    target_turbines, or timestamps with no valid estimate (including those dropped
    by min_neighbors), are NaN.
    """
    out = corrected_df.copy()
    out["ts"] = pd.to_datetime(out["ts"])

    for qoi, spec in winners.items():
        if len(spec) == 3:
            method, param, qoi_dist_offset = spec
        else:
            method, param = spec
            qoi_dist_offset = dist_offset
        # collect per-target estimates, tagging turbine + ts, keep only the consensus col
        pieces = []
        for t in target_turbines:
            est = estimate_consensus(
                prep, qoi=qoi, target=t, method=method, param=param,
                dist_offset=qoi_dist_offset,
                valid_turbines=valid_turbines,
                steer_thresh=steer_thresh, meas_vane_thresh=meas_vane_thresh,
                min_neighbors=min_neighbors,
                smooth=smooth, smooth_window=smooth_window, smooth_gap=smooth_gap,
                smooth_min_periods=smooth_min_periods, smooth_how=smooth_how,
            )
            pieces.append(est[["ts", "turbine", f"consensus_{qoi}"]])
        allest = pd.concat(pieces, ignore_index=True)
        allest["ts"] = pd.to_datetime(allest["ts"])

        # merge onto the long frame by (turbine, ts)
        out = out.merge(allest, on=["turbine", "ts"], how="left")
        if verbose:
            n = out[f"consensus_{qoi}"].notna().sum()
            print(f"appended consensus_{qoi} ({method}, param={param}): "
                  f"{n} non-NaN rows over {len(target_turbines)} targets")

    return out


# --------------------------------------------------------------------------- #
# Post-consensus calibration (moved from updated_ntf_code.ipynb, cells 11/13/15)
# --------------------------------------------------------------------------- #
def median_regression(x, y, n_iter=30, tol=1e-8, eps=1e-6):
    """Median (L1) regression of y on x via iteratively reweighted least squares.
    Returns (slope, intercept)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]     # OLS start
    for _ in range(n_iter):
        resid = y - X @ beta
        w = 1.0 / np.maximum(np.abs(resid), eps)     # down-weight large residuals
        Xw = X * w[:, None]
        beta_new = np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta[1], beta[0]   # slope, intercept


def shift_consensus_windvane_by_turbine(
    data_df: pd.DataFrame,
    turbine_col: str = "turbine",
    steering_col: str = "targetwindvane",
    measured_col: str = "windvane",
    consensus_col: str = "consensus_windvane",
    unsteered_eps: float = 0.1,
    min_unsteered_samples: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (data_df with a new f"{consensus_col}_shifted" column, offsets).

    On the unsteered slice (|steering_col| <= unsteered_eps), per turbine:

        offset[turbine] = median(consensus_col | turbine, unsteered)
                         - median(measured_col   | turbine, unsteered)

    and that constant is subtracted from consensus_col for ALL rows of that
    turbine (not just the unsteered slice):

        consensus_shifted = consensus_col - offset[turbine]

    Because the median is equivariant to additive shifts, this makes

        median(consensus_shifted | turbine, unsteered)
            == median(measured_col | turbine, unsteered)

    exactly, for any turbine with a stable per-turbine offset estimate --
    i.e. the shifted consensus median lines up with that turbine's own
    measured median on the unsteered slice, whatever that measured median
    actually is (not necessarily 0).
    """
    unsteered_mask = data_df[steering_col].abs() <= unsteered_eps
    unsteered = data_df.loc[unsteered_mask]

    counts = unsteered.groupby(turbine_col).size()
    consensus_median = unsteered.groupby(turbine_col)[consensus_col].median()
    measured_median = unsteered.groupby(turbine_col)[measured_col].median()
    offsets = consensus_median - measured_median

    fleet_consensus_median = unsteered[consensus_col].median()
    fleet_measured_median = unsteered[measured_col].median()
    fleet_fallback = fleet_consensus_median - fleet_measured_median

    thin = counts[counts < min_unsteered_samples]
    if len(thin):
        # Fall back to the fleet-wide unsteered offset for turbines that
        # don't have enough unsteered samples of their own for a stable
        # per-turbine estimate. Flag these -- a thin turbine's offset should
        # be revisited as more data accumulates, and a turbine that NEVER
        # gets enough unsteered samples may indicate it rarely runs
        # aligned (worth checking why) rather than a plotting artifact.
        offsets.loc[thin.index] = fleet_fallback
        print(
            f"[shift_consensus_windvane_by_turbine] {len(thin)} turbine(s) had "
            f"< {min_unsteered_samples} unsteered samples; used fleet-wide "
            f"fallback offset ({fleet_fallback:.3f}) for: {list(thin.index)}"
        )

    missing = set(data_df[turbine_col].unique()) - set(offsets.index)
    if missing:
        for t in missing:
            offsets.loc[t] = fleet_fallback
        print(
            f"[shift_consensus_windvane_by_turbine] {len(missing)} turbine(s) had "
            f"NO unsteered samples at all; used fleet-wide fallback for: {sorted(missing)}"
        )

    out_col = f"{consensus_col}_shifted"
    data_df = data_df.copy()
    data_df[out_col] = data_df[consensus_col] - data_df[turbine_col].map(offsets)

    return data_df, offsets


def calibrate_ratio_by_turbine(data_df, qoi, *, turbine_col="turbine",
                               steering_col="targetwindvane", unsteered_eps=0.1,
                               min_n=30, verbose=True):
    """Per-turbine ratio calibration of consensus_{qoi} (qoi = 'windspeed' or 'power').

    On the unsteered slice (|steering_col| < unsteered_eps), fit per turbine

        measured_{qoi} / consensus_{qoi}  =  slope * measured_{qoi} + intercept

    by median regression (global pooled fit as fallback for turbines with fewer
    than `min_n` usable points), then apply it to ALL rows of that turbine:

        consensus_{qoi}_shifted = consensus_{qoi} * (slope * measured_{qoi} + intercept)

    measured_{qoi} is the plain column named `qoi` (e.g. 'windspeed', 'power').
    Reproduces the per-QoI loops that were in updated_ntf_code.ipynb (cells 13
    and 15). OLS slope/intercept/r are kept in the results table as diagnostics.

    Returns (data_df copy with consensus_{qoi}_shifted, results_df indexed by turbine).
    """
    from scipy import stats

    meas_col = qoi
    cons_col = f"consensus_{qoi}"
    out_col = f"{cons_col}_shifted"

    # --- global (pooled) fit, used as a fallback for low-count turbines ---
    tdata_df = data_df[np.abs(data_df[steering_col]) < unsteered_eps]
    x_all = tdata_df[meas_col]
    y_all = tdata_df[meas_col] / tdata_df[cons_col]
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    slope_global, intercept_global = median_regression(
        x_all[mask_all].to_numpy(), y_all[mask_all].to_numpy()
    )
    if verbose:
        print(f"global fallback: y = {slope_global:.5f}*x + {intercept_global:.5f}")

    data_df = data_df.copy()
    data_df[out_col] = np.nan
    results = {}

    for tid, grp in data_df.groupby(turbine_col):
        tgrp = grp[np.abs(grp[steering_col]) < unsteered_eps]
        x = tgrp[meas_col]
        y = tgrp[meas_col] / tgrp[cons_col]
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())

        if n >= min_n:
            x_f, y_f = x[mask].to_numpy(), y[mask].to_numpy()
            slope, intercept, r, p, se = stats.linregress(x_f, y_f)
            slope_med, intercept_med = median_regression(x_f, y_f)
            used_fallback = False
        else:
            slope, intercept, r, p, se = np.nan, np.nan, np.nan, np.nan, np.nan
            slope_med, intercept_med = slope_global, intercept_global
            used_fallback = True

        crossover = (1 - intercept_med) / slope_med if slope_med != 0 else np.nan
        results[tid] = dict(n=n, slope_med=slope_med, intercept_med=intercept_med,
                            crossover=crossover, used_fallback=used_fallback,
                            slope_ols=slope, intercept_ols=intercept, r=r)
        if verbose:
            flag = " [fallback: global fit]" if used_fallback else ""
            print(f"turbine {tid} (n={n}): y = {slope_med:.5f}*x + {intercept_med:.5f}, "
                  f"crossover={crossover:.2f}{flag}")

        turbine_rows = data_df[turbine_col] == tid
        fitted_ratio_med = slope_med * data_df.loc[turbine_rows, meas_col] + intercept_med
        data_df.loc[turbine_rows, out_col] = data_df.loc[turbine_rows, cons_col] * fitted_ratio_med

    return data_df, pd.DataFrame(results).T
