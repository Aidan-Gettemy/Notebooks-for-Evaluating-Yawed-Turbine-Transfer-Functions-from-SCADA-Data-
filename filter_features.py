"""
Filtering-feature computation for the NTF workflow (used by Load_FilterFeatures.ipynb).

Trimmed from the original NTF_functions.py: only the functions on the live path
are kept. Call chain:

    latlon_to_xy                  -> turbine x/y (used to build x_D, y_D)
    circdiff180                   -> circular difference in [-180, 180)
    build_uniform_cluster_matrix  -> (N x N) equal-weight nearest-`clusternum` matrix
        makedistancedatadf
    calcfilter_feat_vectorized    -> wide per-turbine feature frame
        derive_ref_signal_vectorized
        circular_features_vectorized
            _roll_unit_phasor_mean_deg, _roll_std_complex_like
        farm_mean_and_stdB, farm_zscoreB

Removed (not called anywhere in the current workflow): the loop-based legacy
feature code (calcfilter_feat, circular_features, derive_ref_*), GP
interpolation, filterprocess / filterprocess_faster, make_estim_target_only,
run_NTF_model, the old time_series_kfold (sorted on "timestamp"; the live
version sorts on "ts" and now lives in cv_harness.py), and the comparison plots.
"""

import numpy as np
import pandas as pd
import utm


# --------------------------------------------------------------------------- #
# Geometry + circular helpers
# --------------------------------------------------------------------------- #
def latlon_to_xy(latlist, lonlist):
    """Convert latitude and longitude to x and y"""
    x, y = [], []
    for lat, lon in zip(latlist, lonlist):
        x1, y1, _, _ = utm.from_latlon(lat, lon)
        x.append(x1)
        y.append(y1)
    
    if len(x) == 1:
        x = x[0]
    
    if len(y) == 1:
        y = y[0]
    
    return x, y


def circdiff180(a, b):
    """Calculate the circular difference between two wind direction/wind vane angles"""
    return ((a - b + 720 + 180) % 360) - 180


def makedistancedatadf(target_turbine, turbmetadata):
    """Make a dataframe of distances from a target turbine"""
    # Calculate the wind direction of direct waking for each other turbine
    turbList = turbmetadata.index.to_list()
    
    # Drop the target turbine
    turbList.pop(turbList.index(target_turbine))
    turbines = turbmetadata
    dist_df = pd.DataFrame(columns = ["distance_to_"+str(target_turbine)])
    
    # Calculate the distance between the target turbine and each other turbine
    for p in turbList:
        t1 = p
        t2 = target_turbine
        delx = turbines.loc[t1].x_D-turbines.loc[t2].x_D
        dely = turbines.loc[t1].y_D-turbines.loc[t2].y_D

        a = (delx)**2
        b = (dely)**2
        
        dist_df.loc[f"{t1}", "distance_to_"+str(target_turbine)] = np.sqrt(a+b)

    return dist_df


def build_uniform_cluster_matrix(turbines, turbmetadata, clusternum):
    
    turblist = turbines.index.unique().to_list()
    N = len(turblist)
    
    M = np.zeros((N, N))
    
    for idx, target in enumerate(turblist):
        
        di_df = makedistancedatadf(int(target), turbmetadata)
        cluster_df = di_df.sort_values("distance_to_"+str(target))
        clusterlist = cluster_df.index.to_list()[0:clusternum]
        clusterlist = [int(i) for i in clusterlist]
        
        for c in clusterlist:
            j = turblist.index(c)
            M[idx, j] = 1.0 / clusternum
    
    return M, turblist


# --------------------------------------------------------------------------- #
# Vectorized feature computation
# --------------------------------------------------------------------------- #
def derive_ref_signal_vectorized(winddir_df, cluster_matrix):
    
    WD = winddir_df.values  # shape (T, N)
    
    complex_vals = np.exp(1j * np.deg2rad(WD))
    
    ref_complex = complex_vals @ cluster_matrix.T
    
    ref_deg = (np.rad2deg(np.angle(ref_complex)) + 360) % 360
    
    return ref_deg


def _roll_unit_phasor_mean_deg(deg_mat, n):
    """
    deg_mat: (T, N) degrees with NaNs
    returns: (T, N) rolling circular mean in degrees in [0, 360)
    Matches roll_comp_mean + mean_complex behavior.
    """
    rad = np.deg2rad(deg_mat)
    cosv = np.cos(rad)
    sinv = np.sin(rad)

    cos_df = pd.DataFrame(cosv)
    sin_df = pd.DataFrame(sinv)

    mcos = cos_df.rolling(n, min_periods=1).mean()
    msin = sin_df.rolling(n, min_periods=1).mean()

    ang = (np.rad2deg(np.arctan2(msin.to_numpy(), mcos.to_numpy())) + 360) % 360
    return ang, mcos.to_numpy(), msin.to_numpy(), cos_df, sin_df


def _roll_std_complex_like(deg_mat, n):
    """
    Replicates std_complex over rolling windows of unit phasors:
      std = sqrt(mean(|z - m|^2))
    with NaNs ignored.
    Uses identity:
      mean((x-mx)^2) = mean(x^2) - mx^2
    """
    rad = np.deg2rad(deg_mat)
    x = np.cos(rad)
    y = np.sin(rad)

    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)

    mx = x_df.rolling(n, min_periods=1).mean().to_numpy()
    my = y_df.rolling(n, min_periods=1).mean().to_numpy()

    ex2 = (x_df * x_df).rolling(n, min_periods=1).mean().to_numpy()
    ey2 = (y_df * y_df).rolling(n, min_periods=1).mean().to_numpy()

    # variance in complex plane = var(x) + var(y)
    var = (ex2 - mx*mx) + (ey2 - my*my)

    # numerical safety
    var = np.clip(var, 0.0, None)
    return np.sqrt(var)


def circular_features_vectorized(WD, REF, n1, n2):
    """
    WD, REF: (T, N) arrays in degrees with NaNs.
    Returns:
      circ_dif_means: (T, N)
      std_ratio:      (T, N)
      stdfeature:     (T, N)
    matching legacy circular_features().
    """

    # Rolling circular means (match roll_comp_mean)
    target_mean_roll, *_ = _roll_unit_phasor_mean_deg(WD,  n1)
    ref_mean_roll,    *_ = _roll_unit_phasor_mean_deg(REF, n1)

    # Rolling stds (match roll_comp_std + std_complex)
    target_std_roll = _roll_std_complex_like(WD,  n1)
    ref_std_roll    = _roll_std_complex_like(REF, n1)

    std_ratio = target_std_roll / ref_std_roll  # will produce inf where ref_std_roll==0, same as legacy

    # Circular difference between means (match circdiff180)
    circ_dif_means = ((target_mean_roll - ref_mean_roll + 720 + 180) % 360) - 180

    # Rolling std of circ_dif_means over n2, dropping NaNs (match your loop)
    # FAST rolling pop-std of circ_dif_means over n2, ignoring NaNs
    circ_df = pd.DataFrame(circ_dif_means)

    mean = circ_df.rolling(n2, min_periods=1).mean()
    mean2 = (circ_df * circ_df).rolling(n2, min_periods=1).mean()

    var = mean2 - mean * mean
    var = var.clip(lower=0)

    stdfeature = np.sqrt(var).to_numpy()

    return circ_dif_means, std_ratio, stdfeature


def farm_mean_and_stdB(winddir_df: pd.DataFrame):
    WD  = winddir_df.to_numpy(dtype=float)
    rad = np.deg2rad(WD)
    x, y = np.cos(rad), np.sin(rad)

    mx = np.nanmean(x, axis=1)
    my = np.nanmean(y, axis=1)

    farm_mean_deg = (np.rad2deg(np.arctan2(my, mx)) + 360) % 360

    # mean resultant length R in [0, 1]
    R = np.sqrt(mx * mx + my * my)
    R = np.clip(R, 1e-12, 1.0)            # guard log(0) and R>1 rounding

    # circular standard deviation, in DEGREES  (sqrt(-2 ln R) is in radians)
    farm_std_deg = np.rad2deg(np.sqrt(-2.0 * np.log(R)))

    return farm_mean_deg, farm_std_deg


def farm_zscoreB(winddir_df: pd.DataFrame):
    """
    Returns:
      z: (T, N) z-score-like distance = circdiff180(wd, farm_mean) / farm_stdB
    """
    farm_mean_deg, farm_stdB = farm_mean_and_stdB(winddir_df)

    WD = winddir_df.to_numpy(dtype=float)
    # circdiff180 elementwise
    dif = ((WD - farm_mean_deg[:, None] + 720 + 180) % 360) - 180

    # Avoid divide-by-zero: if std is 0 or NaN, z should be NaN
    denom = farm_stdB.copy()
    denom[(denom == 0) | np.isnan(denom)] = np.nan

    z = dif / denom[:, None]
    return z, farm_mean_deg, farm_stdB


def calcfilter_feat_vectorized(turbines, Alldata, cluster_matrix, n1, n2):
    """
    Fully vectorized rewrite of calcfilter_feat.
    Mathematically equivalent.
    """

    # ---------------------------------------------------------
    # 1. Unstack once
    # ---------------------------------------------------------
    turblist = turbines.index.unique().to_list()
    N = len(turblist)

    winddir_df = Alldata.winddir_deg.unstack(level="turbine_id")[turblist]
    windsp_df  = Alldata.windspeed_mps.unstack(level="turbine_id")[turblist]
    genpwr_df  = Alldata.power_kw.unstack(level="turbine_id")[turblist]
    windvn_df  = Alldata.windvane_deg.unstack(level="turbine_id")[turblist]
    nacdir_df  = Alldata.nacelledir_deg.unstack(level="turbine_id")[turblist]
    targetwindvane_df = Alldata.target_windvane_deg.unstack(level="turbine_id")[turblist]
    ti_df = Alldata.ti.unstack(level="turbine_id")[turblist]
    state_df = Alldata.state.unstack(level="turbine_id")[turblist]

    

    # -------------------------------------
    # Reference signals (vectorized)
    # -------------------------------------
    WD = winddir_df.values
    WS = windsp_df.values
    PW = genpwr_df.values

    ref_sig = derive_ref_signal_vectorized(winddir_df, cluster_matrix)
    ref_ws  = WS @ cluster_matrix.T
    ref_pw  = PW @ cluster_matrix.T

    # -------------------------------------
    # Circular features (vectorized)
    # -------------------------------------
    circ_dif_mean, ratio_std_comp, circ_dif_std = circular_features_vectorized(
        WD,
        ref_sig,
        n1,
        n2
    )

    # -------------------------------------
    # Farm std upgraded: want the 
    # -------------------------------------
    farm_mean_deg, farm_stdB = farm_mean_and_stdB(winddir_df)
    z, _, _ = farm_zscoreB(winddir_df)

    # -------------------------------------
    # Build output dictionary EXACT naming
    # -------------------------------------
    result_dict = {}

    for i, turb in enumerate(turblist):
        # Name columns by the actual turbine ID. (Originally `i + 401`, which is
        # only correct when turblist is exactly 401, 402, ... in order.)
        turb_num = int(turb)

        result_dict["ref_sig_data"+str(turb_num)] = ref_sig[:, i]
        result_dict["ref_ws_data"+str(turb_num)]  = ref_ws[:, i]
        result_dict["ref_pwr_data"+str(turb_num)] = ref_pw[:, i]
        result_dict["circ_dif_mean"+str(turb_num)] = circ_dif_mean[:, i]
        result_dict["ratio_std_comp"+str(turb_num)] = ratio_std_comp[:, i]
        result_dict["circ_dif_std"+str(turb_num)] = circ_dif_std[:, i]
        result_dict[f"std_farm_avg_wd{turb_num}"] = farm_stdB  # vector length T
        result_dict[f"farm_z_wd{turb_num}"] = z[:, i]

        result_dict["winddir"+str(turb_num)] = WD[:, i]
        result_dict["windspeed"+str(turb_num)] = WS[:, i]
        result_dict["windvane"+str(turb_num)] = windvn_df.values[:, i]
        result_dict["power"+str(turb_num)] = PW[:, i]
        result_dict["nacdir"+str(turb_num)] = nacdir_df.values[:, i]
        result_dict["targetwindvane"+str(turb_num)] = targetwindvane_df.values[:, i]
        result_dict["ti"+str(turb_num)] = ti_df.values[:, i]
        result_dict["state"+str(turb_num)] = state_df.values[:, i]

    return pd.DataFrame(result_dict, index=winddir_df.index)
