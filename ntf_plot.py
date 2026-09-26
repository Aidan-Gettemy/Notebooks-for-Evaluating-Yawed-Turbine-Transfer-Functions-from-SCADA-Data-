"""
Plotting for the final NTF stage (updated_ntf_code.ipynb).

  plot_ntf_figure           : 2 x N NTF figure -- row 1 median curves per series
                              level, row 2 90% envelope for one level; optional
                              binned box-plot overlay of raw data on one column.
  style_axes                : the font-size clean-up that was pasted after every
                              plot_ntf_figure call.
  find_level                : density level enclosing a given probability mass
                              (KDE contour levels).
  plot_residual_diagnostics : residual / histogram / Q-Q panel (the "Figure_11a_*"
                              cells).
  plot_feature_importance   : xgboost gain-importance bar chart.

plot_ntf_figure accepts either the new list form
    plot_ntf_figure([(grid_gb, "GB regressor"), (grid_cos, "Cosine law")], ...)
or the old two-grid form
    plot_ntf_figure(grid_gb, grid_lr, ...)
which is treated as [(grid_gb, "GB regressor"), (grid_lr, "Linear regression")].

Grids come from ntf_fit.predict_grid / to_ratio_grid / build_cosine_grid: columns
are the sweep feature, the series feature, and q0/q1/q2 (= 0.05/0.50/0.95).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# NTF figure
# --------------------------------------------------------------------------- #
def _series_levels(grid, series_col):
    return sorted(grid[series_col].unique())


def _refs(ax, grid, sweep_col, show_identity, ratio):
    """Draw reference lines: x=0, and either y=x (bias plots) or y=1 (ratio plots)."""
    ax.axvline(0, color="gray", lw=0.6, ls="-", alpha=0.5)
    if ratio:
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    else:
        ax.axhline(0, color="gray", lw=0.6, ls="-", alpha=0.5)
        if show_identity:
            lo = float(grid[sweep_col].min()); hi = float(grid[sweep_col].max())
            ax.plot([lo, hi], [lo, hi], color="tab:orange", lw=2.0, label="y = x", zorder=1)


def _add_binned_boxplot(ax, data, sweep_col, value_col, bin_width=2.0,
                        whis=(5, 95), x_range=None, min_n=5, color="0.4", **kwargs):
    """Overlay binned box-and-whisker summaries of raw (sweep_col, value_col)
    data onto ax. Bins are bin_width-wide; whiskers default to the 5th/95th
    percentiles (matching the paper's quantile convention) rather than
    matplotlib's default 1.5*IQR rule."""
    x = data[sweep_col].to_numpy()
    y = data[value_col].to_numpy()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    lo, hi = x_range if x_range is not None else (x.min(), x.max())
    edges = np.arange(lo, hi + bin_width, bin_width)

    box_vals, positions = [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        vals = y[(x >= b0) & (x < b1)]
        if len(vals) >= min_n:          # skip bins too sparse to summarize
            box_vals.append(vals)
            positions.append((b0 + b1) / 2)

    defaults = dict(widths=bin_width * 0.8, whis=whis, showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
                    medianprops=dict(color=color, lw=1.2),
                    whiskerprops=dict(color=color), capprops=dict(color=color))
    defaults.update(kwargs)
    return ax.boxplot(box_vals, positions=positions, manage_ticks=False, zorder=1, **defaults)


def plot_ntf_figure(models, grid_lr=None, *, sweep_col, series_col,
                    quantiles=(0.05, 0.50, 0.95),
                    envelope_level_index=2,
                    xlabel="Measured Wind Vane (deg)",
                    ylabel="Predicted Wind Vane (deg)",
                    series_label="Power", series_unit="kW",
                    show_identity=True, ratio=False, title=None,
                    box_data=None, box_value_col=None, box_target_index=None,
                    box_bin_width=2.0, box_whis=(5, 95)):
    """Render the 2xN NTF figure, one column per (grid, tag) pair in `models`.

    models : list of (grid, tag) pairs, OR a single grid (old call style) with the
        LR grid passed as the second positional argument `grid_lr`.
    envelope_level_index : which series level (0-based, clamped to the last
        level) gets the 90% band in row 2.
    ratio : True draws y=1 instead of y=x (windspeed / power ratio plots).
    box_data / box_value_col: optional raw (unbinned) DataFrame + column name,
        drawn as binned box-and-whisker summaries in row 1.
    box_target_index: which column (0-based) gets the overlay -- e.g. 1 for the
        "Cosine law" column. None (default) draws no overlay.
    box_bin_width / box_whis: bin width (degrees) and whisker percentiles.
    """
    if isinstance(models, pd.DataFrame):          # old signature: (grid_gb, grid_lr)
        if grid_lr is None:
            raise ValueError("old-style call needs both grid_gb and grid_lr")
        models = [(models, "GB regressor"), (grid_lr, "Linear regression")]

    qlo, qmid, qhi = "q0", "q1", "q2"             # 0.05, 0.50, 0.95 by construction
    n = len(models)
    letters = "abcdefghijklmnopqrstuvwxyz"

    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(2, n)
    row1, row2 = axes[0], axes[1]

    # ---- row 1: median curves, one per series level ---------------------- #
    for i, (ax, (grid, tag)) in enumerate(zip(row1, models)):
        if box_data is not None and i == box_target_index:
            xmin, xmax = grid[sweep_col].min(), grid[sweep_col].max()
            _add_binned_boxplot(ax, box_data, sweep_col, box_value_col,
                                bin_width=box_bin_width, whis=box_whis,
                                x_range=(xmin, xmax))
        for lvl in _series_levels(grid, series_col):
            g = grid[grid[series_col] == lvl].sort_values(sweep_col)
            ax.plot(g[sweep_col], g[qmid], marker="o", ms=3, zorder=3,
                    label=f"{series_label} = {int(lvl)} {series_unit}")
        _refs(ax, grid, sweep_col, show_identity, ratio)
        ax.set_title(f"({letters[i]}) {tag}")
    row1[0].legend(fontsize=7, loc="best")

    # ---- row 2: 90% envelope for one series level ------------------------ #
    for i, (ax, (grid, tag)) in enumerate(zip(row2, models)):
        levels = _series_levels(grid, series_col)
        env_level = levels[min(envelope_level_index, len(levels) - 1)]
        g = grid[grid[series_col] == env_level].sort_values(sweep_col)
        ax.plot(g[sweep_col], g[qmid], marker="o", ms=3, color="tab:blue", zorder=3,
                label=f"Prediction: {int(env_level)} {series_unit}")
        ax.fill_between(g[sweep_col], g[qlo], g[qhi], alpha=0.2, color="tab:blue",
                        label=f"90% PI: {int(env_level)} {series_unit}")
        _refs(ax, grid, sweep_col, show_identity, ratio)
        ax.set_title(f"({letters[n + i]}) {tag}")
        ax.legend(fontsize=7, loc="best")

    for ax in row2:
        ax.set_xlabel(xlabel)
    for ax in (row1[0], row2[0]):
        ax.set_ylabel(ylabel if not ratio else ylabel.replace("Predicted", "Ratio"))

    if title:
        fig.suptitle(title, y=1.00)
    fig.tight_layout()
    return fig, axes


def style_axes(fig, axes, *, label_size=21, title_size=21, tick_size=15,
               legend_size=15):
    """Resize labels, titles, ticks and legend text on every axis, then re-run
    tight_layout. Replaces the block that followed each plot_ntf_figure call
    (plot_ntf_figure sets no font sizes, so they otherwise inherit rcParams)."""
    for ax in np.ravel(axes):
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.title.set_size(title_size)
        ax.tick_params(labelsize=tick_size)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():   # leg.prop alone doesn't resize existing labels
                text.set_fontsize(legend_size)
    fig.tight_layout()
    return fig, axes


# --------------------------------------------------------------------------- #
# KDE contour helper
# --------------------------------------------------------------------------- #
def find_level(z, prob):
    """Density value whose super-level set encloses `prob` of the total mass of
    grid z (e.g. prob=0.9 -> the 90% highest-density contour level)."""
    z_flat = z.flatten()
    z_sorted = np.sort(z_flat)[::-1]
    cumsum = np.cumsum(z_sorted)
    cumsum /= cumsum[-1]
    idx = np.searchsorted(cumsum, prob)
    idx = min(idx, len(z_sorted) - 1)
    return z_sorted[idx]


# --------------------------------------------------------------------------- #
# GB diagnostics (the Figure_11a_* and feature_importance_* cells)
# --------------------------------------------------------------------------- #
def plot_residual_diagnostics(bst, X, y, quantiles, *, xlabel, xticks, yticks,
                              panel_labels=("(a)", "(b)", "(c)"), fname=None,
                              show=True):
    """Median-prediction residual diagnostics for a multi-quantile xgb NTF:
    (1) normalized residual vs prediction, (2) residual histogram, (3) normal Q-Q.

    bst, quantiles : from ntf_fit.fit_ntf_xgb
    X, y           : the SAME arrays the model was fit on (e.g. X_wv, y_wv)
    xlabel         : e.g. "predicted wind vane values (deg)"
    xticks, yticks : tick arrays for panel 1, e.g. np.arange(-55, 30, 15)
    """
    import xgboost as xgb
    from scipy import stats

    all_quantile_preds = bst.predict(xgb.DMatrix(X))     # (n, n_quantiles)
    median_col = list(quantiles).index(0.50)
    predictions = all_quantile_preds[:, median_col]
    y = np.asarray(y, dtype=float)
    if y.shape != predictions.shape:
        raise ValueError(f"y has shape {y.shape} but predictions have shape "
                         f"{predictions.shape}; pass the target the model was fit on")

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(27, 9))
    axes[0].plot(predictions, ([0] * len(predictions)), color="black", linewidth=5)
    residuals = y - predictions
    normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    print(np.mean(residuals))
    print(np.std(residuals))

    axes[0].scatter(predictions, normalized_residuals, alpha=0.25, s=50, color="teal",
                    marker="o", edgecolor="k")
    axes[0].set_xlabel(xlabel, fontsize=40)
    axes[0].set_xlim(min(predictions) - 1, max(predictions) + 1)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(axes[0].get_xticks(), fontsize=35)
    axes[0].set_ylabel("normalized residual", fontsize=45)
    axes[0].set_ylim(min(normalized_residuals) - 1, max(normalized_residuals) + 1)
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=35)

    axes[1].hist(normalized_residuals, bins=40, orientation="horizontal", density=True,
                 alpha=0.25, color="teal", edgecolor="black")
    axes[1].set_xlabel("density", fontsize=45)
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks(np.arange(0, 1.5, 0.5))
    axes[1].set_xticklabels(axes[1].get_xticks(), fontsize=35)

    stats.probplot(normalized_residuals, dist="norm", plot=axes[2])
    axes[2].set_xlabel("theoretical quantiles", fontsize=40)
    axes[2].set_xlim(-2, 2)
    axes[2].set_xticks(np.arange(-2, 4, 2))
    axes[2].set_xticklabels(axes[2].get_xticks(), fontsize=35)
    axes[2].set_ylabel("")
    axes[2].set_title("")

    for i in range(3):
        axes[i].text(0.02, 0.95, panel_labels[i], transform=axes[i].transAxes,
                     fontsize=40, fontweight="bold", va="top", ha="left",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    for line in axes[2].lines:
        if line.get_linestyle() == "-":      # theoretical quantile line
            line.set_color("red")
            line.set_linewidth(5)
        elif line.get_linestyle() == "":     # points
            line.set_marker("o")
            line.set_markerfacecolor("blue")
            line.set_alpha(0.7)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_feature_importance(bst, feature_names, label_map, *, panel_label="(a)",
                            fname=None, show=True):
    """Bar chart of xgboost gain importance, as % of total gain.

    feature_names : the model's input order (e.g. windvane_inputs)
    label_map     : display names, e.g. {"power": "Power", "ti": "TI",
                    "windvane": "Wind Vane"}
    """
    import seaborn as sns

    importance = bst.get_score(importance_type="gain")
    importance_named = {feature_names[int(k[1:])]: v for k, v in importance.items()}
    total = sum(importance_named.values())

    records = [{"feature": feat,
                "percent": 100 * importance_named.get(feat, 0.0) / total if total > 0 else 0}
               for feat in feature_names]
    importance_all = pd.DataFrame(records)
    importance_all["feature"] = importance_all["feature"].map(label_map)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=importance_all, x="feature", y="percent",
                color=sns.color_palette("viridis", 1)[0], ax=ax)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Feature Importance (%)", fontsize=20)
    ax.set_xlabel("")
    ax.tick_params(labelsize=21)
    ax.text(0.02, 0.95, panel_label, transform=ax.transAxes, fontsize=25,
            fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
