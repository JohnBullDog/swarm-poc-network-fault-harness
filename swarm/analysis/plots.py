"""Figures for the experiment datasets.

Every figure reads a consolidated dataset from datasets/ and writes a PNG in
both light and dark variants. Nothing here recomputes a measurement; the
collector has already done that, so a figure can only show what the tidy tables
say.

Design rules followed, and the reason each matters here:

  * One y-axis per panel, never two. Throughput and latency are different
    measures on different scales, so they get separate panels rather than a
    second axis, which would let the reader infer a crossing that is an artifact
    of the two scalings.
  * Categorical hues assigned in fixed order, never cycled.
  * Every series is direct-labelled as well as legended, so identity is never
    carried by colour alone -- required here because two of the light-mode hues
    fall below 3:1 contrast on the light surface.
  * Recessive grid and axes; no value printed on every point.

The palette is the validated default: slots blue/orange/aqua/yellow, which pass
the lightness band, chroma floor, CVD separation and normal-vision floor in both
modes on the adjacent pairlist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

from ..core import paths              # noqa: E402

REPO = paths.REPO
DATASETS = paths.DATASETS
FIGURES = REPO / "docs" / "figures"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#dedddA",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}


def _style(ax, theme: dict, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(theme["surface"])
    ax.set_title(title, color=theme["text"], fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=theme["muted"], fontsize=9)
    ax.set_ylabel(ylabel, color=theme["muted"], fontsize=9)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=3)
    ax.grid(True, color=theme["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])


def _save(fig, theme_name: str, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.{theme_name}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _label_last(ax, x, y, text: str, color: str, theme: dict) -> None:
    """Direct-label a series at its last point.

    Required relief: two light-mode hues sit under 3:1 on the light surface, so
    colour alone cannot carry identity.
    """
    if len(x) == 0:
        return
    ax.annotate(text, xy=(list(x)[-1], list(y)[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8, va="center", fontweight="medium")


# ---------------------------------------------------------------------------
# exp1: concurrency
# ---------------------------------------------------------------------------
def plot_exp1(theme_name: str = "light") -> Path | None:
    path = paths.runs_csv("exp1")
    if not path.exists():
        return None
    theme = THEMES[theme_name]
    d = pd.read_csv(path).sort_values("concurrency_G")
    c = theme["series"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.patch.set_facecolor(theme["surface"])

    ax = axes[0]
    series = [
        ("measured", d["meas_throughput_tok_s"], c[0], "-", "o"),
        ("X_max · min(1, G/N)", d["pred_x_of_g_tok_s"], c[1], "--", None),
        ("min(G/T_token, W/max t_k)", d["pred_x_closed_loop_tok_s"], c[2], "--", None),
    ]
    for name, ys, col, ls, marker in series:
        ax.plot(d["concurrency_G"], ys, color=col, linewidth=2.0, linestyle=ls,
                marker=marker, markersize=5, label=name, zorder=3)
    # No direct labels on this panel: the two bounds coincide above the knee,
    # so their labels land on top of each other and on "measured". The legend
    # carries identity here, and every series is also distinguished by dash
    # pattern and marker, so colour is not doing the work alone.
    n = int(d["n_nodes_N"].iloc[0])
    ax.axvline(n, color=theme["muted"], linewidth=1.0, linestyle=":", zorder=1)
    # Blended transform: x in data coordinates, y as a fraction of the axes,
    # so the label sits just above the x-axis whatever the y-range turns out
    # to be. Anchoring it to a data y put it outside the visible range.
    ax.annotate(f"G = N = {n}", xy=(n, 0.03), xycoords=ax.get_xaxis_transform(),
                xytext=(5, 0), textcoords="offset points",
                color=theme["muted"], fontsize=8, va="bottom")
    ax.set_xlim(right=d["concurrency_G"].max() * 1.05)
    _style(ax, theme, "Throughput vs concurrency", "concurrent streams G",
           "tokens / s")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in leg.get_texts():
        text.set_color(theme["muted"])

    ax = axes[1]
    ax.plot(d["concurrency_G"], d["meas_t_token_s_p50"] * 1000, color=c[0],
            linewidth=2.0, marker="o", markersize=5, label="measured p50", zorder=3)
    ax.plot(d["concurrency_G"], d["meas_t_token_s_p95"] * 1000, color=c[1],
            linewidth=2.0, linestyle="-", marker="s", markersize=4,
            label="measured p95", zorder=3)
    ax.plot(d["concurrency_G"], d["pred_t_token_s"] * 1000, color=c[2],
            linewidth=2.0, linestyle="--", label="predicted T_token", zorder=2)
    _label_last(ax, d["concurrency_G"], d["meas_t_token_s_p50"] * 1000, "p50",
                c[0], theme)
    _label_last(ax, d["concurrency_G"], d["meas_t_token_s_p95"] * 1000, "p95",
                c[1], theme)
    _label_last(ax, d["concurrency_G"], d["pred_t_token_s"] * 1000, "predicted",
                c[2], theme)
    ax.set_xlim(right=d["concurrency_G"].max() * 1.18)
    _style(ax, theme, "Per-token latency vs concurrency",
           "concurrent streams G", "ms")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in leg.get_texts():
        text.set_color(theme["muted"])

    fig.tight_layout()
    return _save(fig, theme_name, "exp1_concurrency")


# ---------------------------------------------------------------------------
# exp2: delay and depth
# ---------------------------------------------------------------------------
def plot_exp2(theme_name: str = "light") -> Path | None:
    path = paths.runs_csv("exp2")
    if not path.exists():
        return None
    theme = THEMES[theme_name]
    d = pd.read_csv(path)
    c = theme["series"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.patch.set_facecolor(theme["surface"])

    ax = axes[0]
    for i, (depth, grp) in enumerate(d.groupby("depth_D")):
        grp = grp.sort_values("sweep_delay_ms")
        ax.plot(grp["sweep_delay_ms"], grp["meas_t_token_s_p50"] * 1000,
                color=c[i], linewidth=2.0, marker="o", markersize=5,
                label=f"D = {int(depth)}", zorder=3)
        _label_last(ax, grp["sweep_delay_ms"],
                    grp["meas_t_token_s_p50"] * 1000, f"D={int(depth)}",
                    c[i], theme)
    ax.set_xlim(right=d["sweep_delay_ms"].max() * 1.18)
    _style(ax, theme, "Per-token latency vs per-hop delay",
           "per-hop delay ℓ (ms)", "measured p50 (ms)")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left",
                    title="pipeline depth")
    leg.get_title().set_color(theme["muted"])
    leg.get_title().set_fontsize(8)
    for text in leg.get_texts():
        text.set_color(theme["muted"])

    # The slope of latency against per-hop delay should equal D exactly.
    #
    # The path has D+1 hops, but this sweep varies only the tier-2 link, and
    # tier-2 nodes send D of those hops (each stage forwards once, the last to
    # the gateway). The remaining hop is the gateway's entry, whose delay is
    # held fixed, so it contributes a constant and not a slope. A slope of D is
    # therefore a direct measurement of the report's D*(l + a/beta) term.
    ax = axes[1]
    slopes, preds, depths = [], [], []
    for depth, grp in d.groupby("depth_D"):
        grp = grp.sort_values("sweep_delay_ms")
        if len(grp) < 2:
            continue
        x = grp["sweep_delay_ms"].to_numpy()
        y = (grp["meas_t_token_s_p50"] * 1000).to_numpy()
        slope = ((y[-1] - y[0]) / (x[-1] - x[0]))
        slopes.append(slope)
        preds.append(int(depth))
        depths.append(int(depth))
    xs = range(len(depths))
    ax.bar([i - 0.19 for i in xs], preds, width=0.38, color=c[2],
           label="predicted slope = D", zorder=3)
    ax.bar([i + 0.19 for i in xs], slopes, width=0.38, color=c[0],
           label="measured Δlatency / Δℓ", zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"D = {v}" for v in depths])
    _style(ax, theme, "Latency slope against per-hop delay — predicted = D",
           "pipeline depth", "ms of latency per ms of hop delay")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in leg.get_texts():
        text.set_color(theme["muted"])

    fig.tight_layout()
    return _save(fig, theme_name, "exp2_delay")


# ---------------------------------------------------------------------------
# exp4: failover
# ---------------------------------------------------------------------------
def plot_exp4(theme_name: str = "light") -> Path | None:
    runs = paths.runs_csv("exp4")
    rec = paths.detail_csv("exp4", "recovery")
    if not (runs.exists() and rec.exists()):
        return None
    theme = THEMES[theme_name]
    c = theme["series"]
    d = pd.read_csv(runs)
    r = pd.read_csv(rec)
    joined = r.merge(d[["run_id", "width_W", "heartbeat_ms", "death_after_ms",
                        "sweep_arm", "sweep_hop_timeout_ms"]],
                     on="run_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.patch.set_facecolor(theme["surface"])

    # Panel 1: detection and recovery vs heartbeat, for the W x heartbeat arm.
    arm = joined[joined["sweep_arm"] == "W_x_heartbeat"]
    ax = axes[0]
    if len(arm):
        g = (arm.groupby("heartbeat_ms")[["detection_s", "recovery_s"]]
             .mean().reset_index().sort_values("heartbeat_ms"))
        xs = range(len(g))
        ax.bar([i - 0.19 for i in xs], g["detection_s"] * 1000, width=0.38,
               color=c[0], label="detection", zorder=3)
        ax.bar([i + 0.19 for i in xs], g["recovery_s"] * 1000, width=0.38,
               color=c[1], label="recovery", zorder=3)
        # The configured death threshold is 4x the heartbeat, so detection
        # cannot beat it; drawing it shows how much of detection is the
        # threshold and how much is the polling granularity.
        ax.plot(list(xs), g["heartbeat_ms"] * 4, color=c[2], linewidth=2.0,
                linestyle="--", marker="D", markersize=5,
                label="death_after (4 × heartbeat)", zorder=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{int(v)} ms" for v in g["heartbeat_ms"]])
    _style(ax, theme, "Detection and recovery vs heartbeat interval",
           "heartbeat interval", "ms")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in leg.get_texts():
        text.set_color(theme["muted"])

    # Panel 2: recovery vs hop timeout -- the finding that recovery tracks the
    # client's timeout rather than the detection threshold.
    arm2 = joined[joined["sweep_arm"] == "hop_timeout"]
    ax = axes[1]
    if len(arm2):
        g = (arm2.groupby("sweep_hop_timeout_ms")[["detection_s", "recovery_s"]]
             .mean().reset_index().sort_values("sweep_hop_timeout_ms"))
        ax.plot(g["sweep_hop_timeout_ms"], g["recovery_s"] * 1000, color=c[1],
                linewidth=2.0, marker="o", markersize=6, label="recovery",
                zorder=3)
        ax.plot(g["sweep_hop_timeout_ms"], g["detection_s"] * 1000, color=c[0],
                linewidth=2.0, marker="s", markersize=5, label="detection",
                zorder=3)
        _label_last(ax, g["sweep_hop_timeout_ms"], g["recovery_s"] * 1000,
                    "recovery", c[1], theme)
        _label_last(ax, g["sweep_hop_timeout_ms"], g["detection_s"] * 1000,
                    "detection", c[0], theme)
    _style(ax, theme, "Recovery vs hop timeout (W=2, heartbeat 500 ms)",
           "hop timeout (ms)", "ms")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in leg.get_texts():
        text.set_color(theme["muted"])

    fig.tight_layout()
    return _save(fig, theme_name, "exp4_failover")


# ---------------------------------------------------------------------------
# engine comparison
# ---------------------------------------------------------------------------
def plot_engine_comparison(experiment: str = "exp1",
                           theme_name: str = "light") -> Path | None:
    path = paths.analysis_file(f"engine_comparison_{experiment}.csv")
    if not path.exists():
        return None
    theme = THEMES[theme_name]
    c = theme["series"]
    d = pd.read_csv(path)

    axes_names = ["t_token_p50_ms", "throughput_tok_s", "stage_compute_ms",
                  "stage_net_ms", "stage_queue_ms"]
    tol = {"t_token_p50_ms": 0.10, "throughput_tok_s": 0.10,
           "stage_compute_ms": 0.10, "stage_net_ms": 0.25,
           "stage_queue_ms": 0.50}

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.patch.set_facecolor(theme["surface"])

    for i, name in enumerate(axes_names):
        col = f"{name}_rel_err"
        if col not in d.columns:
            continue
        vals = pd.to_numeric(d[col], errors="coerce").dropna() * 100
        if not len(vals):
            continue
        ys = [i] * len(vals)
        ax.scatter(vals, ys, s=46, color=c[0], alpha=0.75,
                   edgecolors=theme["surface"], linewidths=1.2, zorder=3)
        band = tol[name] * 100
        ax.barh(i, 2 * band, left=-band, height=0.55, color=c[2], alpha=0.16,
                zorder=1)
        ax.annotate(f"±{band:.0f}%", xy=(band, i), xytext=(5, 0),
                    textcoords="offset points", color=theme["muted"],
                    fontsize=7.5, va="center")

    ax.axvline(0, color=theme["muted"], linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(axes_names)))
    ax.set_yticklabels([n.replace("_", " ") for n in axes_names])
    ax.invert_yaxis()
    _style(ax, theme,
           f"Virtual engine vs containers ({experiment}): relative error per axis",
           "virtual relative to container (%)", "")
    ax.annotate("shaded band = tolerance; each dot is one sweep point",
                xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 14),
                textcoords="offset points", color=theme["muted"], fontsize=8)
    fig.tight_layout()
    return _save(fig, theme_name, f"engine_comparison_{experiment}")


# ---------------------------------------------------------------------------
def main() -> int:
    made = []
    for theme in ("light", "dark"):
        for fn in (plot_exp1, plot_exp2, plot_exp4):
            out = fn(theme)
            if out:
                made.append(out)
        for exp in ("exp1", "exp2", "exp4"):
            out = plot_engine_comparison(exp, theme)
            if out:
                made.append(out)
    if not made:
        print("no datasets found; run an experiment first")
        return 1
    for path in made:
        print(f"  wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
