"""Draw the README's charts from the recorded evidence.

Nothing here invents a number. Every value is read from `bench/baseline.json`
or `bench/performance.json`, so a chart cannot drift from the run it claims to
depict -- the same discipline the tables are held to, applied to the pictures.

    python bench/make_charts.py

Writes light and dark variants into `docs/assets/`. GitHub renders a README on
either background, and `<picture>` with a `prefers-color-scheme` media query
picks the right one; a single mid-tone chart that "works on both" is legible on
neither.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

# One hue carries the subject, grey carries the context. Not a categorical
# palette: these charts each have one point to make, and cycling hues would
# give equal visual weight to the thing being compared against.
THEMES = {
    "light": {
        "ink": "#24292f", "muted": "#57606a", "grid": "#d8dee4",
        "ours": "#2a78d6", "theirs": "#8c959f", "accent": "#d95926",
        "good": "#1baf7a", "warn": "#eda100", "bad": "#e34948",
        "neutral": "#b1bac4",
    },
    "dark": {
        "ink": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
        "ours": "#58a6ff", "theirs": "#6e7681", "accent": "#ff8a4c",
        "good": "#3fb950", "warn": "#d29922", "bad": "#f85149",
        "neutral": "#484f58",
    },
}


def style(theme):
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": theme["ink"],
        "axes.labelcolor": theme["muted"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "axes.edgecolor": theme["grid"],
        "font.size": 10,
        "font.family": "DejaVu Sans",
    })


def strip(ax, theme):
    """Recessive chrome: the data should be the only loud thing."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(length=0)


def scaling_chart(perf, theme, path):
    rows = [r for r in perf["measured"]["rows"] if r.get("speedup")]
    if not rows:
        return False
    groups = [r["groups"] for r in rows]
    ours = [r["ours_seconds"] for r in rows]
    theirs = [r["statsmodels_seconds"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    ax.plot(groups, theirs, marker="o", markersize=5, linewidth=2,
            color=theme["theirs"], label="statsmodels", zorder=2)
    ax.plot(groups, ours, marker="o", markersize=5, linewidth=2.4,
            color=theme["ours"], label="mixedlm-rs", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grouping factor levels")
    ax.set_ylabel("time to fit (seconds)")
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    strip(ax, theme)

    # Direct labels rather than a legend: two series, so a legend is a lookup
    # table for something the reader can already see.
    ax.annotate("statsmodels", (groups[-1], theirs[-1]),
                textcoords="offset points", xytext=(-6, 12), ha="right",
                color=theme["theirs"], fontsize=11)
    ax.annotate("mixedlm-rs", (groups[-1], ours[-1]),
                textcoords="offset points", xytext=(-6, -18), ha="right",
                color=theme["ours"], fontsize=11, fontweight="medium")

    best = max(rows, key=lambda r: r["speedup"])
    ax.annotate(f"{best['speedup']:.0f}x at {best['groups']:,} groups",
                (best["groups"], best["ours_seconds"]),
                textcoords="offset points", xytext=(10, 22),
                color=theme["accent"], fontsize=10,
                arrowprops={"arrowstyle": "-", "color": theme["accent"],
                            "linewidth": 1})

    fig.tight_layout()
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    return True


def outcomes_chart(baseline, theme, path):
    counts = baseline["differential"]["counts"]
    cases = baseline["differential"]["cases"]
    order = [
        ("we found a better optimum", "better optimum", theme["good"]),
        ("same optimum", "same optimum", theme["neutral"]),
        ("reference did not converge", "statsmodels did not converge",
         theme["warn"]),
        ("we found a worse optimum", "worse optimum", theme["bad"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 1.7), dpi=200)
    left = 0
    for key, _label, colour in order:
        value = counts.get(key, 0)
        if value:
            # A 2px surface gap between segments rather than a stroke.
            ax.barh([0], [value], left=left, height=0.52, color=colour,
                    edgecolor="none", zorder=3)
            if value / cases > 0.06:
                ax.text(left + value / 2, 0, str(value), ha="center",
                        va="center", color="#ffffff", fontsize=11,
                        fontweight="medium", zorder=4)
        left += value

    ax.set_xlim(0, cases)
    ax.set_ylim(-0.55, 0.85)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    for key, label, colour in order:
        value = counts.get(key, 0)
        ax.plot([], [], color=colour, linewidth=6, label=f"{label} — {value}")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0, 0.05), ncols=2,
                       frameon=False, fontsize=9.5, handlelength=1.1,
                       borderpad=0, columnspacing=1.6, handletextpad=0.6)
    for text in legend.get_texts():
        text.set_color(theme["muted"])

    fig.tight_layout()
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    return True


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "bench" / "baseline.json"
    perf_path = ROOT / "bench" / "performance.json"

    missing = [p.name for p in (baseline_path, perf_path) if not p.is_file()]
    if missing:
        print(f"cannot draw from recorded evidence; missing {missing}. "
              "Run bench/baseline.py and bench/performance.py first.",
              file=sys.stderr)
        return 1

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    perf = json.loads(perf_path.read_text(encoding="utf-8"))

    written = []
    for name, theme in THEMES.items():
        style(theme)
        target = ASSETS / f"scaling-{name}.svg"
        if scaling_chart(perf, theme, target):
            written.append(target)
        target = ASSETS / f"outcomes-{name}.svg"
        if outcomes_chart(baseline, theme, target):
            written.append(target)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)} "
              f"({path.stat().st_size / 1024:.0f} KB)")
    print(f"\nfrom baseline {baseline['environment']['commit'][:10]} "
          f"and performance {perf['environment']['commit'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
