"""Draw the README's charts from the recorded evidence.

Nothing here invents a number. Every value is read from `bench/baseline.json`,
`bench/performance.json`, `bench/three_way.json` or `bench/phases.json`, so a chart cannot drift from the run it claims to
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
        # The three-package charts. Slots 1-3 of the reference categorical
        # palette -- the only three that validate all-pairs -- checked against
        # GitHub's own README surface in each mode:
        #   node validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light
        #        --pairs all --surface "#ffffff"         -> all checks pass
        # Light aqua sits under 3:1 on white, so every line also carries a
        # distinct marker shape and a direct label: identity is never hue alone.
        "surface": "#ffffff", "primary": "#0b0b0b", "secondary": "#52514e",
        "hairline": "#e1e0d9",
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
        "st_neutral": "#c3c2b7", "st_good": "#0ca30c",
        "st_warning": "#fab219", "st_serious": "#ec835a",
        "st_critical": "#d03b3b",
    },
    "dark": {
        "ink": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
        "ours": "#58a6ff", "theirs": "#6e7681", "accent": "#ff8a4c",
        "good": "#3fb950", "warn": "#d29922", "bad": "#f85149",
        "neutral": "#484f58",
        #   node validate_palette.js "#3987e5,#d95926,#199e70" --mode dark
        #        --pairs all --surface "#0d1117"         -> all checks pass
        "surface": "#0d1117", "primary": "#ffffff", "secondary": "#c3c2b7",
        "hairline": "#2c2c2a",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
        "st_neutral": "#5c5b56", "st_good": "#0ca30c",
        "st_warning": "#fab219", "st_serious": "#ec835a",
        "st_critical": "#d03b3b",
    },
}

# One identity per package, shared by every three-package chart. Colour follows
# the package, never its rank, and the marker shape is the second channel.
PACKAGES = (
    ("mixedlm-rs", "mixedlm_rs", "s1", "o"),
    ("lme4", "lme4", "s2", "s"),
    ("statsmodels", "statsmodels", "s3", "^"),
)


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
                color=theme["ours"], fontsize=11, fontweight="bold")

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
                        fontweight="bold", zorder=4)
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


def _seconds(value, _pos=None):
    """Three significant figures: a fourth digit of a timing is noise."""
    if value >= 1:
        return f"{value:.3g} s"
    return f"{value * 1000:.3g} ms"


def _count(value, _pos=None):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _chrome(ax, theme):
    """Hairline, solid, recessive: the data is the only loud thing."""
    ax.grid(True, which="major", color=theme["hairline"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["hairline"])
    ax.tick_params(which="both", length=0, colors=theme["secondary"])


def three_way_scaling_chart(three, theme, path):
    """Fit time against group count for all three packages, on one axis."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    rows = three["timing"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    handles = []
    for label, key, slot, marker in PACKAGES:
        pts = [(r["groups"], r[key]["seconds"]) for r in rows if r.get(key)]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        style_kw = {"color": theme[slot], "linewidth": 2, "marker": marker,
                    "markersize": 7, "markeredgecolor": theme["surface"],
                    "markeredgewidth": 1.6}
        # A 2px ring in the surface colour keeps a marker legible where it
        # crosses another series.
        ax.plot(xs, ys, zorder=3, **style_kw)
        # A direct label at the line's end, in ink rather than series colour:
        # it names the line beside it and carries the one value worth reading.
        ax.annotate(f"{label}  {_seconds(ys[-1])}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(9, 0), va="center",
                    color=theme["primary"], fontsize=10)
        handles.append(Line2D([], [], label=label, **style_kw))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(60, max(r["groups"] for r in rows) * 9)
    ax.xaxis.set_major_formatter(FuncFormatter(_count))
    ax.yaxis.set_major_formatter(FuncFormatter(_seconds))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("groups in the random-effects factor (log scale)",
                  color=theme["secondary"])
    ax.set_ylabel("time to fit (log scale)", color=theme["secondary"])
    _chrome(ax, theme)

    legend = ax.legend(handles=handles, loc="upper left", frameon=False,
                       fontsize=10, handlelength=2.2)
    for text in legend.get_texts():
        text.set_color(theme["primary"])

    fig.tight_layout()
    fig.savefig(path, transparent=True, dpi=200)
    plt.close(fig)
    return True


# The outcome of one Python fit, judged against lme4's optimum on the same
# file. Status colours, because these outcomes do mean good or bad; each is
# also named in the legend, so none is read from colour alone.
LME4_OUTCOMES = (
    ("matches lme4", "st_neutral"),
    ("higher likelihood than lme4", "st_good"),
    ("lower, and flagged not converged", "st_warning"),
    ("lower, but reported converged", "st_critical"),
    ("raised an error", "st_serious"),
)


def lme4_outcome(entry, tolerance):
    if entry.get("error"):
        return "raised an error"
    gap = entry.get("deviance_gap_to_lme4")
    if gap is None:
        return None                       # lme4 itself failed on this file
    if abs(gap) <= tolerance:
        return "matches lme4"
    if gap > 0:
        return "higher likelihood than lme4"
    if entry.get("converged"):
        return "lower, but reported converged"
    return "lower, and flagged not converged"


def lme4_outcome_counts(three, tolerance):
    counts = {key: dict.fromkeys((o for o, _ in LME4_OUTCOMES), 0)
              for _, key, _, _ in PACKAGES if key != "lme4"}
    for row in three["accuracy"]["rows"]:
        for key, tally in counts.items():
            outcome = lme4_outcome(row[key], tolerance)
            if outcome:
                tally[outcome] += 1
    return counts


def lme4_agreement_chart(three, theme, path):
    """Where each Python package lands relative to lme4, on the hard fixtures."""
    from matplotlib.patches import Patch

    sys.path.insert(0, str(ROOT / "bench"))
    import tolerances

    counts = lme4_outcome_counts(three, tolerances.DEVIANCE_ABS)
    order = [("mixedlm-rs", "mixedlm_rs"), ("statsmodels", "statsmodels")]
    total = max(sum(c.values()) for c in counts.values())
    # Dark ink where a light status fill would swallow white text.
    dark_ink = {"st_warning"} | ({"st_neutral"} if theme["surface"] == "#ffffff"
                                 else set())

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for y, (_label, key) in enumerate(reversed(order)):
        left = 0
        for outcome, colour in LME4_OUTCOMES:
            value = counts[key][outcome]
            if not value:
                continue
            # An edge in the surface colour is the 2px gap between segments:
            # it separates without adding ink.
            ax.barh(y, value, left=left, height=0.56, color=theme[colour],
                    edgecolor=theme["surface"], linewidth=2, zorder=3)
            if value / total >= 0.06:
                ax.text(left + value / 2, y, str(value), ha="center",
                        va="center", fontsize=10.5, fontweight="bold",
                        color="#0b0b0b" if colour in dark_ink else "#ffffff",
                        zorder=4)
            left += value

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label for label, _ in reversed(order)],
                       color=theme["primary"], fontsize=11)
    ax.set_xlim(0, total)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # The legend doubles as the count table for segments too narrow to label.
    handles = []
    for outcome, colour in LME4_OUTCOMES:
        a, b = counts["mixedlm_rs"][outcome], counts["statsmodels"][outcome]
        if a or b:
            handles.append(Patch(color=theme[colour],
                                 label=f"{outcome}  ({a} / {b})"))
    legend = ax.legend(handles=handles, loc="upper left",
                       bbox_to_anchor=(0, -0.02), ncols=2, frameon=False,
                       fontsize=9.5, handlelength=1.1, columnspacing=1.6,
                       title="counts: mixedlm-rs / statsmodels",
                       title_fontsize=9, alignment="left")
    legend.get_title().set_color(theme["secondary"])
    for text in legend.get_texts():
        text.set_color(theme["primary"])

    fig.tight_layout()
    fig.savefig(path, transparent=True, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


# The phase benchmark's cases, in the words a reader would use for them.
RELEASE_CASES = (
    ("slope-18", "random slope, 18 groups"),
    ("slope-500", "random slope, 500 groups"),
    ("slope-50k", "random slope, 50k groups"),
    ("slope-125k", "random slope, 125k groups"),
    ("slope-20k-p30", "random slope, 20k groups, 30 fixed effects"),
    ("intercept-125k", "random intercept, 125k groups"),
    ("unbalanced-125k", "unbalanced intercept, 125k groups"),
    ("visits-125k", "slope in visit number, 125k groups"),
)


def release_speedup_chart(phases, theme, path):
    """How much faster a full fit is than the previous release, per case."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    builds = phases["builds"]
    threads = [t for t in phases["settings"]["threads"]
               if t in builds["candidate"]["threads"]]
    cases = [(key, label) for key, label in RELEASE_CASES
             if key in builds["candidate"]["threads"][threads[0]]]
    # Colour, marker and label side per thread count: 1 thread above the
    # row, many below, so two close values never print over each other.
    marks = {threads[0]: ("s1", "o", 7, "bottom"),
             threads[-1]: ("s2", "s", -8, "top")}

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(cases) + 1.4))
    handles = []
    for t in threads:
        slot, marker, dy, va = marks[t]
        xs = [builds["baseline"]["threads"][t][k]["fit"]["median_ms"]
              / builds["candidate"]["threads"][t][k]["fit"]["median_ms"]
              for k, _ in cases]
        ys = range(len(cases) - 1, -1, -1)
        style_kw = {"color": theme[slot], "marker": marker, "markersize": 8,
                    "markeredgecolor": theme["surface"],
                    "markeredgewidth": 1.4, "linestyle": "none"}
        ax.plot(xs, list(ys), zorder=3, **style_kw)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{x:.1f}x", (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", va=va, fontsize=8.5,
                        color=theme["secondary"])
        name = "1 thread" if t == "1" else f"{t} threads"
        handles.append(Line2D([], [], label=name, **style_kw))

    ax.axvline(1, color=theme["secondary"], linewidth=1, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 10)
    ax.set_xticks([1, 2, 4, 8])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}x"))
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([label for _, label in reversed(cases)],
                       color=theme["primary"], fontsize=9.5)
    ax.set_ylim(-0.6, len(cases) - 0.3)
    ax.set_xlabel(f"full fit, times faster than {builds['baseline']['version']}"
                  " (log scale)", color=theme["secondary"])
    _chrome(ax, theme)
    ax.grid(False, axis="y")

    legend = ax.legend(handles=handles, loc="lower right",
                       bbox_to_anchor=(1, 1), ncols=2, frameon=False,
                       fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(theme["primary"])

    fig.tight_layout()
    fig.savefig(path, transparent=True, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return True


def _optional(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


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
    three = _optional(ROOT / "bench" / "three_way.json")
    phases = _optional(ROOT / "bench" / "phases.json")

    written = []
    for name, theme in THEMES.items():
        style(theme)
        target = ASSETS / f"scaling-{name}.svg"
        if scaling_chart(perf, theme, target):
            written.append(target)
        target = ASSETS / f"outcomes-{name}.svg"
        if outcomes_chart(baseline, theme, target):
            written.append(target)
        # The optional records: each chart is drawn only if its run exists.
        if three and three.get("timing"):
            target = ASSETS / f"three-way-scaling-{name}.svg"
            if three_way_scaling_chart(three, theme, target):
                written.append(target)
        if three and three.get("accuracy"):
            target = ASSETS / f"lme4-agreement-{name}.svg"
            if lme4_agreement_chart(three, theme, target):
                written.append(target)
        if phases:
            target = ASSETS / f"release-speedup-{name}.svg"
            if release_speedup_chart(phases, theme, target):
                written.append(target)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)} "
              f"({path.stat().st_size / 1024:.0f} KB)")
    print(f"\nfrom baseline {baseline['environment']['commit'][:10]} "
          f"and performance {perf['environment']['commit'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
