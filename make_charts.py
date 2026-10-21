#!/usr/bin/env python3
"""Turn results/*_main.* into the PNG charts in results/figures/.

"Time to first kernel" = every measured stage up to and including the first
kernel launch + sync (second_launch is excluded; it happens after).
"""
import csv
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAG = "main"

# Validated categorical slots (light) + neutral for "other"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#b9b8b2"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": GRID,
})

FOOTNOTE = ("RTX 3050 4GB Laptop · driver 615.71 · CUDA 13.4 · i5-13420H · "
            "median of fresh processes · github: cuda-coldstart")


def load_summary():
    out = {}
    with open(os.path.join(RES, f"summary_{TAG}.csv")) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["name"], {})[r["stage"]] = float(r["median_ms"])
    return out


def load_wake():
    recs = [json.loads(l) for l in open(os.path.join(RES, f"wake_{TAG}.jsonl"))]
    out = {}
    for state in ["suspended", "active", "other_client"]:
        rs = [r for r in recs if r["program"] == "driver" and r["state"] == state]
        out[state] = {k: statistics.median(r["stages"][k] for r in rs)
                      for k in rs[0]["stages"]}
    return out


def time_to_first(stages):
    return sum(v for k, v in stages.items()
               if k not in ("second_launch", "sum_of_stages", "process_wall"))


def split(stages):
    """Group driver stages into the four buckets shown in the hero chart."""
    init = stages.get("cuInit", 0)
    ctx = stages.get("context_create", 0)
    compile_ = (stages.get("load_libnvrtc", 0) + stages.get("nvrtc_compile", 0)
                + stages.get("module_load", 0))
    other = time_to_first(stages) - init - ctx - compile_
    return [init, ctx, compile_, other]


def fmt_ms(ms):
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


def style_axes(ax):
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    # Positions in inches from the top so every figure height looks the same
    h = fig.get_figheight()
    fig.text(0.035, 1 - 0.45 / h, title, fontsize=22, fontweight="bold",
             color=INK, va="top")
    fig.text(0.035, 1 - 0.95 / h, subtitle, fontsize=13, color=INK2, va="top")
    fig.text(0.035, 0.18 / h, FOOTNOTE, fontsize=9.5, color=MUTED)


def hero_chart(summary, wake):
    rows = [
        ("GPU asleep (D3cold)", "1 kernel · SASS", wake["suspended"]),
        ("GPU awake, idle", "1 kernel · SASS", wake["active"]),
        ("GPU awake + another CUDA app", "1 kernel · SASS",
         wake["other_client"]),
        ("PTX only, empty JIT cache", "256 kernels",
         summary["driver/ptx/k256/-/cache-cold"]),
        ("PTX only, warm JIT cache", "256 kernels",
         summary["driver/ptx/k256/-/cache-warm"]),
        ("NVRTC compile at runtime", "256 kernels → SASS",
         summary["driver/nvrtc-cubin/k256/-/cache-none"]),
    ]
    buckets = [("cuInit (incl. GPU wake-up)", BLUE),
               ("Context creation", ORANGE),
               ("Compile / JIT / module load", AQUA),
               ("Everything else (alloc, copy, the kernel)", GRAY)]

    fig = plt.figure(figsize=(12, 7.4), dpi=150)
    ax = fig.add_axes([0.30, 0.10, 0.62, 0.66])
    header(fig, "What happens before your first CUDA kernel runs?",
           "Time from main() to the first kernel's result, broken down by "
           "stage. The kernel itself takes 0.4 ms.")

    # Two groups with a gap: power state (top), how code is shipped (bottom)
    ys = [7.0, 6.0, 5.0, 2.6, 1.6, 0.6]
    max_total = max(time_to_first(r[2]) for r in rows)
    for (label, sub, stages), y in zip(rows, ys):
        left = 0
        for (name, color), ms in zip(buckets, split(stages)):
            if ms <= 0:
                continue
            ax.barh(y, ms, left=left, height=0.62, color=color,
                    edgecolor=SURFACE, linewidth=2)
            if ms > max_total * 0.07:
                ax.text(left + ms / 2, y, fmt_ms(ms), ha="center",
                        va="center", fontsize=10, color="white",
                        fontweight="bold")
            left += ms
        ax.text(left + max_total * 0.012, y, fmt_ms(left), va="center",
                fontsize=12.5, fontweight="bold", color=INK)
        ax.text(-max_total * 0.02, y + 0.12, label, ha="right", va="center",
                fontsize=12, color=INK)
        ax.text(-max_total * 0.02, y - 0.2, sub, ha="right", va="center",
                fontsize=10, color=MUTED)

    ax.text(-max_total * 0.02, 7.85, "LAPTOP GPU POWER STATE", ha="right",
            fontsize=9.5, color=MUTED, fontweight="bold")
    ax.text(-max_total * 0.02, 3.45, "HOW THE KERNELS ARE SHIPPED", ha="right",
            fontsize=9.5, color=MUTED, fontweight="bold")

    style_axes(ax)
    ax.set_yticks([])
    ax.set_ylim(0.0, 8.1)
    ax.set_xlim(0, max_total * 1.13)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f} s"))

    handles = [matplotlib.patches.Patch(color=c, label=n) for n, c in buckets]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False,
               bbox_to_anchor=(0.03, 0.83), fontsize=10.5,
               handlelength=1.2, columnspacing=1.6)
    fig.savefig(os.path.join(FIG, "1_hero_breakdown.png"))
    plt.close(fig)


def scaling_chart(summary):
    counts = [1, 32, 256]
    # (name, key, color, label nudge in ms so near-equal end labels don't
    # collide: SASS and warm-cache PTX finish within a few ms of each other)
    series = [
        ("SASS / cubin (precompiled)", "driver/cubin/k{}/-/cache-none", BLUE,
         -110),
        ("PTX, warm JIT cache", "driver/ptx/k{}/-/cache-warm", ORANGE, 110),
        ("PTX, empty JIT cache", "driver/ptx/k{}/-/cache-cold", AQUA, 0),
        ("NVRTC → SASS at runtime", "driver/nvrtc-cubin/k{}/-/cache-none",
         YELLOW, 0),
    ]
    fig = plt.figure(figsize=(12, 7.4), dpi=150)
    ax = fig.add_axes([0.08, 0.12, 0.62, 0.62])
    header(fig, "Ship SASS, or pay for the compiler at startup",
           "Time to first kernel vs. number of kernels in the module "
           "(Driver API, GPU awake).")

    xs = list(range(len(counts)))
    for name, key, color, nudge in series:
        ys = [time_to_first(summary[key.format(n)]) for n in counts]
        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=name)
        ax.plot([xs[-1] + 0.03, xs[-1] + 0.07], [ys[-1], ys[-1] + nudge],
                color=color, linewidth=1.5)
        ax.text(xs[-1] + 0.08, ys[-1] + nudge, f"{name} · {fmt_ms(ys[-1])}",
                va="center", fontsize=11, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=2,
              bbox_to_anchor=(-0.01, 1.11), handlelength=1.6)

    style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(xs, [f"{n} kernel{'s' if n > 1 else ''}" for n in counts],
                  fontsize=11.5)
    ax.set_xlim(-0.15, len(counts) - 0.85)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f} s"))
    fig.savefig(os.path.join(FIG, "2_jit_scaling.png"))
    plt.close(fig)


def lazy_chart(summary):
    rows = [
        ("CUDA_MODULE_LOADING=EAGER", summary["runtime/ptx/k256/EAGER/cache-cold"]),
        ("CUDA_MODULE_LOADING=LAZY", summary["runtime/ptx/k256/LAZY/cache-cold"]),
        ("LAZY + SASS in the binary", summary["runtime/sass/k256/LAZY/cache-none"]),
    ]
    parts = [("runtime_init", "During init (cudaFree(0))", BLUE),
             ("first_launch", "During the first kernel launch", ORANGE)]

    fig = plt.figure(figsize=(12, 6.2), dpi=150)
    ax = fig.add_axes([0.27, 0.13, 0.63, 0.52])
    header(fig, "Lazy loading moves the JIT bill. It doesn't cancel it.",
           "Runtime API, PTX-only binary with 256 kernels, empty JIT cache. "
           "Only one kernel is ever launched.")

    max_total = max(time_to_first(r[1]) for r in rows)
    for y, (label, st) in zip([2, 1, 0], rows):
        left = 0
        for key, _, color in parts:
            ms = st[key]
            ax.barh(y, ms, left=left, height=0.6, color=color,
                    edgecolor=SURFACE, linewidth=2)
            if ms > max_total * 0.08:
                ax.text(left + ms / 2, y, fmt_ms(ms), ha="center",
                        va="center", fontsize=10.5, color="white",
                        fontweight="bold")
            left += ms
        total = time_to_first(st)
        ax.text(total + max_total * 0.012, y, fmt_ms(total), va="center",
                fontsize=12.5, fontweight="bold")
        ax.text(-max_total * 0.02, y, label, ha="right", va="center",
                fontsize=11.5, family="DejaVu Sans Mono")

    style_axes(ax)
    ax.set_yticks([])
    ax.set_xlim(0, max_total * 1.12)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.1f} s"))
    handles = [matplotlib.patches.Patch(color=c, label=n) for _, n, c in parts]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
               bbox_to_anchor=(0.03, 0.80), fontsize=10.5, handlelength=1.2)
    fig.savefig(os.path.join(FIG, "3_lazy_loading.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    summary, wake = load_summary(), load_wake()
    hero_chart(summary, wake)
    scaling_chart(summary)
    lazy_chart(summary)
    print("wrote", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
