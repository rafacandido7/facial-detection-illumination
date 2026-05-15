"""Generate figures for the article from LFW and proprio summary CSVs."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 200,
    }
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

lfw_csv = ROOT / "results" / "lfw_subset" / "summary.csv"
proprio_csv = ROOT / "results" / "proprio" / "summary.csv"

df_lfw = pd.read_csv(lfw_csv)
df_lfw.columns = [c.strip() for c in df_lfw.columns]

df_proprio = pd.read_csv(proprio_csv) if proprio_csv.exists() else None
if df_proprio is not None:
    df_proprio.columns = [c.strip() for c in df_proprio.columns]

DETECTORS = ["haar", "hog", "yunet"]
LABELS_DET = {"haar": "Haar", "hog": "HOG+SVM", "yunet": "YuNet"}
CONDITIONS = ["bright", "dark", "lateral", "overexposed"]
LABELS_COND = {
    "bright": "Boa luz",
    "dark": "Escuro",
    "lateral": "Lateral",
    "overexposed": "Superexp.",
}
COLORS = {
    "haar": ("#2196F3", "#90CAF9"),
    "hog": ("#4CAF50", "#A5D6A7"),
    "yunet": ("#F44336", "#EF9A9A"),
}


def plot_f1_bars(ax, df, title):
    n_det = len(DETECTORS)
    group_w = 0.8
    bar_w = group_w / (n_det * 2 + 1)
    x = np.arange(len(CONDITIONS))

    for di, det in enumerate(DETECTORS):
        for pi, pass_ in enumerate(["raw", "clahe"]):
            offset = (di * 2 + pi - (n_det - 1) - 0.5) * bar_w
            vals = []
            for cond in CONDITIONS:
                row = df[
                    (df["detector"] == det)
                    & (df["condition"] == cond)
                    & (df["pass"] == pass_)
                ]
                vals.append(row["f1"].values[0] if len(row) else 0)
            color = COLORS[det][pi]
            label = f"{LABELS_DET[det]} ({'Raw' if pass_ == 'raw' else 'CLAHE'})"
            ax.bar(
                x + offset,
                vals,
                width=bar_w,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS_COND[c] for c in CONDITIONS])
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title)


# --------------------------------------------------------------------------
# Fig 1 — F1-score LFW (+ próprio se disponível)
# --------------------------------------------------------------------------
if df_proprio is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
    plot_f1_bars(ax1, df_lfw, "LFW Subset (n=25/condição)")
    plot_f1_bars(ax2, df_proprio, "Dataset Próprio (n=4–6/condição)")
    ax2.set_ylabel("")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
else:
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    plot_f1_bars(ax1, df_lfw, "F1-score por Condição de Iluminação (LFW Subset)")
    ax1.legend(ncol=3, loc="lower right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT / "fig_f1_bar.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_f1_bar.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_f1_bar")

# --------------------------------------------------------------------------
# Fig 2 — Mean inference time per detector (LFW)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.5))

mean_times = {det: df_lfw[df_lfw["detector"] == det]["mean_ms"].mean() for det in DETECTORS}

bars = ax.bar(
    [LABELS_DET[d] for d in DETECTORS],
    [mean_times[d] for d in DETECTORS],
    color=["#2196F3", "#4CAF50", "#F44336"],
    width=0.5,
    edgecolor="white",
)
for bar, det in zip(bars, DETECTORS):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{mean_times[det]:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=7,
    )
ax.set_ylabel("Tempo médio (ms)")
ax.set_title("Tempo de Inferência por Detector")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(mean_times.values()) * 1.3)
fig.tight_layout()
fig.savefig(OUT / "fig_time_bar.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_time_bar.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_time_bar")
