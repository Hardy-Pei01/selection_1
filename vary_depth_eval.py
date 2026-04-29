import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style config ──────────────────────────────────────────────────────────────
ALGO_LABELS = {"IBEA": "IBEA", "NSGAII": "NSGA-II", "MOEAD": "MOEA/D"}
ALGO_COLORS = {"IBEA": "#378ADD", "NSGAII": "#D85A30", "MOEAD": "#1D9E75"}
SETTING_STYLE = {"intertemporal": "-", "table": "--"}
SETTING_MARK = {"intertemporal": "o", "table": "s"}
SETTING_LABEL = {"intertemporal": "Static", "table": "Adaptive"}

ALGOS = ["IBEA", "NSGAII", "MOEAD"]

# Folder regex: depth is the FIRST integer in `data_{depth}_{x}`.
# Suffix after `_obj_` is left flexible (e.g. deterministic, observable, non_param).
DEPTH_RE = re.compile(r"data_(\d+)_\d+$")
LEAF_RE = re.compile(
    r"(intertemporal|table)_(IBEA|NSGAII|MOEAD)_single_(many|multi)_obj_\w+"
)


# ── Helper ────────────────────────────────────────────────────────────────────
def count_solutions(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f) - 1  # subtract header


# ── Loader: vary depth, two objective-count categories ────────────────────────
def load_dataset(base="./tree_data/moea_depth_vary"):
    """
    Folder pattern: data_{depth}_{x}/{setting}_{ALGO}_single_{many|multi}_obj_{suffix}/
    Returns DataFrame: depth, algo, setting, obj, solutions
    """
    records = []
    for entry in sorted(os.listdir(base)):
        m_depth = DEPTH_RE.match(entry)
        if not m_depth:
            continue
        depth = int(m_depth.group(1))
        for folder in sorted(os.listdir(os.path.join(base, entry))):
            m = LEAF_RE.match(folder)
            if not m:
                continue
            setting, algo, obj = m.group(1), m.group(2), m.group(3)
            folder_path = os.path.join(base, entry, folder)
            # Pick the "archives_..._0.csv" — sorts before "..._combined.csv"
            csv_files = [f for f in os.listdir(folder_path)
                         if f.startswith("archives_") and f.endswith(".csv")]
            if not csv_files:
                continue
            filepath = os.path.join(folder_path, sorted(csv_files)[0])
            records.append(dict(
                depth=depth, algo=algo, setting=setting, obj=obj,
                solutions=count_solutions(filepath),
            ))
    return pd.DataFrame(records)


# ── Load ──────────────────────────────────────────────────────────────────────
df = load_dataset()
DEPTHS = sorted(df["depth"].unique())


# ── Plotting helper ───────────────────────────────────────────────────────────
def plot_lines(ax, df, x_col, x_vals, obj_filter, x_label, title):
    subset = df[df["obj"] == obj_filter] if obj_filter else df
    for algo in ALGOS:
        for setting in ["intertemporal", "table"]:
            data = (
                subset[(subset["algo"] == algo) & (subset["setting"] == setting)]
                    .set_index(x_col)["solutions"]
                    .reindex(x_vals)
            )
            ax.plot(
                x_vals, data.values,
                color=ALGO_COLORS[algo],
                linestyle=SETTING_STYLE[setting],
                marker=SETTING_MARK[setting],
                linewidth=1.8, markersize=6,
                label=f"{ALGO_LABELS[algo]} – {SETTING_LABEL[setting]}",
            )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Number of solutions", fontsize=10)
    ax.set_xticks(x_vals)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def add_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=10)


# ── Figure: vary depth ────────────────────────────────────────────────────────
os.makedirs("./tree_figures", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Number of solutions produced — vary depth",
             fontsize=13, fontweight="bold", y=1.01)

plot_lines(ax1, df, "depth", DEPTHS, "many", "Depth", "8-objective")
plot_lines(ax2, df, "depth", DEPTHS, "multi", "Depth", "2-objective")
add_legend(fig, ax1)

fig.tight_layout()
fig.savefig("./tree_figures/vary_depth.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved solutions_vary_depth.png")
