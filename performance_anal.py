import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style config (shared) ─────────────────────────────────────────────────────
ALGO_LABELS = {"IBEA": "IBEA", "NSGAII": "NSGA-II", "MOEAD": "MOEA/D"}
ALGO_COLORS = {"IBEA": "#378ADD", "NSGAII": "#D85A30", "MOEAD": "#1D9E75"}
SETTING_STYLE = {"intertemporal": "-", "table": "--"}
SETTING_MARK = {"intertemporal": "o", "table": "s"}


# ── Helper ────────────────────────────────────────────────────────────────────
def count_solutions(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f) - 1  # subtract header row


# ── Dataset 1: vary depth (8, 10, 12), two objective-count categories ─────────
def load_dataset1(base="archive_data"):
    """
    Folder pattern: data_{depth}/{setting}_{ALGO}_single_{many|multi}_obj_non_param/
    Returns a DataFrame with columns: depth, algo, setting, obj, solutions
    """
    records = []
    pattern = re.compile(
        r"(intertemporal|table)_(IBEA|NSGAII|MOEAD)_single_(many|multi)_obj_non_param"
    )
    for entry in os.listdir(base):
        m_depth = re.match(r"data_(\d+)$", entry)
        if not m_depth:
            continue
        depth = int(m_depth.group(1))
        for folder in os.listdir(os.path.join(base, entry)):
            m = pattern.match(folder)
            if not m:
                continue
            setting, algo, obj = m.group(1), m.group(2), m.group(3)
            folder_path = os.path.join(base, entry, folder)
            for f in os.listdir(folder_path):
                if f.startswith("archives_") and f.endswith(".csv"):
                    records.append(dict(
                        depth=depth, algo=algo, setting=setting, obj=obj,
                        solutions=count_solutions(os.path.join(folder_path, f)),
                    ))
    return pd.DataFrame(records)


# ── Dataset 2: fixed depth=10, vary n_objectives (2, 5, 8) ───────────────────
def load_dataset2(base="archive_data2"):
    """
    Folder pattern: data_{depth}_{n_obj}/{setting}_{ALGO}_single_many_obj_non_param/
    Returns a DataFrame with columns: depth, n_obj, algo, setting, solutions
    """
    records = []
    pattern = re.compile(
        r"(intertemporal|table)_(IBEA|NSGAII|MOEAD)_single_many_obj_non_param"
    )
    for entry in os.listdir(base):
        m_dir = re.match(r"data_(\d+)_(\d+)$", entry)
        if not m_dir:
            continue
        depth, n_obj = int(m_dir.group(1)), int(m_dir.group(2))
        for folder in os.listdir(os.path.join(base, entry)):
            m = pattern.match(folder)
            if not m:
                continue
            setting, algo = m.group(1), m.group(2)
            folder_path = os.path.join(base, entry, folder)
            for f in os.listdir(folder_path):
                if f.startswith("archives_") and f.endswith(".csv"):
                    records.append(dict(
                        depth=depth, n_obj=n_obj, algo=algo, setting=setting,
                        solutions=count_solutions(os.path.join(folder_path, f)),
                    ))
    return pd.DataFrame(records)


# ── Load ──────────────────────────────────────────────────────────────────────
df1 = load_dataset1("./tree_data/vary_depths")
df2 = load_dataset2("./tree_data/vary_objectives")

DEPTHS = sorted(df1["depth"].unique())
N_OBJS = sorted(df2["n_obj"].unique())
ALGOS = ["IBEA", "NSGAII", "MOEAD"]


# ── Shared plotting helper ────────────────────────────────────────────────────
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
                label=f"{ALGO_LABELS[algo]} – {setting}",
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


# ── Figure 1: vary depth (many + multi) ──────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig1.suptitle("Number of solutions produced — vary depth",
              fontsize=13, fontweight="bold", y=1.01)

plot_lines(ax1, df1, "depth", DEPTHS, "many", "Depth", "Many objectives (8 obj)")
plot_lines(ax2, df1, "depth", DEPTHS, "multi", "Depth", "Multi-objectives (2 obj)")
add_legend(fig1, ax1)

fig1.tight_layout()
fig1.show()
fig1.savefig("./tree_figures/solutions_vary_depth.png", dpi=150, bbox_inches="tight")
# print("Saved solutions_vary_depth.png")

# ── Figure 2: vary objectives (depth 10) ─────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(6, 5))
fig2.suptitle("Number of solutions produced — vary objectives (depth 10)",
              fontsize=13, fontweight="bold", y=1.01)

plot_lines(ax3, df2, "n_obj", N_OBJS, None, "No. objectives", "")
add_legend(fig2, ax3)

fig2.tight_layout()
fig2.show()
fig2.savefig("./tree_figures/solutions_vary_objectives.png", dpi=150, bbox_inches="tight")
print("Saved solutions_vary_objectives.png")
