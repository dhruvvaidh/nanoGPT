"""
Regenerates the figures in docs/images/ from the real code and the real training log.

    python docs/make_plots.py              # run from the repo root

- docs/images/lr_schedule.png      : get_lr() from train.py, evaluated at every step
- docs/images/training_curves.png  : train/val loss + HellaSwag from log/log.txt
                                     (the log-plotting cell of notebooks/walkthrough.ipynb)

Only needs numpy + matplotlib. get_lr and its constants are pulled out of train.py's source
instead of importing it, so torch/tiktoken/transformers don't have to be installed.
The training-curves figure is skipped if log/log.txt isn't there (it's gitignored).
"""
import ast
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")
LOG_FILE = os.path.join(ROOT, "log", "log.txt")

# chart chrome (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.labelcolor": INK_2,
    "font.size": 10.5, "lines.linewidth": 2,
})


def load_get_lr():
    """exec train.py's hyperparameter assignments and get_lr, without importing train.py (which needs torch)"""
    with open(os.path.join(ROOT, "train.py")) as f:
        tree = ast.parse(f.read())
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_lr"
            or isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets)]
    ns = {"math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "train.py", "exec"), ns)
    return ns


def plot_lr_schedule():
    ns = load_get_lr()
    get_lr, max_steps, warmup_steps = ns["get_lr"], ns["max_steps"], ns["warmup_steps"]
    steps = np.arange(max_steps)
    lrs = np.array([get_lr(s) for s in steps])

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(steps, lrs, color=SERIES_1)
    ax.axvline(warmup_steps, color=AXIS, linestyle="--", linewidth=1)
    ax.annotate(f"warmup ends\nstep {warmup_steps}", (warmup_steps, lrs.max()),
                xytext=(10, -4), textcoords="offset points", va="top", color=INK_2, fontsize=9.5)
    ax.annotate(f"max_lr {ns['max_lr']:.0e}", (int(np.argmax(lrs)), lrs.max()),
                xytext=(0, 6), textcoords="offset points", ha="left", color=INK_2, fontsize=9.5)
    ax.annotate(f"min_lr {ns['min_lr']:.0e}", (max_steps - 1, lrs[-1]),
                xytext=(0, 8), textcoords="offset points", ha="right", color=INK_2, fontsize=9.5)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, lrs.max() * 1.15)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule: linear warmup, then cosine decay")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "lr_schedule.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def plot_training_curves():
    if not os.path.exists(LOG_FILE):
        print(f"skipping training curves: {LOG_FILE} not found")
        return

    # from the last cell of notebooks/walkthrough.ipynb
    sz = "124M"
    loss_baseline = {"124M": 3.2924}[sz]
    hella2_baseline = {"124M": 0.294463, "350M": 0.375224, "774M": 0.431986, "1558M": 0.488946}[sz]  # GPT-2
    hella3_baseline = {"124M": 0.337, "350M": 0.436, "774M": 0.510, "1558M": 0.547}[sz]  # GPT-3

    # parse the individual lines, group by stream (train,val,hella)
    with open(LOG_FILE) as f:
        lines = f.readlines()
    streams = {}
    for line in lines:
        step, stream, val = line.strip().split()
        streams.setdefault(stream, {})[int(step)] = float(val)
    # convert each stream from {step: val} to (steps[], vals[])
    streams_xy = {k: list(zip(*sorted(v.items()))) for k, v in streams.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: losses: both train and val
    xs, ys = streams_xy["train"]
    ax1.plot(xs, ys, color=SERIES_1, linewidth=1, alpha=0.85, label=f"train loss (min {min(ys):.3f})")
    xs, ys = streams_xy["val"]
    ax1.plot(xs, ys, color=SERIES_2, label=f"val loss (min {min(ys):.3f})")
    ax1.axhline(loss_baseline, color=MUTED, linestyle="--", linewidth=1.2,
                label=f"OpenAI GPT-2 ({sz}) val loss {loss_baseline}")
    ax1.set_yscale("log")
    ax1.set_ylim(top=4.0)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    print("Min Train Loss:", min(streams_xy["train"][1]))
    print("Min Validation Loss:", min(streams_xy["val"][1]))

    # Panel 2: HellaSwag eval
    if "hella" in streams_xy:
        xs, ys = streams_xy["hella"]
        ax2.plot(xs, ys, color=SERIES_1, label=f"this model (max {max(ys):.4f})")
        ax2.axhline(hella2_baseline, color=MUTED, linestyle="--", linewidth=1.2,
                    label=f"OpenAI GPT-2 ({sz}) {hella2_baseline:.4f}")
        ax2.axhline(hella3_baseline, color=MUTED, linestyle=":", linewidth=1.5,
                    label=f"OpenAI GPT-3 ({sz}) {hella3_baseline:.3f}")
        ax2.set_xlabel("step")
        ax2.set_ylabel("acc_norm")
        ax2.set_title("HellaSwag accuracy")
        ax2.legend()
        print("Max Hellaswag eval:", max(ys))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_lr_schedule()
    plot_training_curves()
