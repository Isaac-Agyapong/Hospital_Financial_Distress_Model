"""Shared chart style for this project: clean white canvas, bold sans headline with a grey takeaway line, a short
coloured tick before the title, direct labels, no chart junk.

Colour meanings (the app uses the same ones):
    RISK     coral red: hospitals that went on to lose money / high risk
    MODEL    indigo: the machine learning model
    RULE     sand grey: rules of thumb and everything else
    SAFE     teal: lower risk / made money
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RISK, RISK_L = "#E4572E", "#F4A58E"
MODEL, MODEL_L = "#3F37C9", "#A5A1EA"
RULE, RULE_L = "#B7AFA3", "#E4DFD8"
SAFE = "#12A594"
AMBER = "#F2A541"
INK, INK_2, GRID, PAPER = "#1B1B2F", "#5E6472", "#ECECF2", "#FFFFFF"
FONT = ["Segoe UI", "DejaVu Sans"]

IMAGE_DIR = Path(__file__).resolve().parents[1] / "Image"
IMAGE_DIR.mkdir(exist_ok=True)


def apply():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (9, 4.8),
        "font.family": FONT, "font.size": 10.5, "text.color": INK,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.9,
        "axes.axisbelow": True, "legend.frameon": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def title(ax, text, sub=None, colour=MODEL):
    ax.set_title(text, fontsize=15, fontweight="bold", loc="left", pad=32 if sub else 14, color=INK)
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                    color=INK_2, fontsize=10.5, va="bottom")
    ax.annotate("", (-0.012, 1), xycoords="axes fraction")  # keeps layout stable
    ax._tick_colour = colour


def pct(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def source(fig, text="Source: CMS Hospital Provider Cost Reports 2011-2023. Model tested on years it never saw."):
    fig.text(0.01, -0.02, text, color=INK_2, fontsize=8, ha="left", va="top")


def save(fig, name):
    fig.tight_layout()
    fig.canvas.draw()
    for ax in fig.axes:        # short coloured tick before each headline
        colour = getattr(ax, "_tick_colour", None)
        if colour:
            bb = ax.title.get_window_extent().transformed(fig.transFigure.inverted())
            fig.add_artist(plt.Rectangle((bb.x0 - 0.018, bb.y0 + 0.004), 0.008, bb.height * 0.8,
                                         transform=fig.transFigure, color=colour, lw=0))
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
