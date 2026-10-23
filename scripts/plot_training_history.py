"""Three-panel training-curve figure from a training-history CSV.

    python scripts/plot_training_history.py src/builder/training/logs/<tarname>_history.csv
    python scripts/plot_training_history.py <csv> --components --smooth 5 --out fig.pdf

Rows (shared x = epochs since the start of the first recorded episode):
    1. learning rate (log scale)
    2. loss (log scale): training loss (window mean every N batches) and validation
       loss at the pre-training baseline and every fit check; --components adds the
       weighted terms (saturation, chemistry, moles, bulk, hinge) that sum to the total
    3. adaptive dropout rate (a step function: it only changes at fit checks)
Dashed vertical lines mark episode transitions; each episode is named above the top panel.

The CSV is written by builder.training.history.TrainingHistory (recipe key
`history_every_n_batches`). Each episode warm-starts from the previous episode's BEST
checkpoint, not its last weights, so curves are drawn per episode and not joined across
transitions -- a drop in loss at a boundary is that reload, not a training step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order (validated default palette, light mode): train, val, then terms.
INK = '#2b2b2a'
MUTED = '#8a8984'
TRAIN = '#2a78d6'      # blue
VAL = '#eb6834'        # orange
TERM_COLORS = {        # components, in fixed order after train/val
    'sat': '#1baf7a',   # aqua
    'chem': '#6250d6',  # violet
    'mole': '#e87ba4',  # magenta
    'bulk': '#008300',  # green
    'hinge': '#e34948', # red
}
TERM_LABELS = {'sat': 'saturation', 'chem': 'chemistry', 'mole': 'moles',
               'bulk': 'bulk', 'hinge': 'absent hinge'}


def episode_spans(df):
    """(name, start, end) per episode in run order. start = epoch - episode_epoch, which
    is exactly the offset the recorder used, so it is right even for an episode whose
    first row is not at its start."""
    spans = []
    for name, g in df.groupby('episode', sort=False):
        start = float((g['epoch'] - g['episode_epoch']).iloc[0])
        spans.append((name, start, float(g['epoch'].max())))
    return spans


def _per_episode(ax, df, y, smooth=1, **kw):
    """Plot one series episode by episode, so lines never join across a transition.
    `label` is applied to the first episode only (one legend entry)."""
    label = kw.pop('label', None)
    for _, g in df.groupby('episode', sort=False):
        g = g.dropna(subset=[y])
        if g.empty:
            continue
        vals = g[y].rolling(smooth, min_periods=1).mean() if smooth > 1 else g[y]
        ax.plot(g['epoch'], vals, label=label, **kw)
        label = None


def plot_history(csv_path, out=None, components=False, smooth=1, show_val=True,
                 figsize=(6.5, 6.5), dpi=300):
    df = pd.read_csv(csv_path)
    train = df[df['kind'] == 'train']
    val = df[df['kind'] == 'val']
    spans = episode_spans(df)

    plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8})
    fig, axes = plt.subplot_mosaic([['lr'], ['loss'], ['dropout']], sharex=True,
                                   figsize=figsize, height_ratios=[1, 1.6, 1],
                                   layout='constrained')
    ax_lr, ax_loss, ax_do = axes['lr'], axes['loss'], axes['dropout']

    # LR and dropout come from every row (val rows carry them too, including each
    # episode's baseline at its start). A row reports the state over the stretch since the
    # previous row -- a train row its window, a val row the run-up to that check, before
    # the check adapts dropout -- so they are drawn as 'steps-pre'.
    state = df.sort_values('epoch', kind='stable')

    # --- learning rate: train rows plus each episode's baseline (its starting LR). Other
    # val rows are left out: they read the LR after the scheduler's last step, which would
    # add a small stair at every check.
    lr_rows = state[(state['kind'] == 'train') | (state['episode_epoch'] == 0)]
    _per_episode(ax_lr, lr_rows, 'lr', color=TRAIN, lw=1.5)
    ax_lr.set_yscale('log')
    ax_lr.set_ylabel('Learning rate')

    # --- loss
    if components:
        for term, color in TERM_COLORS.items():
            col = f'loss_{term}'
            if col in train and (train[col].fillna(0) != 0).any():
                _per_episode(ax_loss, train, col, smooth, color=color, lw=1.0,
                             label=TERM_LABELS[term])
    _per_episode(ax_loss, train, 'loss_total', smooth, color=TRAIN, lw=1.5,
                 label='train (total)' if components else 'train')
    if show_val and not val.empty:
        ax_loss.plot(val['epoch'], val['loss_total'], ls='none', marker='o', ms=4,
                     mfc=VAL, mec='white', mew=0.8, label='validation', zorder=3)
    ax_loss.set_yscale('log')
    ax_loss.set_ylabel('Loss (weighted)')
    ax_loss.legend(loc='best', ncols=2 if components else 1, frameon=True, framealpha=0.9,
                   edgecolor='none')

    # --- dropout (steps: it changes only at fit checks)
    _per_episode(ax_do, state, 'dropout', color=TRAIN, lw=1.5, drawstyle='steps-pre')
    ax_do.set_ylabel('Dropout rate')
    ax_do.set_ylim(bottom=0)
    ax_do.set_xlabel('Epoch')

    # --- episode transitions and names
    for ax in (ax_lr, ax_loss, ax_do):
        ax.grid(True, axis='y', color='#e4e3df', lw=0.6)
        ax.set_axisbelow(True)
        for _, start, _ in spans[1:]:
            ax.axvline(start, color=MUTED, lw=0.8, ls='--', zorder=0)
    x0, x1 = spans[0][1], max(end for *_, end in spans)
    ax_do.set_xlim(x0, x1 + 0.01 * (x1 - x0))
    for name, start, end in spans:
        ax_lr.annotate(name, xy=((start + end) / 2, 1.0), xycoords=('data', 'axes fraction'),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                       fontsize=8, color=INK, annotation_clip=False)

    if out:
        fig.savefig(out, dpi=dpi)
        print(f"saved {out}")
    return fig, axes


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('csv', type=Path)
    p.add_argument('--out', type=Path, default=None,
                   help='output file (.pdf/.png/.svg); default: <csv stem>.pdf beside the CSV')
    p.add_argument('--components', action='store_true',
                   help='also plot the weighted loss terms that sum to the total')
    p.add_argument('--smooth', type=int, default=1,
                   help='rolling mean over this many recorded points (training curves only)')
    p.add_argument('--no-val', action='store_true', help='omit validation points')
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    out = a.out or a.csv.with_suffix('.pdf')
    plot_history(a.csv, out, components=a.components, smooth=a.smooth, show_val=not a.no_val)
    if a.show:
        plt.show()


if __name__ == '__main__':
    main()
