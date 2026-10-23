"""Training history across a recipe's sequential episodes (learning rate, adaptive dropout,
input noise and the weighted loss terms), written to one CSV for the training-curve figure
(scripts/plot_training_history.py).

main.py makes one `TrainingHistory` per run when any block of the recipe sets
`history_every_n_batches`, and hands it to every train episode. The trainer calls
`add_batch` after each optimizer step and `add_val` at each validation pass (the
pre-training baseline and every fit check). Every N batches one 'train' row is emitted
holding the batch-size-weighted MEAN of each loss term over those N batches, plus the
learning rate, dropout rate and noise at the window's last batch. The last window of an
episode is flushed even when shorter than N.

x axis. `epoch` is continuous across episodes: an episode starts where the previous
recorded episode actually stopped (early stopping included, so a stop at a mid-epoch fit
check leaves a fractional offset). Inside an episode it is
`epoch_idx + rows_seen_this_epoch / n_rows`, counted in rows rather than batches because
ChunkedMemmapTrainLoader's len() is only approximate. `episode_epoch` is the same thing
without the offset.

Loss scale. Upper-stage terms are the WEIGHTED terms of trainer._upper_loss at the scale
the optimizer sees (`trainer.LOSS_SCALE` included), so
    loss_sat + loss_chem + loss_mole + loss_bulk + loss_hinge == loss_total,
and 'val' rows (the validation total from _evaluate_upper_model, same objective in eval
mode) are directly comparable to loss_total. Val rows carry only loss_total. Lower-stage
(`strategy: lower`) episodes have just the phase-weighted binary loss, unscaled: it goes in
both loss_total and loss_sat.

Tune episodes are not recorded (trials are throwaway branches with no place on a single
epoch axis), and neither are derivative (Sobolev) episodes; neither advances the axis.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOSS_TERMS = ('sat', 'chem', 'mole', 'bulk', 'hinge')
COLUMNS = ('episode', 'kind', 'global_batch', 'epoch', 'episode_epoch', 'lr', 'dropout',
           'noise', 'loss_total', *(f'loss_{t}' for t in LOSS_TERMS))
CONFIG_KEY = 'history_every_n_batches'


def every_n_from(value: Any) -> Optional[int]:
    """Recipe value -> positive batch count, or None (= do not record)."""
    if value is None or str(value).strip().lower() in ('', 'null', 'none', 'false', '0'):
        return None
    n = int(float(value))
    if n < 1:
        raise ValueError(f"{CONFIG_KEY} must be a positive integer or null, got {value!r}")
    return n


def history_wanted(config: Dict[str, Any]) -> bool:
    """True when the global block or any episode block sets a batch count."""
    if every_n_from(config.get(CONFIG_KEY)) is not None:
        return True
    return any(isinstance(v, dict) and every_n_from(v.get(CONFIG_KEY)) is not None
               for v in config.values())


def upper_terms(total, sat, chem, mole, bulk, sat_alpha, chem_alpha, mole_alpha, bulk_alpha,
                scale, hinge_on) -> Dict[str, float]:
    """Weighted, scaled loss terms of one upper-stage batch as floats (inputs may be
    0-d tensors). The hinge term is not returned by _upper_loss, so it is recovered as
    the remainder of the total; it is reported as exactly 0 when the hinge is off rather
    than as float round-off."""
    f = lambda v: float(v.item() if hasattr(v, 'item') else v)
    total = f(total)
    terms = dict(sat=scale * sat_alpha * f(sat),
                 chem=scale * chem_alpha * f(chem),
                 mole=scale * mole_alpha * f(mole),
                 # _upper_loss leaves bulk out entirely when its weight is 0 (NaN guard).
                 bulk=scale * bulk_alpha * f(bulk) if bulk_alpha != 0 else 0.0)
    terms['hinge'] = (total - sum(terms.values())) if hinge_on else 0.0
    terms['total'] = total
    return terms


class TrainingHistory:
    def __init__(self, path):
        self.path = Path(path)
        self.rows: List[Dict[str, Any]] = []
        self.episodes: List[Tuple[str, float, float]] = []  # (name, start epoch, end epoch)
        self.epoch_offset = 0.0
        self.global_batch = 0
        self.episode: Optional[str] = None

    # ----------------------------------------------------------------- episode bounds
    @property
    def active(self) -> bool:
        return self.episode is not None

    def start_episode(self, name: str, n_rows: int, every_n_batches: Any = None) -> None:
        if self.active:
            raise RuntimeError(f"start_episode({name!r}) while {self.episode!r} is still open")
        self.episode = str(name)
        self.n_rows = max(int(n_rows), 1)
        self.every = every_n_from(every_n_batches)
        self.pos = 0.0          # epochs into this episode
        self._state = (float('nan'),) * 3
        self._reset_window()
        what = f"every {self.every} batches" if self.every else "off for this episode"
        print(f"[history] {name}: recording {what}; starts at epoch {self.epoch_offset:.4f}")

    def end_episode(self) -> None:
        if not self.active:
            return
        self._flush()
        start = self.epoch_offset
        self.epoch_offset += self.pos
        self.episodes.append((self.episode, start, self.epoch_offset))
        print(f"[history] {self.episode}: epochs {start:.4f} -> {self.epoch_offset:.4f}")
        self.episode = None

    # ----------------------------------------------------------------- recording
    def add_batch(self, epoch_idx: int, rows_in_epoch: int, batch_rows: int, lr: float,
                  dropout: float, noise: float, terms: Dict[str, float]) -> None:
        """After an optimizer step. `rows_in_epoch` includes this batch; `terms` holds
        'total' and any of LOSS_TERMS (missing ones count as 0)."""
        self.global_batch += 1
        self.pos = epoch_idx + rows_in_epoch / self.n_rows
        if self.every is None:
            return
        w = float(batch_rows)
        self._w_rows += w
        self._w_batches += 1
        for key in self._w_sums:
            self._w_sums[key] += w * float(terms.get(key, 0.0))
        self._state = (float(lr), float(dropout), float(noise))
        if self._w_batches >= self.every:
            self._flush()

    def epoch_done(self, epoch_idx: int) -> None:
        """The batch loop of `epoch_idx` ran to completion: snap to the integer epoch."""
        self.pos = float(epoch_idx + 1)

    def add_val(self, lr: float, dropout: float, noise: float, val_loss: float) -> None:
        if self.every is None:
            return
        self._append('val', lr, dropout, noise, float(val_loss), {})

    # ----------------------------------------------------------------- output
    def write_csv(self, path=None) -> Path:
        """Rewrite the whole file (called after every episode, so a crash late in a run
        keeps the earlier episodes). Written to a temp file first, then swapped in."""
        path = Path(path) if path is not None else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, path)
        return path

    # ----------------------------------------------------------------- internals
    def _reset_window(self) -> None:
        self._w_rows = 0.0
        self._w_batches = 0
        self._w_sums = dict.fromkeys(('total',) + LOSS_TERMS, 0.0)

    def _flush(self) -> None:
        if not self._w_batches:
            return
        means = {k: v / self._w_rows for k, v in self._w_sums.items()}
        self._append('train', *self._state, means.pop('total'), means)
        self._reset_window()

    def _append(self, kind, lr, dropout, noise, total, terms) -> None:
        row = dict(episode=self.episode, kind=kind, global_batch=self.global_batch,
                   epoch=self.epoch_offset + self.pos, episode_epoch=self.pos,
                   lr=float(lr), dropout=float(dropout), noise=float(noise),
                   loss_total=total)
        row.update({f'loss_{t}': terms[t] for t in LOSS_TERMS if t in terms})
        self.rows.append(row)
