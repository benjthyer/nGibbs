"""
Search the absent-side hinge margin `m` (trainer._absent_hinge_loss) for the best
abundance-weighted presence F1.

Every trial fine-tunes the SAME warm-start checkpoint for one episode of the recipe,
with only `absent_hinge_margin` changed, then scores the episode's saved best
checkpoint on a held-out bundle. The hinge weight, epochs, schedule and everything else
come from the recipe, so the trials differ in m alone.

Search:
  1. bracket  -- train `--m-mid` and `--m-far`.
  2. expand   -- while the largest m tried is the best by more than `--min-delta`,
                 double it (at most `--max-doublings` times).
  3. refine   -- `--refine-rounds` times, train the midpoints between the best m and its
                 neighbours (0 is the left bound and is not itself trained).
A control run with the hinge off (`absent_hinge_weight: 0`, same episode) and the
untouched warm-start model are scored too, so a gain can be told apart from "it simply
trained for longer".

Objective: pooled F1 over phases, phase p weighted by
    w_p = 0.3 + 0.7 * (a_p - min a) / (max a - min a),
a_p = phase p's mean modal wt% over the eval bundle (zeros included). Weighted counts
are pooled before taking F1: F1 = 2*sum(w*TP) / (2*sum(w*TP) + sum(w*FP) + sum(w*FN)).
Presence is the model's own call, `likelihoods > 0.5` (g_phi > 0 for ContinuousModel).

Results go to <out-dir>/results.json after every trial; rerunning with the same
--out-dir and settings reuses finished trials, so an interrupted search resumes.

    python scripts/tune_absent_hinge.py \
        --config recipes/training/HeFESToQuickRampAssymetryFix.yaml \
        --warm-start HeFESTo_Mars_NPT_train5 --episode train6 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

from ngibbs.engine.NN import rebuild_MELTS_model                        # noqa: E402
from ngibbs.utils.file_utils import load_ml_bundle                      # noqa: E402
from builder.training.main import (_resolve_train_roots, _resolve_bundle_path,  # noqa: E402
                                   _deep_update)
from builder.training.loadTrainData import load_ML_data                 # noqa: E402
from builder.training.trainer import abundance_f1_weights, weighted_pooled_f1  # noqa: E402

TRAIN_DIR = REPO / 'src' / 'builder' / 'training'
MAIN_PY = TRAIN_DIR / 'main.py'
EPISODE_RE = re.compile(r'^(train|tune)\d+$')
# Lines of a trial's main.py output echoed live with --echo key (the log keeps all of it).
ECHO_RE = re.compile(r'^(Loading|Baseline Test Loss|Epoch \d+:|New best|No improvement in episode|'
                     r'Saved trained|\[presence g_phi > 0\]|\[boundary_temperature\]|'
                     r'Traceback|\w*Error\b)')


def say(msg=''):
    print(f"[search {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _name(m):
    return 'baseline' if m == -1 else 'no hinge' if m is None else f'm={m:g}'


# --------------------------------------------------------------------------- recipe
def _load_recipe(path):
    with open(TRAIN_DIR / 'config' / 'defaults.yaml', encoding='utf-8') as fh:
        defaults = yaml.safe_load(fh) or {}
    with open(path, encoding='utf-8') as fh:
        recipe = yaml.safe_load(fh) or {}
    return recipe, _deep_update(deepcopy(defaults), recipe)


def _pick_episode(recipe, name):
    trains = [k for k in recipe if re.match(r'^train\d+$', k)]
    if name is None:
        if not trains:
            raise ValueError("Recipe has no train<N> episode.")
        name = trains[-1]
    if name not in recipe or not re.match(r'^train\d+$', name):
        raise ValueError(f"--episode {name!r} is not a train<N> block of the recipe "
                         f"(found: {trains}).")
    return name


def _hinge_weight(recipe, episode):
    w = recipe[episode].get('absent_hinge_weight', recipe.get('absent_hinge_weight', 0.0))
    w = float(w or 0.0)
    if w <= 0:
        raise ValueError("Set `absent_hinge_weight: <w>` (> 0) in the recipe, top level or in "
                         f"the {episode} block; this script searches only the margin m.")
    return w


def _tag(m):
    return 'nohinge' if m is None else 'm' + f'{m:.4f}'.rstrip('0').rstrip('.').replace('.', 'p')


def _trial_recipe(recipe, episode, warm_start, epochs, ckpt_name, m):
    """The recipe with every other episode removed and only the hinge margin changed
    (m=None: hinge off, the control)."""
    trial = {k: deepcopy(v) for k, v in recipe.items() if not EPISODE_RE.match(k)}
    block = deepcopy(recipe[episode])
    trial['warm_start'] = warm_start
    if epochs is not None:
        block['epochs'] = int(epochs)
    block['DictFilePath'] = ckpt_name
    if m is None:
        block['absent_hinge_weight'] = 0.0
    else:
        block['absent_hinge_margin'] = float(m)
    trial[episode] = block
    return trial


# --------------------------------------------------------------------------- scoring
class Evaluator:
    """Holds the eval bundle once; scores any checkpoint on it."""

    def __init__(self, bundle_stem, warm_ckpt, device, batch_size=65536):
        normalizer = rebuild_MELTS_model(str(warm_ckpt)).ml_indexer.feature_normalizer
        data, ml_indexer = load_ML_data(bundle_stem, feature_normalizer=normalizer,
                                        with_derivatives=False)
        self.x, self.b = data.features, data.binarylabels > 0.5
        self.names = sorted(ml_indexer.mass_phasedict, key=ml_indexer.mass_phasedict.get)
        mass = np.asarray(load_ml_bundle(Path(bundle_stem + '.tar.gz')).mass_labels,
                          dtype=np.float64)[:, :self.b.shape[1]]
        # Column-order check. load_ML_data drops out-of-bounds rows, so the frequencies
        # agree closely rather than exactly; a misordered column would be far off.
        if not np.allclose((mass > 0).mean(0), self.b.numpy().mean(0), atol=0.02):
            raise RuntimeError("mass_labels and binary_labels disagree on phase presence; "
                               "column order is not what this script assumes.")
        self.abundance = mass.mean(0)
        self.w = abundance_f1_weights(self.abundance)
        self.device, self.batch_size = device, batch_size

    def score(self, ckpt):
        model = rebuild_MELTS_model(str(ckpt)).to(self.device).eval()
        P = self.b.shape[1]
        tp, fp, fn = (np.zeros(P) for _ in range(3))
        with torch.no_grad():
            for s in range(0, len(self.x), self.batch_size):
                pred = (model(self.x[s:s + self.batch_size].to(self.device))[0] > 0.5).cpu()
                truth = self.b[s:s + self.batch_size]
                tp += (pred & truth).sum(0).numpy()
                fp += (pred & ~truth).sum(0).numpy()
                fn += (~pred & truth).sum(0).numpy()
        del model
        torch.cuda.empty_cache()
        f1 = lambda t, p, n: 2 * t / max(2 * t + p + n, 1e-12)
        w = self.w
        return {
            'objective': float(weighted_pooled_f1(tp, fp, fn, w)),
            'precision': float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
            'recall': float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
            'per_phase': {n: {'f1': float(f1(tp[i], fp[i], fn[i])),
                              'precision': float(tp[i] / max(tp[i] + fp[i], 1)),
                              'recall': float(tp[i] / max(tp[i] + fn[i], 1))}
                          for i, n in enumerate(self.names)},
        }


# --------------------------------------------------------------------------- driver
class Search:
    def __init__(self, args):
        self.args = args
        self.recipe, merged = _load_recipe(args.config)
        self.episode = _pick_episode(self.recipe, args.episode)
        self.weight = _hinge_weight(self.recipe, self.episode)
        self.warm = args.warm_start or self.recipe.get('warm_start')
        if not self.warm or str(self.warm).lower() == 'none':
            raise ValueError("Give --warm-start (or warm_start in the recipe): every trial "
                             "must start from the same trained model.")
        ck = merged['checkpoints']
        self.warm_ckpt = TRAIN_DIR / ck['load_dir'] / f'{self.warm}.tar'
        self.save_dir = TRAIN_DIR / ck['save_dir']
        if not self.warm_ckpt.exists():
            raise FileNotFoundError(self.warm_ckpt)

        self.out = Path(args.out_dir or REPO / 'hinge_search' /
                        f"{Path(args.config).stem}_{self.episode}_{self.warm}")
        (self.out / 'recipes').mkdir(parents=True, exist_ok=True)
        (self.out / 'logs').mkdir(exist_ok=True)
        self.settings = {'config': str(Path(args.config).resolve()), 'episode': self.episode,
                         'warm_start': self.warm, 'epochs': args.epochs,
                         'absent_hinge_weight': self.weight}
        self.results_path = self.out / 'results.json'
        self.results = self._load_results()

        tarname = merged['tarname']
        roots = _resolve_train_roots(tarname)
        stem = None
        for cand in ([args.eval_bundle] if args.eval_bundle else
                     [f'{tarname}_Valid', f'{tarname}_Test']):
            s = _resolve_bundle_path(cand, roots)
            if Path(s + '.tar.gz').exists():
                stem = s
                break
        if stem is None:
            raise FileNotFoundError(f"No eval bundle found for {tarname} (tried _Valid, _Test).")
        self.settings['eval_bundle'] = stem
        self.t_start = time.time()
        self.trials_run = 0

        say("=" * 64)
        say(f"recipe       {args.config}  (episode {self.episode})")
        say(f"warm start   {self.warm_ckpt}")
        say(f"per trial    {args.epochs} epochs, absent_hinge_weight={self.weight:g}, only m varies")
        say(f"search       bracket m={args.m_mid:g}, {args.m_far:g}; up to {args.max_doublings} "
            f"doublings; {args.refine_rounds} refine rounds; min-delta {args.min_delta:g}")
        say(f"extra runs   {'no-hinge control' if args.control else 'no control'}"
            f"{', m=0' if args.include_zero else ''}")
        say(f"output       {self.out}")
        if stem.endswith('_Test'):
            say("WARNING: scoring on the _Test bundle, which also picks each trial's best "
                "checkpoint. Pass --eval-bundle to use held-out data.")
        say(f"loading eval bundle {stem} (once) ...")
        t0 = time.time()
        self.ev = Evaluator(stem, self.warm_ckpt, args.device)
        say(f"eval bundle loaded: {len(self.ev.x):,} rows x {len(self.ev.names)} phases "
            f"({time.time() - t0:.0f} s)")
        say("phase weights (mean wt% -> w):")
        for n, a, w in sorted(zip(self.ev.names, self.ev.abundance, self.ev.w), key=lambda r: -r[2]):
            print(f"    {n:22s} {a:7.3f}  {w:.3f}")

    def _load_results(self):
        if not self.results_path.exists():
            return {}
        saved = json.loads(self.results_path.read_text())
        if saved.get('settings') != self.settings:
            raise ValueError(f"{self.results_path} was made with different settings:\n"
                             f"  saved: {saved.get('settings')}\n  now:   {self.settings}\n"
                             "Use a new --out-dir.")
        say(f"resuming from {self.results_path}: "
            f"{', '.join(_name(r['m']) for r in saved['trials'].values())} already done")
        return saved['trials']

    def _save(self):
        self.results_path.write_text(json.dumps(
            {'settings': self.settings, 'trials': self.results}, indent=1))

    def run(self, m):
        tag = _tag(m)
        if tag in self.results:
            say(f"{_name(m)}: already done (F1w={self.results[tag]['objective']:.5f}), skipping")
            return self.results[tag]['objective']
        ckpt_name = f"hingesearch_{self.warm}_{tag}.tar"
        ckpt = self.save_dir / ckpt_name
        if ckpt.exists():
            ckpt.unlink()       # left over from an interrupted trial of this same search
        recipe_path = self.out / 'recipes' / f'{tag}.yaml'
        recipe_path.write_text(yaml.safe_dump(
            _trial_recipe(self.recipe, self.episode, self.warm, self.args.epochs, ckpt_name, m),
            sort_keys=False))
        log_path = self.out / 'logs' / f'{tag}.txt'
        self.trials_run += 1
        say("-" * 64)
        say(f"trial {self.trials_run}: {_name(m)}  -- training {self.args.epochs} epochs")
        say(f"  log: {log_path}")
        t0 = time.time()
        tail = []
        with open(log_path, 'w') as log:
            proc = subprocess.Popen([self.args.python, '-u', str(MAIN_PY), 'train',
                                     '--config', str(recipe_path), '--warm_start', self.warm],
                                    cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                tail = (tail + [line.rstrip()])[-25:]
                if self.args.echo == 'all' or (self.args.echo == 'key' and ECHO_RE.match(line)):
                    print(f"    {_name(m):>8s} | {line.rstrip()}", flush=True)
            proc.wait()
        if proc.returncode != 0:
            say(f"trial {_name(m)} FAILED (exit {proc.returncode}); last lines of the log:")
            for line in tail:
                print(f"    {line}")
            raise RuntimeError(f"Trial {tag} failed (exit {proc.returncode}); see {log_path}")
        # main.py only saves when the episode beats its own starting loss.
        improved = ckpt.exists()
        say(f"  trained in {(time.time() - t0) / 60:.1f} min; scoring "
            f"{'its best checkpoint' if improved else 'the warm start (no epoch beat its starting loss)'} ...")
        res = self.ev.score(ckpt if improved else self.warm_ckpt)
        res.update(m=m, checkpoint=str(ckpt) if improved else None,
                   minutes=(time.time() - t0) / 60)
        self.results[tag] = res
        self._save()
        base = self.results['baseline']['objective']
        say(f"  {_name(m)}: F1w={res['objective']:.5f} ({res['objective'] - base:+.5f} vs baseline)  "
            f"P={res['precision']:.4f}  R={res['recall']:.4f}  ({res['minutes']:.1f} min)")
        self.phase_table(res)
        self.standings()
        return res['objective']

    def phase_table(self, res):
        """Phasewise P/R/F1 of one trial, heaviest-weighted phase first, with the change in
        F1 against the baseline (the warm start)."""
        base = self.results['baseline']['per_phase']
        print(f"      {'phase':22s} {'w':>5s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'dF1':>8s}"
              f"   | abundance-weighted F1 {res['objective']:.5f}", flush=True)
        for i in np.argsort(-self.ev.w):
            n = self.ev.names[i]
            pp = res['per_phase'][n]
            print(f"      {n:22s} {self.ev.w[i]:5.2f} {pp['precision']:7.3f} {pp['recall']:7.3f} "
                  f"{pp['f1']:7.3f} {pp['f1'] - base[n]['f1']:+8.4f}", flush=True)

    def standings(self):
        """Compact scoreboard after every trial."""
        best = self._best() if self._hinge_trials() else None
        rows = sorted(self.results.values(), key=lambda r: (r['m'] is not None, r['m'] or 0))
        done = [r['minutes'] for r in self.results.values() if 'minutes' in r]
        say(f"  standings ({len(done)} trial(s), {(time.time() - self.t_start) / 3600:.2f} h "
            f"elapsed{f', ~{np.mean(done):.0f} min/trial' if done else ''}):")
        for r in rows:
            star = '  <- best' if r['m'] is not None and r['m'] == best else ''
            print(f"      {_name(r['m']):>10s}  F1w {r['objective']:.5f}  "
                  f"P {r['precision']:.4f}  R {r['recall']:.4f}{star}", flush=True)

    def _hinge_trials(self):
        return {r['m']: r['objective'] for r in self.results.values()
                if r.get('m') is not None and r['m'] >= 0}

    def _best(self):
        t = self._hinge_trials()
        return max(t, key=t.get)

    def search(self):
        a = self.args
        if 'baseline' not in self.results:
            say("scoring the warm start as the baseline ...")
            res = self.ev.score(self.warm_ckpt)
            res.update(m=-1, checkpoint=str(self.warm_ckpt))   # m=-1: not a hinge trial
            self.results['baseline'] = res
            self._save()
        b = self.results['baseline']
        say(f"baseline: F1w={b['objective']:.5f}  P={b['precision']:.4f}  R={b['recall']:.4f}")
        self.phase_table(b)
        if a.control or a.include_zero:
            say("=" * 64)
            say("reference runs (not part of the bracket search)")
        if a.control:
            self.run(None)
        if a.include_zero:
            self.run(0.0)

        # 1. bracket
        say("=" * 64)
        say(f"PHASE 1/3  bracket: m={a.m_mid:g} (mid) and m={a.m_far:g} (far)")
        f_mid, f_far = self.run(a.m_mid), self.run(a.m_far)
        if f_far > f_mid + a.min_delta:
            say(f"far beats mid ({f_far:.5f} vs {f_mid:.5f}): the best m may be larger -> expand")
        else:
            say(f"far does not beat mid by min-delta ({f_far:.5f} vs {f_mid:.5f}): "
                f"best m is in (0, {a.m_far:g}] -> no expansion")

        # 2. expand while the largest m keeps winning
        say("=" * 64)
        say("PHASE 2/3  expand: double m while the largest m keeps winning")
        for k in range(a.max_doublings):
            t = self._hinge_trials()
            top = max(t)
            others = [v for kk, v in t.items() if kk != top]
            if not others:
                say(f"only m={top:g} has been tried; nothing to compare -> stop expanding")
                break
            if t[top] <= max(others) + a.min_delta:
                say(f"largest m={top:g} (F1w {t[top]:.5f}) is not best by min-delta over "
                    f"the rest (best other {max(others):.5f}) -> stop expanding")
                break
            say(f"largest m={top:g} is best ({t[top]:.5f} vs {max(others):.5f}) -> "
                f"doubling to m={2 * top:g} ({k + 1}/{a.max_doublings})")
            self.run(2 * top)
        else:
            if a.max_doublings:
                say(f"hit --max-doublings={a.max_doublings}; the optimum may lie above "
                    f"m={max(self._hinge_trials()):g}")

        # 3. refine around the best by halving its neighbouring intervals
        say("=" * 64)
        say(f"PHASE 3/3  refine: halve the intervals around the best m ({a.refine_rounds} rounds)")
        for r in range(a.refine_rounds):
            ms = sorted(self._hinge_trials())
            best = self._best()
            i = ms.index(best)
            left = ms[i - 1] if i > 0 else 0.0
            right = ms[i + 1] if i + 1 < len(ms) else None
            todo = [(left + best) / 2] + ([(best + right) / 2] if right is not None else [])
            todo = [m for m in todo if all(abs(m - k) > a.m_tol for k in ms)]
            if not todo:
                say(f"bracket around m={best:g} is narrower than --m-tol={a.m_tol:g}; stopping.")
                break
            say(f"round {r + 1}/{a.refine_rounds}: best m={best:g} in "
                f"[{left:g}, {'open' if right is None else f'{right:g}'}] -> trying "
                f"{', '.join(f'{m:g}' for m in todo)}")
            for m in todo:
                self.run(m)
        say(f"search finished in {(time.time() - self.t_start) / 3600:.2f} h "
            f"({self.trials_run} trial(s) trained this session)")
        self.report()

    def report(self):
        rows = sorted(self.results.values(), key=lambda r: (r['m'] is not None, r['m'] or 0))
        base = self.results['baseline']
        print("\n" + "=" * 72)
        print(f"{'trial':>10s} {'F1w':>9s} {'dF1w':>9s} {'P':>8s} {'R':>8s}")
        for r in rows:
            name = ('baseline' if r['m'] == -1 else 'no hinge' if r['m'] is None
                    else f"m={r['m']:g}")
            print(f"{name:>10s} {r['objective']:9.5f} {r['objective'] - base['objective']:+9.5f} "
                  f"{r['precision']:8.4f} {r['recall']:8.4f}")
        best = self._best()
        bres = self.results[_tag(best)]
        ref = self.results.get(_tag(None), base)
        print(f"\nBest m = {best:g}   checkpoint: {bres['checkpoint']}")
        print(f"Per-phase F1 change vs {'no-hinge control' if ref is not base else 'baseline'} "
              f"(sorted by weight):")
        order = np.argsort(-self.ev.w)
        for i in order:
            n = self.ev.names[i]
            d = bres['per_phase'][n]['f1'] - ref['per_phase'][n]['f1']
            pp = bres['per_phase'][n]
            print(f"    {n:22s} w={self.ev.w[i]:.2f}  F1 {pp['f1']:.4f} ({d:+.4f})  "
                  f"P {pp['precision']:.3f}  R {pp['recall']:.3f}")
        print(f"\nResults: {self.results_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='Training recipe (yaml).')
    p.add_argument('--warm-start', default=None,
                   help="Checkpoint every trial starts from (name in checkpoints.load_dir, "
                        "no .tar). Default: the recipe's warm_start.")
    p.add_argument('--episode', default=None,
                   help='train<N> block to run per trial. Default: the last one in the recipe.')
    p.add_argument('--epochs', type=int, default=15,
                   help="Epochs per trial (overrides the episode's). Default 15.")
    p.add_argument('--m-mid', type=float, default=1.0)
    p.add_argument('--m-far', type=float, default=4.0)
    p.add_argument('--max-doublings', type=int, default=4)
    p.add_argument('--refine-rounds', type=int, default=3)
    p.add_argument('--min-delta', type=float, default=1e-3,
                   help='F1w gain a larger m must show to keep doubling (noise guard).')
    p.add_argument('--m-tol', type=float, default=0.05,
                   help='Stop refining once neighbouring m values are closer than this.')
    p.add_argument('--include-zero', action='store_true',
                   help='Also train m=0 (otherwise 0 is only the lower bound).')
    p.add_argument('--no-control', dest='control', action='store_false',
                   help='Skip the hinge-off control run.')
    p.add_argument('--eval-bundle', default=None,
                   help='Bundle to score on. Default: <tarname>_Valid, else <tarname>_Test.')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--python', default=sys.executable, help='Interpreter for main.py.')
    p.add_argument('--echo', choices=['key', 'all', 'none'], default='key',
                   help="Training output shown live: 'key' (epochs, new bests, presence P/R; "
                        "default), 'all', or 'none'. The per-trial log always has everything.")
    args = p.parse_args()
    if not 0 < args.m_mid < args.m_far:
        p.error('need 0 < --m-mid < --m-far')
    Search(args).search()


if __name__ == '__main__':
    main()
