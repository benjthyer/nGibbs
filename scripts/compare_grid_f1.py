"""Per-phase F1 comparison from a grid_deployment_test.py output directory.

    python scripts/compare_grid_f1.py grid_test_out/<run> [--ref <model-label-substring>]

For each dataset, prints a phase x model table of F1 (precision/recall in brackets with
-v), the macro-F1, and each model's delta vs the reference model (default: the last
model in the manifest, typically the shipped one)."""
import argparse, csv, json
from pathlib import Path


def read_stats(path):
    rows = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        phases = header[1:]
        for row in r:
            rows[row[0]] = row[1:]
    out = {}
    for i, ph in enumerate(phases):
        def g(k):
            try:
                return float(rows[k][i])
            except (KeyError, ValueError, IndexError):
                return float('nan')
        p, rc, n = g('precision'), g('recall'), g('n_rows_present')
        f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else float('nan')
        out[ph] = dict(p=p, r=rc, f1=f1, n=n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--ref', default=None)
    ap.add_argument('-v', action='store_true')
    ap.add_argument('--min-n', type=float, default=30, help='skip phases with fewer GT rows')
    a = ap.parse_args()
    run = Path(a.run_dir)
    for ddir in sorted((run / 'stats').iterdir()):
        if not ddir.is_dir():
            continue
        files = sorted(ddir.glob('M*.csv'), key=lambda p: int(p.name.split('_')[0][1:]))
        stats = {f.stem.split('_', 1)[1]: read_stats(f) for f in files}
        models = list(stats)
        ref = next((m for m in models if a.ref and a.ref in m), models[-1])
        phases = [ph for ph in stats[ref] if stats[ref][ph]['n'] >= a.min_n]
        print(f"\n=== {ddir.name}   (ref = {ref})")
        w = max(len(p) for p in phases) + 1
        print(' ' * w + ''.join(f"{'M'+str(i+1):>10}" for i in range(len(models))) + '   n')
        macro = {m: [] for m in models}
        for ph in phases:
            line = f"{ph:<{w}}"
            for m in models:
                s = stats[m].get(ph)
                f1 = s['f1'] if s else float('nan')
                macro[m].append(f1)
                d = f1 - stats[ref][ph]['f1']
                flag = '!' if (m != ref and d < -0.01) else ('+' if (m != ref and d > 0.01) else ' ')
                line += f"{f1:9.4f}{flag}"
                if a.v and s:
                    line += f"({s['p']:.3f}/{s['r']:.3f})"
            print(line + f"  {int(stats[ref][ph]['n'])}")
        print(f"{'macro-F1':<{w}}" + ''.join(f"{sum(v)/len(v):9.4f} " for v in macro.values()))
        print(f"{'worst-F1':<{w}}" + ''.join(f"{min(v):9.4f} " for v in macro.values()))
        for i, m in enumerate(models):
            print(f"  M{i+1} = {m}")


if __name__ == '__main__':
    main()
