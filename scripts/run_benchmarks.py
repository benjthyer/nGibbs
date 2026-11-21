"""Generate or check the benchmark for every model in ngibbs.engine.models.

Each model is scored on its validation dataset and its standard workspace (the
raw standard-rock runs shipped with it). Baselines live with the model checkpoints,
under src/ngibbs/engine/TrainedModels/<family>/benchmark_baseline/<cpu|cuda>/. See
ngibbs.deployment_tests.benchmark.

Usage:
    python scripts/run_benchmarks.py                 # check every model against its baseline
    python scripts/run_benchmarks.py --no-plots      # same, without drawing plots
    python scripts/run_benchmarks.py --generate      # (re)write the baselines (tables + plots)
    python scripts/run_benchmarks.py --models HeFESToEmulator MELTS120Emulator --device cuda

Exits non-zero if any checked model regresses, is missing a baseline, or errors.
"""

import argparse
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', action='store_true',
                        help='write baselines instead of checking against them')
    parser.add_argument('--models', nargs='*', help='model names (default: all)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--no-plots', dest='plots', action='store_false',
                        help='skip drawing plots (drawn by default; git-ignored, not packaged)')
    args = parser.parse_args()

    from ngibbs.deployment_tests import BenchmarkRegressionError
    from ngibbs.engine.models import CPU_MODELS, GPU_MODELS

    models = GPU_MODELS if args.device == 'cuda' else CPU_MODELS
    if not models:
        print(f'No models available on {args.device}.')
        return 1
    names = args.models or sorted(models)
    unknown = [n for n in names if n not in models]
    if unknown:
        print(f'Unknown model(s) {unknown}; available: {sorted(models)}')
        return 1

    status = {}
    for name in names:
        model = models[name]
        print(f'\n===== {name} =====')
        try:
            if args.generate:
                idx = model.benchmark(model.benchmark_baseline_dir, plots=args.plots)
                bad = [k for k, c in idx['components'].items() if c['status'] == 'error']
                status[name] = f'baseline written ({len(idx["tables"])} tables' + \
                    (f'; errored components: {bad})' if bad else ')')
            else:
                if not (model.benchmark_baseline_dir / 'benchmark_index.json').is_file():
                    status[name] = 'FAILED: no baseline (run with --generate)'
                    continue
                r = model.check_benchmark(plots=args.plots)
                status[name] = 'passed' + (' (checkpoints changed)' if r['model_changed'] else '')
        except BenchmarkRegressionError as e:
            status[name] = f"FAILED: see {e.comparison['report_dir']}"
        except Exception:
            traceback.print_exc()
            status[name] = 'ERRORED'

    print('\n===== Summary =====')
    for name, s in status.items():
        print(f'  {name}: {s}')
    return 1 if any(s.startswith(('FAILED', 'ERRORED')) for s in status.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
