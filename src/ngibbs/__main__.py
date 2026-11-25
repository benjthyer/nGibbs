"""Command line entry for the nGibbs package (also installed as `ngibbs`).

    ngibbs models [NAME | all]
    ngibbs tune [--verify] [--show] [--clear] [--model NAME] ...
"""
import sys


def _usage() -> int:
    print(__doc__.strip())
    print("\nCommands:\n"
          "  models  list the models on Hugging Face: the newest version of each, every\n"
          "          version of one model (ngibbs models MELTS120), or everything (all)\n"
          "  tune    tune and persist VRAM-aware GPU batch sizes (ngibbs tune --help)")
    return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        return _usage()
    command, rest = argv[0], argv[1:]
    if command == 'models':
        if rest and rest[0] in ('-h', '--help'):
            print("usage: ngibbs models [NAME | all]")
            return 0
        from .registry import available_models
        available_models(rest[0] if rest else None)
        return 0
    if command == 'tune':
        from .utils.batch_tuning import main as tune_main
        return tune_main(rest)
    print(f"Unknown command {command!r}.", file=sys.stderr)
    return _usage()


if __name__ == '__main__':
    raise SystemExit(main())
