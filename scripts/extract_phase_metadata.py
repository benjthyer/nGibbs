"""
Extract lines from a metadata text file that contain phase names/abundances
after the batch descriptor (e.g. "120batch 140 graphite:0.0004").

Usage:
    python extract_phase_metadata.py <input_file> [output_file]

If output_file is omitted, writes to <input_file>_phases.txt.
"""

import sys
from pathlib import Path


def extract_phases(input_path: Path, output_path: Path) -> int:
    results = []
    with input_path.open("r") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split(" ")
            if len(parts) > 2:
                phase_text = " ".join(parts[1:])
                results.append(f"{idx}\t{phase_text}")

    with output_path.open("w") as f:
        f.write("\n".join(results))
        if results:
            f.write("\n")

    return len(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_phase_metadata.py <input_file> [output_file]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(input_path.stem + "_phases.txt")

    count = extract_phases(input_path, output_path)
    print(f"Found {count} lines with phase data -> {output_path}")


if __name__ == "__main__":
    main()
