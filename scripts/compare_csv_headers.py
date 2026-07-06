#!/usr/bin/env python3
"""Compare the header row of two CSV files and report headers unique to each."""

import argparse
import csv
import sys


def read_header(path):
    with open(path, newline="") as f:
        row = next(csv.reader(f), [])
    return [h.strip() for h in row]


def main():
    parser = argparse.ArgumentParser(description="Compare CSV headers between two files.")
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()

    headers1 = read_header(args.file1)
    headers2 = read_header(args.file2)

    set1, set2 = set(headers1), set(headers2)

    only_in_1 = [h for h in headers1 if h not in set2]
    only_in_2 = [h for h in headers2 if h not in set1]

    if not only_in_1 and not only_in_2:
        print("Headers match exactly.")
        return

    if only_in_1:
        print(f"Only in {args.file1}:")
        for h in only_in_1:
            print(f"  {h}")
    else:
        print(f"Only in {args.file1}: (none)")

    if only_in_2:
        print(f"Only in {args.file2}:")
        for h in only_in_2:
            print(f"  {h}")
    else:
        print(f"Only in {args.file2}: (none)")

    sys.exit(1 if (only_in_1 or only_in_2) else 0)


if __name__ == "__main__":
    main()
