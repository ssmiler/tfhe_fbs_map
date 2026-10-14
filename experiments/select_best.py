#!/usr/bin/env python3
"""Copy each benchmark's lowest-cost search .lbf, .log, and input into experiments/best/."""

import argparse
import csv
import math
from pathlib import Path
import shutil


EXPERIMENTS = Path(__file__).resolve().parent


def select_files(csv_paths, destination):
    copies = []
    destinations = set()
    for csv_path in csv_paths:
        if not csv_path.name.endswith("_agg_est.csv"):
            raise ValueError(f"expected a *_agg_est.csv file: {csv_path}")
        suite = csv_path.name.removesuffix("_agg_est.csv")
        if not suite:
            raise ValueError(f"missing suite name: {csv_path}")
        best = {}
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"empty CSV: {csv_path}")
            cost_column = reader.fieldnames[-1]
            for row in reader:
                if row["mapper"] != "search":
                    continue
                bench = row["bench"]
                if not bench or Path(bench).name != bench or bench in (".", ".."):
                    raise ValueError(f"invalid benchmark name: {bench!r}")
                cost = float(row[cost_column])
                size = int(row["fbs_size"])
                if not math.isfinite(cost) or cost < 0 or size < 2:
                    raise ValueError(f"invalid cost or FBS size for {bench} in {csv_path}")
                key = (cost, size)
                if bench not in best or key < best[bench][0]:
                    best[bench] = (key, row["filename"])
        if not best:
            raise ValueError(f"no search results in {csv_path}")
        for bench, ((cost, size), filename) in sorted(best.items()):
            input_path = Path(filename)
            if not input_path.is_absolute():
                input_path = EXPERIMENTS / input_path
            files = [
                (csv_path.parent / suite / f"{bench}_{size}_search{extension}",
                 destination / suite / f"{bench}{extension}")
                for extension in (".lbf", ".log")
            ]
            files.append((input_path, destination / suite / input_path.name))
            for source, target in files:
                if not source.is_file():
                    raise FileNotFoundError(f"missing selected file: {source}")
                if target in destinations:
                    raise ValueError(f"duplicate selected destination: {target}")
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"destination is not a regular file: {target}")
                destinations.add(target)
                copies.append((source, target))
            print(f"{suite}/{bench}: FBS size {size}, {cost_column} {cost:g}")
    return copies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="*", help="*_agg_est.csv files (default: outputs/*_agg_est.csv)")
    parser.add_argument("--dry-run", action="store_true", help="validate and list selections without copying files")
    args = parser.parse_args()
    csv_paths = args.csv or sorted((EXPERIMENTS / "outputs").glob("*_agg_est.csv"))
    if not csv_paths:
        parser.error("no *_agg_est.csv files found")
    try:
        copies = select_files(csv_paths, EXPERIMENTS / "best")
        if not args.dry_run:
            for source, target in copies:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{'Would copy' if args.dry_run else 'Copied'} {len(copies)} files ({len(copies) // 3} circuits).")


if __name__ == "__main__":
    main()
