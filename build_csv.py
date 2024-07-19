import glob
from pathlib import Path
import pandas as pd
import ast
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse log files and aggregate to a csv")
    parser.add_argument("prefix",
                        help="prefix where to glob for .log files")
    parser.add_argument(
        "-o", "--out", default="out.csv", help="output csv")

    args = parser.parse_args()

    data = list()
    for file in glob.glob(args.prefix + "*.log"):
        print(file)
        with open(file) as f:
            lines = f.readlines()
            if len(lines):
                last = lines[-1]
                d = ast.literal_eval(last)
                d["bench"] = Path(d["filename"]).stem
                data.append(d)

    df = pd.DataFrame(data)

    df.sort_values(["bench", "mapper", "fbs_size"], inplace=True)

    df.to_csv(args.out, index=False)
