import pandas as pd
import argparse
import subprocess


get_boot_cost_dict = dict()


def get_boot_cost(row):
    precision, norm2 = row["fbs_size"], row["norm2_linprod"]
    key = (precision, norm2)
    if key not in get_boot_cost_dict:
        print(f"Call {args.opt} with {precision} {norm2}")
        output = subprocess.check_output([args.opt, f"--precision={precision}", f"--sq-norm2={norm2}"])
        get_boot_cost_dict[key] = int(output.decode().split(",")[-2].strip())
    return get_boot_cost_dict[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Append execution estimates to aggregated logs csv file")
    parser.add_argument("-i", "--inp", required=True, help="input csv")
    parser.add_argument("--opt", default="./concrete/compilers/concrete-optimizer/optimizer",
                        help="path to patched concrete v0-optimizer")
    parser.add_argument(
        "-o", "--out", default="out_est.csv", help="output csv")

    args = parser.parse_args()

    df = pd.read_csv(args.inp)

    df.sort_values(["bench", "mapper", "fbs_size"], inplace=True)
    df["boot_cost"] = df.apply(get_boot_cost, axis=1)

    df.to_csv(args.out, index=False)
