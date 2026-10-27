import pandas as pd
import argparse
import subprocess
from pathlib import Path


get_boot_cost_dict = dict()


def get_boot_cost(row):
    precision, norm2 = int(row["execution_fbs_size"]), row["norm2_linprod"]
    key = (precision, norm2)
    if key not in get_boot_cost_dict:
        print(f"Call {args.opt} with {precision} {norm2}")
        output = subprocess.check_output([args.opt, f"--precision={precision}", f"--sq-norm2={norm2}"])
        get_boot_cost_dict[key] = int(output.decode().split(",")[-2].strip())
    return get_boot_cost_dict[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Append execution estimates to aggregated logs csv file")
    parser.add_argument("inp", nargs="+", help="input csv")
    parser.add_argument(
        "--opt",
        default=str(
            Path(__file__).resolve().parent
            / "concrete/compilers/concrete-optimizer/target/release/v0-parameters"
        ),
        help="path to the compiled patched Concrete v0-parameters binary",
    )

    args = parser.parse_args()

    for name in args.inp:
        df = pd.read_csv(name)

        df.sort_values(["bench", "mapper", "fbs_size"], inplace=True)
        df["boot_cost"] = df.apply(get_boot_cost, axis=1)
        df["total_cost"] = df["boot_cost"] * df["nb_bootstrap"] 

        out_name = name.replace(".csv", "_est.csv")
        df.to_csv(out_name, index=False)
