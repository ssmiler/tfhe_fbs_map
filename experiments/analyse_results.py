import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools as it

parser = argparse.ArgumentParser()
parser.add_argument(
    "--table-format",
    choices=("latex", "markdown", "console"),
    default="latex",
    help="format used for tables written to stdout (default: latex)",
)
args = parser.parse_args()


def print_table(df, index=True):
    if args.table_format == "latex":
        print(df.to_latex(index=index))
        return

    def cell(value):
        if pd.isna(value) or str(value) in ("nan", "NaN"):
            return ""
        value = str(value).replace("\\#", "#").replace("\n", " ")
        return value.replace("|", "\\|") if args.table_format == "markdown" else value

    columns = df.columns
    if isinstance(columns, pd.MultiIndex):
        columns = [" ".join(str(part) for part in column if part) for column in columns]
    else:
        columns = [str(column) for column in columns]
    columns = list(map(cell, columns))

    if args.table_format == "console":
        table = df.copy()
        table.columns = columns
        table = table.map(cell)
        if index:
            table.insert(0, cell(df.index.name or "bench"), list(map(cell, df.index)))
        print(table.to_string(index=False))
        print()
        return

    headers = ([df.index.name or "bench"] if index else []) + columns
    rows = df.itertuples(index=index, name=None)

    print("| " + " | ".join(map(cell, headers)) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(map(cell, row)) + " |")
    print()


def format_benchmark_index(index):
    if args.table_format == "latex":
        return "\\textsf{" + index.str.replace("_", "\\_") + "}"
    return index


def format_result_table(df):
    formatted = df.astype(str)
    for column in df.columns:
        label = column[-1].replace("\\", "")
        if label in ("#gates", "#boots."):
            formatted[column] = df[column].map(
                lambda value: "" if pd.isna(value) else str(int(value)))
        elif label == "cost":
            prefix, suffix = ("$", "\\%$") if args.table_format == "latex" else ("", "%")
            formatted[column] = df[column].map(
                lambda value: "" if pd.isna(value) or round(value) == 0 else f"{prefix}{round(value)}{suffix}")
        if "avg." in df.index and label in ("#gates", "#boots.", "FBS size"):
            formatted.loc["avg.", column] = ""
    prefix, suffix = ("$", "\\%$") if args.table_format == "latex" else ("", "%")
    for mapper in ("search",):
        count_column = (mapper, "\\#boots.")
        decrease_column = (mapper, "decrease")
        cost_column = (mapper, "cost")
        size_column = (mapper, "FBS size")
        if count_column not in df.columns:
            continue
        formatted[count_column] = [
            "" if pd.isna(decrease) or round(decrease) == 0 else
            f"{count} ({prefix}{-round(decrease)}{suffix})".strip()
            for count, decrease in zip(formatted[count_column], df[decrease_column])
        ]
        if cost_column in df.columns and size_column in df.columns:
            formatted.loc[df[cost_column].isna() | (df[cost_column] >= 0), size_column] = ""
        formatted.drop(columns=[decrease_column], inplace=True)
    return formatted


def format_speedup_table(df):
    formatted = format_result_table(df)
    speedups = [("search", "speedup"), ("AutoHoG", "speedup")]
    for row in range(formatted.shape[0]):
        comparable = not np.isnan(df.iloc[row][speedups[1]])
        best = max(df.iloc[row][speedups]) if comparable else None
        for column in speedups:
            value = df.iloc[row][column]
            if np.isnan(value):
                text = "-"
            elif args.table_format == "latex":
                text = f"{value:.2f}\\times"
            else:
                text = f"{value:.2f}×"
            if comparable and value == best:
                if args.table_format == "latex":
                    text = f"\\mathbf{{{text}}}"
                elif args.table_format == "markdown":
                    text = f"**{text}**"
                else:
                    text = f"{text} (best)"
            formatted.loc[df.index[row], column] = f"${text}$" if args.table_format == "latex" else text
    return formatted

def add_best_cols(df):
    df["total_cost_best"] = df.groupby(
        ["bench", "mapper"]).total_cost.cummin()
    return df

def f1(df):
    t = df[df.mapper == "basic"]
    d = dict(zip(t.bench, t.nb_bootstrap))

    df1 = df[df.mapper == "search"]
    nb_bootstrap_input = df1.bench.map(d)

    t = (df1.time / nb_bootstrap_input).describe()
    print(f"Execution time per gate:\n{t}")


def f2(df):
    # reference cost and search cost
    df1 = df
    t_ref = df1.loc[df1.mapper == "basic", ["total_cost", "nb_bootstrap", "bench"]].set_index("bench")

    def g(df1):
        t = df1.groupby("bench").apply(lambda d: d.iloc[d.total_cost.argmin()].loc[["total_cost", "nb_bootstrap", "execution_fbs_size", "max_lut_size"]], include_groups=False)
        t["ratio_cost"] = t_ref["total_cost"] / t["total_cost"]
        t["ratio_boots"] = (t["nb_bootstrap"] / t_ref["nb_bootstrap"] - 1)*100
        t["FBS size"] = t.apply(lambda e: f"{int(e.execution_fbs_size)} ({int(e.max_lut_size):2})", axis = 1)
        t.drop(["total_cost", "nb_bootstrap", "execution_fbs_size", "max_lut_size"], inplace=True, axis=1)
        t.columns = ["speedup", "boots.", "FBS size"]
        return t

    df1 = df
    df1 = df1[df1.mapper == "search"]
    t_search = g(df1)
    t_search.columns = pd.MultiIndex.from_tuples(it.product(["search"], t_search.columns))

    df1 = t_search
    return df1


def f2b(df):
    # reference cost and search cost
    df1 = df
    t_ref = df1.loc[df1.mapper == "basic", ["total_cost", "nb_bootstrap", "bench"]].set_index("bench")

    def g(df1):
        t = df1.groupby("bench").apply(lambda d: d.iloc[d.total_cost.argmin()].loc[["total_cost", "nb_bootstrap", "execution_fbs_size", "max_lut_size"]], include_groups=False)
        t["ratio_cost"] = (t["total_cost"] / t_ref["total_cost"] - 1) * 100
        t["decrease"] = (1 - t["nb_bootstrap"] / t_ref["nb_bootstrap"]) * 100
        t["FBS size"] = t.apply(lambda e: f"{int(e.execution_fbs_size)} ({int(e.max_lut_size)})", axis = 1)
        t = t[["ratio_cost", "nb_bootstrap", "decrease", "FBS size"]].copy()
        t.columns = ["cost", "\\#boots.", "decrease", "FBS size"]
        return t

    df1 = df
    df1 = df1[df1.mapper == "search"]
    t_search = g(df1)
    t_search.columns = pd.MultiIndex.from_tuples(it.product(["search"], t_search.columns))

    df1 = t_search
    df1.insert(0, ("", "\\#gates"), t_ref["nb_bootstrap"])
    return df1


def f4(df):
    # reference cost and search cost
    df1 = df
    t_ref = df1.loc[df1.mapper == "basic", ["nb_bootstrap", "bench"]].set_index("bench")

    def g(df1):
        t = df1.groupby("bench").apply(lambda d: d.iloc[d.nb_bootstrap.argmin()].loc[["nb_bootstrap", "execution_fbs_size", "max_lut_size"]], include_groups=False)
        t["ratio_boots"] = t_ref["nb_bootstrap"] / t["nb_bootstrap"]
        t["decrease"] = (1 - t["nb_bootstrap"] / t_ref["nb_bootstrap"]) * 100
        t["FBS size"] = t.apply(lambda e: f"{int(e.execution_fbs_size)} ({int(e.max_lut_size):2})", axis = 1)
        t = t[["nb_bootstrap", "decrease", "ratio_boots", "FBS size"]].copy()
        t.columns = ["\\#boots.", "decrease", "speedup", "FBS size"]
        return t

    df1 = df
    df1 = df1[df1.mapper == "search"]
    df1 = df1[df1.execution_fbs_size <= 32]
    t_search = g(df1)
    t_search.columns = pd.MultiIndex.from_tuples(it.product(["search"], t_search.columns))

    df1 = t_search
    df1.insert(0, ("", "\\#gates"), t_ref["nb_bootstrap"])
    return df1



# epfl
df = pd.read_csv("epfl_agg_est.csv")
df = add_best_cols(df)
bench_arith = ["adder", "bar", "div", "hyp", "log2",
               "max", "multiplier", "sin", "sqrt", "square"]
bench_ctrl = ["arbiter", "cavlc", "ctrl", "dec", "i2c",
              "int2float", "mem_ctrl", "priority", "router", "voter"]

df1 = f2b(df)
df1 = df1.loc[bench_arith + bench_ctrl]
df1.index = format_benchmark_index(df1.index)
df1.loc["avg."] = df1.mean(numeric_only = True)

print_table(format_result_table(df1))



# epfl timings

# Execution time (ms) divided by the number of input circuit gates for each FBS size
df1 = df
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.bench.isin(bench_arith + bench_ctrl)]
d1 = df1.groupby("fbs_size").apply(lambda d: (d.time / d.nb_bootstrap.iloc[0]).mean(), include_groups=False)*1000
plt.plot(d1.index, d1.values, ".-", label="search")

plt.legend()
plt.ylabel('execution time per gate (ms)')
plt.xlabel('Mapping FBS size (p)')
plt.savefig("exec_time_per_fbs.pdf")



# iscas85
# from AutoHoG paper, Fig. 7 and text, TFHE/AutoHoG time
t_autohog = pd.DataFrame()
t_autohog.loc["c17", "speedup"] = 0.1 / 0.04
t_autohog.loc["c432", "speedup"] = 3 / 1.39
# t_autohog.loc["c499", "speedup"] = 3.3 / 1.4
# t_autohog.loc["c880", "speedup"] = 6.0 / 2.4
t_autohog.loc["c1355", "speedup"] = 8.99 / 1.49
# t_autohog.loc["c1908", "speedup"] = 10.5 / 2.5
# t_autohog.loc["c2670", "speedup"] = 15.3 / 3.1
t_autohog.loc["c3540", "speedup"] = 21.82 / 5.6
# t_autohog.loc["c5315", "speedup"] = 35.8 / 7.8
# t_autohog.loc["c6288", "speedup"] = 39.7 / 8
t_autohog.loc["c7552", "speedup"] = 45.81 / 8.06
t_autohog.columns = pd.MultiIndex.from_tuples(it.product(["AutoHoG"], t_autohog.columns))

# benchmarks values measured using a ruler in Fig. 7
ruler = ["c499", "c880", "c1908", "c2670", "c5315", "c6288"]

df = pd.read_csv("iscas85_agg_est.csv")
df = add_best_cols(df)
df1 = f4(df)
df1.index = df1.index.str.removesuffix("-xag")
df1.sort_index(key = lambda e: e.str.slice(start=1).astype(int), inplace=True)
df1 = df1.join(t_autohog, on = "bench")
df1.index = format_benchmark_index(df1.index)
df1.loc["avg."] = df1.mean(numeric_only = True)

print_table(format_speedup_table(df1))


# iscas89

# from AutoHoG paper, Table IV, TFHE/AutoHoG time
t_autohog = pd.DataFrame()
t_autohog.loc["s27", "speedup"] = 0.14 / 0.11
t_autohog.loc["s298", "speedup"] = 2.06 / 0.60
t_autohog.loc["s344", "speedup"] = 1.77 / 0.58
t_autohog.loc["s349", "speedup"] = 1.87 / 0.67
t_autohog.loc["s382", "speedup"] = 2.50 / 0.56
t_autohog.loc["s386", "speedup"] = 3.16 / 0.54
t_autohog.loc["s400", "speedup"] = 2.60 / 0.55
t_autohog.loc["s420", "speedup"] = 2.76 / 0.94
t_autohog.loc["s444", "speedup"] = 2.84 / 0.60
t_autohog.loc["s510", "speedup"] = 3.50 / 1.02
t_autohog.loc["s526", "speedup"] = 4.32 / 1.03
t_autohog.loc["s641", "speedup"] = 2.67 / 1.25
t_autohog.loc["s713", "speedup"] = 3.40 / 1.39
t_autohog.loc["s820", "speedup"] = 7.12 / 1.50
t_autohog.loc["s832", "speedup"] = 7.71 / 1.61
t_autohog.loc["s838", "speedup"] = 5.68 / 1.89
t_autohog.loc["s953", "speedup"] = 5.93 / 1.69
t_autohog.loc["s1196", "speedup"] = 6.44 / 1.55
t_autohog.loc["s1238", "speedup"] = 6.62 / 1.81
t_autohog.loc["s1423", "speedup"] = 8.40 / 2.79
t_autohog.loc["s1488", "speedup"] = 12.66 / 1.70
t_autohog.loc["s5378", "speedup"] = 23.15 / 3.15
t_autohog.loc["s9234", "speedup"] = 40.25 /11.18
t_autohog.loc["s13207", "speedup"] = 53.28 /22.84
t_autohog.loc["s15850", "speedup"] = 66.71 /30.11
t_autohog.loc["s35932", "speedup"] = 209.56 /65.69
t_autohog.loc["s38584", "speedup"] = 231.75 / 92.21
t_autohog.columns = pd.MultiIndex.from_tuples(it.product(["AutoHoG"], t_autohog.columns))


df = pd.read_csv("iscas89_agg_est.csv")
df = add_best_cols(df)
df1 = f4(df)
df1.index = df1.index.str.removesuffix("-xag")
df1.sort_index(key = lambda e: e.str.slice(start=1).astype(int), inplace=True)
df1 = df1.join(t_autohog, on = "bench")
df1.index = format_benchmark_index(df1.index)
df1.loc["avg."] = df1.mean(numeric_only = True)

print_table(format_speedup_table(df1))




# bristol
df = pd.read_csv("bristol_agg_est.csv")
df = add_best_cols(df)

df1 = df
# df1 = df1[df1.bench == "AES-non-expanded"]
df1 = df1[df1.bench == "aes_128"]
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.execution_fbs_size <= 16]
d = df1[df1.fbs_range != "p"].sort_values("execution_fbs_size")
fig, ax1 = plt.subplots()
ax1.set_xlabel('Execution FBS size (p)')
ax1.set_ylabel('number of bootstraps', color='b')
ax1.plot(d.execution_fbs_size, d.nb_bootstrap, 'b.-')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.set_ylabel('evaluation cost', color='r')
ax2.plot(d.execution_fbs_size, d.total_cost, 'r.-')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("AES circuit")
plt.savefig("aes.pdf")



# generated
df = pd.read_csv("generated_agg_est.csv")
df = add_best_cols(df)

# trivium/kreyvium
df1 = df
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.execution_fbs_size <= 12]
for bench, cnt in zip(["trivium_iter", "kreyvium_iter"], [8, 10]):
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    ax1 = axs[0]
    ax1.set_ylabel('number of bootstraps')
    d = df1[df1.bench == f"{bench}_v1"].sort_values("execution_fbs_size")
    ax1.plot(d.execution_fbs_size, d.nb_bootstrap, 'b.-', label = "version 1")
    d = df1[df1.bench == f"{bench}_v2"].sort_values("execution_fbs_size")
    ax1.plot(d.execution_fbs_size, d.nb_bootstrap, 'g.-', label = "version 2")
    ax1.plot([4], [cnt], 'ro')
    ax1.tick_params(axis='y')
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax2 = axs[1]
    ax2.set_xlabel('Execution FBS size (p)')
    ax2.set_ylabel('evaluation cost')
    d = df1[df1.bench == f"{bench}_v1"].sort_values("execution_fbs_size")
    ax2.plot(d.execution_fbs_size, d.total_cost, 'b.-', label = "")
    d = df1[df1.bench == f"{bench}_v2"].sort_values("execution_fbs_size")
    ax2.plot(d.execution_fbs_size, d.total_cost, 'g.-', label = "")
    ax2.plot([4], [cnt * 40], 'ro')
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(ncol = 2)
    plt.savefig(f"{bench}.pdf")

# aes sbox
df1 = df
df1 = df1[df1.mapper == "search"]
# df1 = df1[df1.execution_fbs_size <= 14]
# df1 = df1[df1.execution_fbs_size >= 4]
d = df1[df1.bench == "aes_sbox"]

# Bon et al. results
bon_aes_nb_fbs = 36
bon_aes_fbs_size = 11
bon_aes_total_cost = bon_aes_nb_fbs * 69 # 69 is the FBS cost with precision 11 and sq-norm2 > 2

bon_ascon_nb_fbs = 5
bon_ascon_fbs_size = 17
bon_ascon_total_cost = bon_ascon_nb_fbs * 75 # 75 is the FBS cost with precision 17 and sq-norm2 > 2

bon_simon_nb_fbs = 1
bon_simon_fbs_size = 9
bon_simon_total_cost = bon_simon_nb_fbs * 47 # 47 is the FBS cost with precision 9 and sq-norm2 > 2

best_aes_idx = d.total_cost.idxmin()
our_aes_nb_fbs = d.loc[best_aes_idx].nb_bootstrap
our_aes_fbs_size = d.loc[best_aes_idx].execution_fbs_size
our_aes_max_lut_size = d.loc[best_aes_idx].max_lut_size
our_aes_total_cost = d.loc[best_aes_idx].total_cost

df1 = df
df1 = df1[df1.mapper == "search"]
# df1 = df1[df1.execution_fbs_size >= 7]
d = df1[df1.bench == "ascon_lut"]

best_ascon_idx = d.total_cost.idxmin()
our_ascon_nb_fbs = d.loc[best_ascon_idx].nb_bootstrap
our_ascon_fbs_size = d.loc[best_ascon_idx].execution_fbs_size
our_ascon_max_lut_size = d.loc[best_ascon_idx].max_lut_size
our_ascon_total_cost = d.loc[best_ascon_idx].total_cost


df1 = df
df1 = df1[df1.mapper == "search"]
# df1 = df1[df1.execution_fbs_size >= 7]
d = df1[df1.bench == "simon_iter"]


best_simon_idx = d.total_cost.idxmin()
our_simon_nb_fbs = d.loc[best_simon_idx].nb_bootstrap
our_simon_fbs_size = d.loc[best_simon_idx].execution_fbs_size
our_simon_max_lut_size = d.loc[best_simon_idx].max_lut_size
our_simon_total_cost = d.loc[best_simon_idx].total_cost


comparison = [
    ("SIMON", bon_simon_nb_fbs, bon_simon_fbs_size, our_simon_nb_fbs, our_simon_fbs_size, our_simon_max_lut_size, bon_simon_total_cost / our_simon_total_cost),
    ("ASCON", bon_ascon_nb_fbs, bon_ascon_fbs_size, our_ascon_nb_fbs, our_ascon_fbs_size, our_ascon_max_lut_size, bon_ascon_total_cost / our_ascon_total_cost),
    ("AES s-box", bon_aes_nb_fbs, bon_aes_fbs_size, our_aes_nb_fbs, our_aes_fbs_size, our_aes_max_lut_size, bon_aes_total_cost / our_aes_total_cost),
]
if args.table_format == "latex":
    for name, bon_count, bon_size, our_count, our_size, our_max_size, speedup in comparison:
        print(f"{name} & {bon_count} & {bon_size} & {our_count} & {our_size} ({our_max_size}) & ${speedup:.2f}\\times$ \\\\")
else:
    comparison = pd.DataFrame(
        [
            (name, int(bon_count), int(bon_size), int(our_count), f"{int(our_size)} ({int(our_max_size)})", f"{speedup:.2f}×")
            for name, bon_count, bon_size, our_count, our_size, our_max_size, speedup in comparison
        ],
        columns=["Circuit", "Bon et al. #FBS", "Bon et al. FBS size", "Our #FBS", "Our FBS size", "Speedup"],
    )
    print_table(comparison, index=False)

