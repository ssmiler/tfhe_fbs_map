import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools as it

def add_best_cols(df):
    df["nb_bootstrap_best"] = df.groupby(
        ["bench", "mapper"]).nb_bootstrap.cummin()
    df["total_cost"] = df["nb_bootstrap"] * df["boot_cost"]
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
    # reference cost, naive cost, search cost
    df1 = df
    t_ref = df1.loc[df1.mapper == "basic", ["total_cost", "nb_bootstrap", "bench"]].set_index("bench")

    def g(df1):
        t = df1.groupby("bench").apply(lambda d: d.iloc[d.total_cost.argmin()].loc[["total_cost", "nb_bootstrap", "fbs_size", "max_lut_size"]], include_groups=False)
        t["ratio_cost"] = t_ref["total_cost"] / t["total_cost"]
        t["ratio_boots"] = (t["nb_bootstrap"] / t_ref["nb_bootstrap"] - 1)*100
        t["FBS size"] = t.apply(lambda e: f"{int(e.fbs_size)} ({int(e.max_lut_size):2})", axis = 1)
        t.drop(["total_cost", "nb_bootstrap", "fbs_size", "max_lut_size"], inplace=True, axis=1)
        t.columns = ["speedup", "boots.", "FBS size"]
        return t

    df1 = df
    df1 = df1[df1.mapper == "naive"]
    t_naive = g(df1)
    t_naive.columns = pd.MultiIndex.from_tuples(it.product(["naive"], t_naive.columns))

    df1 = df
    df1 = df1[df1.mapper == "search"]
    t_search = g(df1)
    t_search.columns = pd.MultiIndex.from_tuples(it.product(["search"], t_search.columns))

    df1 = pd.concat([t_naive, t_search], axis = 1)
    return df1


def f2b(df):
    # reference cost, naive cost, search cost
    df1 = df
    t_ref = df1.loc[df1.mapper == "basic", ["total_cost", "nb_bootstrap", "bench"]].set_index("bench")

    def g(df1):
        t = df1.groupby("bench").apply(lambda d: d.iloc[d.total_cost.argmin()].loc[["total_cost", "nb_bootstrap", "fbs_size", "max_lut_size"]], include_groups=False)
        t["ratio_cost"] = (t["total_cost"] / t_ref["total_cost"] - 1) * 100
        t["ratio_boots"] = (t["nb_bootstrap"] / t_ref["nb_bootstrap"] - 1) * 100
        t["FBS size"] = t.apply(lambda e: f"{int(e.fbs_size)} ({int(e.max_lut_size)})", axis = 1)
        t.drop(["total_cost", "nb_bootstrap", "fbs_size", "max_lut_size"], inplace=True, axis=1)
        t.columns = ["cost", "\#boots.", "FBS size"]
        return t

    df1 = df
    df1 = df1[df1.mapper == "naive"]
    t_naive = g(df1)
    t_naive.columns = pd.MultiIndex.from_tuples(it.product(["naive"], t_naive.columns))

    df1 = df
    df1 = df1[df1.mapper == "search"]
    t_search = g(df1)
    t_search.columns = pd.MultiIndex.from_tuples(it.product(["search"], t_search.columns))

    df1 = pd.concat([t_naive, t_search], axis = 1)
    return df1


def f4(df):
    # reference cost, naive cost, search cost
    df1 = df
    t_ref = df1.loc[df1.mapper == "basic", ["nb_bootstrap", "bench"]].set_index("bench")

    def g(df1):
        t = df1.groupby("bench").apply(lambda d: d.iloc[d.nb_bootstrap.argmin()].loc[["nb_bootstrap", "fbs_size", "max_lut_size"]], include_groups=False)
        t["ratio_boots"] = t_ref["nb_bootstrap"] / t["nb_bootstrap"]
        t["FBS size"] = t.apply(lambda e: f"{int(e.fbs_size)} ({int(e.max_lut_size):2})", axis = 1)
        t.drop(["nb_bootstrap", "fbs_size", "max_lut_size"], inplace=True, axis=1)
        t.columns = ["speedup", "FBS size"]
        return t

    # df1 = df
    # df1 = df1[df1.mapper == "naive"]
    # df1 = df1[df1.fbs_size <= 32]
    # t_naive = g(df1)
    # t_naive.columns = pd.MultiIndex.from_tuples(it.product(["naive"], t_naive.columns))

    df1 = df
    df1 = df1[df1.mapper == "search"]
    df1 = df1[df1.fbs_size <= 32]
    t_search = g(df1)
    t_search.columns = pd.MultiIndex.from_tuples(it.product(["search"], t_search.columns))

    df1 = t_search
    return df1



# epfl
df = pd.read_csv("epfl_agg_est.csv")
df = add_best_cols(df)
bench_arith = ["adder", "bar", "div", "hyp", "log2",
               "max", "multiplier", "sin", "sqrt", "square"]
bench_ctrl = ["arbiter", "cavlc", "ctrl", "dec", "i2c",
              "int2float", "mem_ctrl", "priority", "router", "voter"]

e = df[df.mapper == "basic"].copy()
e.mapper = "naive"
df = pd.concat((df, e))

df1 = f2b(df)
df1 = df1.loc[bench_arith + bench_ctrl]
df1.index = df1.index.str.replace("_", "\_")
df1.index = "\\textsf{" + df1.index + "}"
df1.loc["avg."] = df1.mean(numeric_only = True)

df2 = df1.astype(str)
df2.iloc[:, [0,1,3,4]] = ""

idx = df1.iloc[:, 0] > -0.5
df2.iloc[:, [0, 1]] = df1.iloc[:, [0, 1]].map(lambda e: f"${e:2.0f}\%$")
df2.iloc[idx, [0, 1, 2]] = ""

idx = df1.iloc[:, 3] > -0.5
df2.iloc[:, [3, 4]] = df1.iloc[:, [3, 4]].map(lambda e: f"${e:2.0f}\%$")
df2.iloc[idx, [3, 4, 5]] = ""

df2.iloc[-1, [2,5]] = ""

print(df2.to_latex())



# epfl timings

# Execution time (ms) divided by the number of input circuit gates for each FBS size
df1 = df
df1 = df1[df1.mapper == "naive"]
df1 = df1[df1.bench.isin(bench_arith + bench_ctrl)]
d1 = df1.groupby("fbs_size").apply(lambda d: (d.time / d.nb_bootstrap.iloc[0]).mean(), include_groups=False)*1000
plt.plot(d1.index, d1.values, ".-", label="naive")

df1 = df
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.bench.isin(bench_arith + bench_ctrl)]
d1 = df1.groupby("fbs_size").apply(lambda d: (d.time / d.nb_bootstrap.iloc[0]).mean(), include_groups=False)*1000
plt.plot(d1.index, d1.values, ".-", label="search")

plt.legend()
plt.ylabel('execution time per gate (ms)')
plt.xlabel('FBS size (p)')
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
df1.index = df1.index.str.slice(stop = -4)
df1.sort_index(key = lambda e: e.str.slice(start=1).astype(int), inplace=True)
df1 = df1.join(t_autohog, on = "bench")
df1.index = "\\textsf{" + df1.index + "}"
df1.loc["avg."] = df1.mean(numeric_only = True)

df2 = df1.astype(str)
df2.iloc[:, [0,2]] = ""

s = df1.loc[:, df1.columns.get_level_values(1)=='speedup'].map(lambda e: "-" if np.isnan(e) else f"{e:.2f}\\times")
df2.loc[:, df2.columns.get_level_values(1)=='speedup'] = s
for i in range(df2.shape[0]):
    m = 10000 if df2.iloc[i, 2] == "-" else df2.iloc[i, [0, 2]].max()
    for j in [0, 2]:
        if df2.iloc[i, j] == m:
            df2.iloc[i, j] = f"\\mathbf{{{df2.iloc[i, j]}}}"
        df2.iloc[i, j] = f"${df2.iloc[i, j]}$"

print(df2.to_latex())


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
df1.sort_index(key = lambda e: e.str.slice(start=1).astype(int), inplace=True)
df1 = df1.join(t_autohog, on = "bench")
df1.index = "\\textsf{" + df1.index + "}"
df1.loc["avg."] = df1.mean(numeric_only = True)

df2 = df1.astype(str)
df2.iloc[:, [0,2]] = ""

s = df1.loc[:, df1.columns.get_level_values(1)=='speedup'].map(lambda e: "-" if np.isnan(e) else f"{e:.2f}\\times")
df2.loc[:, df2.columns.get_level_values(1)=='speedup'] = s
for i in range(df2.shape[0]):
    m = 10000 if df2.iloc[i, 2] == "-" else df2.iloc[i, [0, 2]].max()
    for j in [0, 2]:
        if df2.iloc[i, j] == m:
            df2.iloc[i, j] = f"\\mathbf{{{df2.iloc[i, j]}}}"
        df2.iloc[i, j] = f"${df2.iloc[i, j]}$"

print(df2.to_latex())




# bristol
df = pd.read_csv("bristol_agg_est.csv")
df = add_best_cols(df)

df1 = df
# df1 = df1[df1.bench == "AES-non-expanded"]
df1 = df1[df1.bench == "aes_128"]
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.fbs_size <= 16]
d = df1[df1.strict_fbs_size == False]
fig, ax1 = plt.subplots()
ax1.set_xlabel('FBS size (p)')
ax1.set_ylabel('number of bootstraps', color='b')
ax1.plot(d.fbs_size, d.nb_bootstrap, 'b.-')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.set_ylabel('evaluation cost', color='r')
ax2.plot(d.fbs_size, d.total_cost, 'r.-')
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
df1 = df1[df1.fbs_size <= 12]
for bench, cnt in zip(["trivium_iter", "kreyvium_iter"], [8, 10]):
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    ax1 = axs[0]
    ax1.set_ylabel('number of bootstraps')
    d = df1[df1.bench == f"{bench}_v1"]
    ax1.plot(d.fbs_size, d.nb_bootstrap, 'b.-', label = "version 1")
    d = df1[df1.bench == f"{bench}_v2"]
    ax1.plot(d.fbs_size, d.nb_bootstrap, 'g.-', label = "version 2")
    ax1.plot([4], [cnt], 'ro')
    ax1.tick_params(axis='y')
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax2 = axs[1]
    ax2.set_xlabel('FBS size (p)')
    ax2.set_ylabel('evaluation cost')
    d = df1[df1.bench == f"{bench}_v1"]
    ax2.plot(d.fbs_size, d.total_cost, 'b.-', label = "")
    d = df1[df1.bench == f"{bench}_v2"]
    ax2.plot(d.fbs_size, d.total_cost, 'g.-', label = "")
    ax2.plot([4], [cnt * 40], 'ro')
    ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(ncol = 2)
    plt.savefig(f"{bench}.pdf")

# aes sbox
df1 = df
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.fbs_size <= 14]
df1 = df1[df1.fbs_size >= 4]
d = df1[df1.bench == "aes_sbox"]

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True)
ax1 = axs[0]
ax1.set_ylabel('number of bootstraps')
ax1.plot(d.fbs_size, d.nb_bootstrap, 'b.-')
ax1.plot([11], [36], 'ro') # 36 is nb. FBS from Bon et al.
ax1.tick_params(axis='y')
ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

ax2 = axs[1]
ax2.set_xlabel('FBS size (p)')
ax2.set_ylabel('evaluation cost')
ax2.plot(d.fbs_size, d.total_cost, 'b.-')
ax2.plot([11], [36 * 69], 'ro') # 69 is the FBS cost with precision 11 and sq-norm2 > 2
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.legend(ncol = 2)
plt.savefig(f"aes_sbox.pdf")

# ascon lut
df1 = df
df1 = df1[df1.mapper == "search"]
df1 = df1[df1.fbs_size >= 7]
d = df1[df1.bench == "ascon_lut"]

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True)
ax1 = axs[0]
ax1.set_ylabel('number of bootstraps')
ax1.plot(d.fbs_size, d.nb_bootstrap, 'b.-')
ax1.plot([17], [5], 'ro') # 5 is nb. FBS from Bon et al.
ax1.tick_params(axis='y')
ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

ax2 = axs[1]
ax2.set_xlabel('FBS size (p)')
ax2.set_ylabel('evaluation cost')
ax2.plot(d.fbs_size, d.total_cost, 'b.-')
ax2.plot([17], [5 * 75], 'ro') # 5 is the FBS cost with precision 11 and sq-norm2 > 2
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.legend(ncol = 2)
plt.savefig(f"ascon_lut.pdf")
