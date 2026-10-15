# Heuristic for mapping Boolean circuits to TFHE functional bootstrapping

This repository contains the code accompanying the paper *[A Fast Heuristic for Mapping Boolean Circuits to Functional Bootstrapping](https://eprint.iacr.org/2024/1204)*, together with its Rust implementation.

## Mapper

The mapper is implemented in Rust with no external crate dependencies. It accepts BLIF circuits with at most two inputs per gate, combinational ISCAS-style BENCH files, and Bristol Fashion circuits, and writes human-readable FBS and LBF circuits. BENCH `AND`, `OR`, `NAND`, `NOR`, `XOR`, and `XNOR` gates are lowered to two-input chains; `NOT`/`INV` and `BUFF`/`BUF` are also supported. The CLI detects `.bench` files automatically; use `--type bristol` for Bristol files.

Mappings are checked by logical simulation: exhaustively for circuits with at
most 16 inputs, otherwise with 1,000 deterministic samples by default. Use
`--verify-samples` to change the sample count. The mapper does not execute the
circuits on encrypted data.

Build and run it with:

```bash
cargo build --release
target/release/fbs-map circuit.blif \
    --fbs-size 3 \
    --mapper search \
    --trials 3 \
    --output circuit.fbs \
    --output-lbf circuit.lbf
```

The available mapping policies are:

- `basic`: maps each two-input Boolean gate independently to a functional bootstrap (FBS); unary gates use linear operations without bootstrapping.
- `naive`: composes gate-input cones using a fixed-radix linear combination. For independent Boolean inputs, this gives power-of-two weights such as `a + 2*b + 4*c + ...`, up to input order.
- `search`: composes gate-input cones by searching coefficient pairs that minimize a span bound on the FBS input encoding (the paper's greedy heuristic).

See [mapping-policy details](docs/heuristics.md) for coefficient selection,
bootstrap boundaries, and limits.

Use `--max-norm2 N` to limit the squared Euclidean norm of the linear-combination
coefficient vector, `sum_i(c_i²)`, to `N`. This limit is separate from search's
multivalue-table tie-breaker.
For `naive` and `search`, valid negacyclic test vectors can have
up to `2p` entries; `--strict-fbs-size` limits them to `p`. The `basic` policy
uses per-gate LUTs and does not enforce these composition-size limits. Run
`target/release/fbs-map --help` for all controls.

`--trials N` runs the selected mapper first with the topological order encoded
in the circuit, then with `N - 1` deterministic random topological orders
derived from `--seed`. The best mapping is retained and total mapping time is
reported.

> [!NOTE]
> Multi-output FBS support, inspired by [CIM19](https://eprint.iacr.org/2018/622), is experimental.

With `--multioutput-fbs`, bootstrap outputs that use the same input linear
combination are emitted as one multi-output FBS and counted once. Without the
flag, different test vectors remain separate FBSs. Mathematically, the two
full-adder outputs can share the input `a + b + cin`, using one bootstrap with
two test vectors.

The flag also changes mapping decisions: an encoding compatible with an existing
bootstrap input has zero additional FBS count in the mapper's cost model. A cone
can be realigned to a compatible existing encoding before bootstrapping. In
`naive` and `search`, multi-output mode defers eager circuit-output bootstraps;
shared cones are retained until composition requires a bootstrap. This can
create further sharing opportunities, but outputs are grouped only when their
materialized linear inputs coincide.

## How to run the experiments

### Prerequisites

Use [Rust 1.88 or newer](https://blog.rust-lang.org/2025/06/26/Rust-1.88.0/)
(the code uses Rust 2024 let chains), Python 3, Git,
`wget`, and a C/C++ build toolchain. From the repository root, install the Python
packages used by benchmark generation and result analysis, then enter
`experiments`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd experiments
```

Install [ABC](https://github.com/berkeley-abc/abc) and `libreadline` (required to
compile ABC). The installer pins ABC to revision `9478c17`.

```bash
sudo apt install build-essential libreadline-dev
bash install_abc.sh
```

The package-install command above is for Debian/Ubuntu systems. Before running
the benchmarks, clone Concrete, apply the patch for non-power-of-two plaintext
sizes, and build the cost-estimation binary:

```bash
git clone https://github.com/zama-ai/concrete.git -b nightly-2024.04.17
(cd concrete; git apply ../concrete.patch)
cargo build --release --manifest-path concrete/compilers/concrete-optimizer/Cargo.toml --bin v0-parameters
```

### Run the benchmarks

ISCAS85 BENCH circuits are mapped with `lib.genlib`, unmapped, and exported to
BLIF. ISCAS89 circuits are unrolled with `frames -F 10 -i` and exported to BLIF;
their unspecified initial DFF states become additional inputs. The generated
files are stored in `outputs/benchmarks_xag/iscas85` and
`outputs/benchmarks_xag/iscas89`, then read by the Rust mapper. EPFL, generated,
and Bristol circuits are read without additional ABC preprocessing.

From `experiments`, run the benchmark suites, estimate evaluation costs, and
copy their lowest-cost `search` mappings into `best/`:

```bash
bash run_benchmarks.sh
```

The script creates a `Makefile` for each benchmark suite (`epfl`, `iscas85`, etc.).
The makefile targets run the Rust mapper with specific parameters.
Each makefile execution writes to a suite log (`outputs/epfl.log` for EPFL benchmarks, etc.).
Runtime depends on the hardware and whether dependencies and inputs are already
available. The final selection step replaces existing `best/` files with the
selected results.


For example, the EPFL makefile generated by `gen_makefile_epfl.bash` maps
`benchmarks/epfl/arithmetic/adder.blif` with `search` at FBS size 15:

```
...
outputs/epfl/adder_15_search.fbs outputs/epfl/adder_15_search.lbf outputs/epfl/adder_15_search.log: benchmarks/epfl/arithmetic/adder.blif | outputs/epfl
        ../target/release/fbs-map benchmarks/epfl/arithmetic/adder.blif --fbs-size 15 --mapper search --output outputs/epfl/adder_15_search.fbs --output-lbf outputs/epfl/adder_15_search.lbf > outputs/epfl/adder_15_search.log 2>&1
...
```

### Analyse results

From `experiments`, generate tables and figures from the saved mapping results
and evaluation-cost estimates:

```bash
cd outputs
python3 ../analyse_results.py
# or generate Markdown tables ready for README.md
python3 ../analyse_results.py --table-format markdown
# or print aligned plain-text tables
python3 ../analyse_results.py --table-format console
```

> [!NOTE]
> The default table format is raw LaTeX code without table float instructions. Use `--table-format markdown` for Markdown tables or `--table-format console` for aligned plain text.

### Keep the lowest-cost circuits with Git LFS

From `experiments`, select the lowest-cost `search` result for each circuit in
every `outputs/*_agg_est.csv` file:

```bash
python3 select_best.py --dry-run
python3 select_best.py
git add best/
```

The script reads the last CSV column (`total_cost` in the generated files) and
chooses the lowest-cost `search` result for each benchmark, breaking ties by the
smaller FBS size. `run_benchmarks.sh` calls it automatically after cost estimation;
the commands above also allow it to be run separately.
It copies the selected `.lbf` and `.log` files, plus the input circuit, into
`best/<suite>/`, stripping `_<fbs_size>_search` from their names. For example,
`outputs/epfl/adder_5_search.lbf` becomes `best/epfl/adder.lbf`, and its input
becomes `best/epfl/adder.blif`. Each saved circuit has three files; `.fbs` files
are not copied. Input filenames and formats are preserved. The
original inputs, mapping outputs, and CSV files are unchanged. All selected
files are checked before copying anything; missing sources cause an error.
Existing best files are overwritten. To select only one suite,
pass its CSV, for example `python3 select_best.py outputs/epfl_agg_est.csv`.

The files in `experiments/best/` use Git LFS. After installing Git LFS, run `git lfs install --local && git lfs pull --include="experiments/best/**"` from the repository root to download and check them out.

## Results

### EPFL benchmarks

The table summarizes saved runs from `experiments/outputs/epfl_agg_est.csv`,
formatted by `analyse_results.py`.

Each mapper uses the FBS size minimizing estimated evaluation cost: `naive` searches sizes 3–16, 
and `search` searches sizes 2–16. Multi-output
grouping is disabled and negacyclic LUTs up to `2p` are allowed.

`#gates` is the `basic` mapper's bootstrap count at `p=2`, used as the gate-count
baseline; unary linear operations are not included.
`#boots.` shows the selected mapping's bootstrap count followed by its
percentage reduction, for example `256 (-75%)`. The cell is blank when the
reduction rounds to zero.
Cost is the percentage change from the baseline (negative values are savings).
Percentages that round to zero are left blank. When the selected mapping costs
less than the baseline, its FBS size is shown as `p (maximum LUT size)`;
otherwise the FBS-size cell is blank. The average is the arithmetic mean
of the percentages over all 20 circuits. Costs are Concrete estimates, not
measured encrypted-execution times.

| bench | #gates | naive cost | naive #boots. | naive FBS size | search cost | search #boots. | search FBS size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 1020 |  |  |  | -64% | 256 (-75%) | 5 (7) |
| bar | 3336 |  |  |  | -24% | 1664 (-50%) | 7 (10) |
| div | 57247 |  |  |  | -30% | 27644 (-52%) | 5 (10) |
| hyp | 214335 |  |  |  | -39% | 82429 (-62%) | 7 (14) |
| log2 | 32060 |  |  |  | -37% | 14039 (-56%) | 5 (10) |
| max | 2865 | -6% | 1781 (-38%) | 7 (8) | -34% | 905 (-68%) | 9 (15) |
| multiplier | 27062 |  |  |  | -49% | 8720 (-68%) | 7 (14) |
| sin | 5416 |  |  |  | -36% | 2273 (-58%) | 6 (11) |
| sqrt | 24618 |  |  |  | -54% | 7462 (-70%) | 6 (11) |
| square | 18484 |  |  |  | -32% | 8667 (-53%) | 5 (10) |
| arbiter | 11839 | -20% | 6544 (-45%) | 5 (8) | -48% | 4259 (-64%) | 5 (8) |
| cavlc | 693 |  | 455 (-34%) | 7 (8) | -36% | 287 (-59%) | 7 (13) |
| ctrl | 174 | -3% | 109 (-37%) | 7 (8) | -40% | 67 (-61%) | 7 (12) |
| dec | 304 |  |  |  |  |  |  |
| i2c | 1342 | -1% | 874 (-35%) | 7 (8) | -34% | 572 (-57%) | 7 (14) |
| int2float | 260 | -8% | 157 (-40%) | 7 (8) | -49% | 85 (-67%) | 7 (13) |
| mem_ctrl | 46836 | -4% | 28959 (-38%) | 7 (8) | -31% | 20961 (-55%) | 7 (13) |
| priority | 978 |  |  |  | -40% | 388 (-60%) | 6 (11) |
| router | 257 | -9% | 154 (-40%) | 7 (8) | -42% | 94 (-63%) | 7 (14) |
| voter | 13758 |  |  |  | -38% | 5882 (-57%) | 5 (10) |
| avg. |  | -3% | (-15%) |  | -38% | (-58%) |  |
