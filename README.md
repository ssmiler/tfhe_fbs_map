# Heuristic for mapping Boolean circuits to TFHE functional bootstrapping

This repository contains follow-up improvements to the code accompanying the paper *[A Fast Heuristic for Mapping Boolean Circuits to Functional Bootstrapping](https://eprint.iacr.org/2024/1204)*, including a Rust implementation.

The version used for the paper is tagged `ches_2025`. To check it out, run:

```bash
git checkout ches_2025
```

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

- `basic`: uses `search` at FBS size 2 and bootstraps after every gate composition; constants and unary gates retain search's simplifications.
- `search`: composes gate-input cones by searching coefficient pairs that minimize a span bound on the FBS input encoding (the paper's greedy heuristic).

See [mapping-policy details](docs/heuristics.md) for coefficient selection,
bootstrap boundaries, and limits.

Use `--max-norm2 N` to limit the squared Euclidean norm of the linear-combination
coefficient vector, `sum_i(c_i²)`, to `N`. This limit is separate from search's
multivalue-table tie-breaker.
For `search`, valid negacyclic test vectors can span repeated `2p` periods
by default; `--fbs-range 2p` limits them to `2p`, and `--fbs-range p`
limits them to `p`.
The `basic` policy
always uses `p=2`, with the same negacyclic and strict-size rules. Run
`target/release/fbs-map --help` for all controls.

The mapper runs once with the topological order encoded in the circuit.
`--trials N` adds `N` deterministic random topological orders derived from
`--seed`; `--exclude-input-order` skips the input-order run. The best mapping is
retained and total mapping time is reported.
Random scheduling chooses among the eight earliest ready gates in input order
to preserve locality while exploring different topological orders.

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
`search`, multi-output mode defers eager circuit-output bootstraps;
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

From `experiments`, run the benchmark suites and estimate evaluation costs:

```bash
bash run_benchmarks.sh
```

The script creates a `Makefile` for each benchmark suite (`epfl`, `iscas85`, etc.).
The makefile targets run the Rust mapper with specific parameters.
Each makefile execution writes to a suite log (`outputs/epfl.log` for EPFL benchmarks, etc.).
Runtime depends on the hardware and whether dependencies and inputs are already
available.
The generated `*_agg_est.csv` files are also copied to `experiments/` so the
evaluation-cost summaries can be stored with the repository.


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

See the saved Markdown output in [experiments/results.md](experiments/results.md).

> [!NOTE]
> The default table format is raw LaTeX code without table float instructions. Use `--table-format markdown` for Markdown tables or `--table-format console` for aligned plain text.


## Results

### EPFL benchmarks

The table summarizes saved runs from `experiments/epfl_agg_est.csv`,
formatted by `analyse_results.py`.

The search mapper uses the FBS size minimizing estimated evaluation cost over sizes 2–16. Multi-output
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

| bench | #gates | search cost | search #boots. | search FBS size |
| --- | --- | --- | --- | --- |
| adder | 1020 | -72% | 256 (-75%) | 3 (4) |
| bar | 3336 | -24% | 1664 (-50%) | 7 (10) |
| div | 57247 | -30% | 25133 (-56%) | 7 (14) |
| hyp | 214335 | -37% | 93843 (-56%) | 5 (10) |
| log2 | 32060 | -38% | 13807 (-57%) | 5 (10) |
| max | 2865 | -53% | 873 (-70%) | 7 (13) |
| multiplier | 27062 | -50% | 9113 (-66%) | 5 (16) |
| sin | 5416 | -36% | 2412 (-55%) | 5 (10) |
| sqrt | 24618 | -54% | 7405 (-70%) | 6 (11) |
| square | 18484 | -35% | 8260 (-55%) | 5 (10) |
| arbiter | 11839 | -48% | 4259 (-64%) | 5 (8) |
| cavlc | 693 | -36% | 284 (-59%) | 7 (13) |
| ctrl | 174 | -47% | 56 (-68%) | 8 (13) |
| dec | 304 |  |  |  |
| i2c | 1342 | -35% | 562 (-58%) | 7 (13) |
| int2float | 260 | -50% | 84 (-68%) | 7 (13) |
| mem_ctrl | 46836 | -30% | 22753 (-51%) | 5 (10) |
| priority | 978 | -40% | 387 (-60%) | 6 (11) |
| router | 257 | -42% | 94 (-63%) | 7 (14) |
| voter | 13758 | -44% | 4872 (-65%) | 7 (14) |
| avg. |  | -40% | (-58%) |  |

### Comparison with Bon et al.

Bon, Pointcheval, and Rivain, [*Optimized Homomorphic Evaluation of Boolean Functions*](https://eprint.iacr.org/2023/1589), IACR ePrint 2023/1589.

The following comparison uses the saved search results in
`experiments/generated_agg_est.csv` and the Bon et al. values in
`analyse_results.py`. Our FBS size is shown as execution size (maximum LUT
size). Speedup is the ratio of estimated Bon et al. cost to estimated search
cost; it is not a measurement of encrypted execution time.

| Circuit | Bon et al. #FBS | Bon et al. FBS size | Search #FBS | Search FBS size | Estimated speedup |
| --- | --- | --- | --- | --- | --- |
| SIMON | 1 | 9 | 1 | 2 (9) | 1.57× |
| ASCON | 5 | 17 | 10 | 2 (7) | 1.25× |
| AES s-box | 36 | 11 | 48 | 2 (11) | 1.62× |
