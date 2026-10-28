# Rust mapping policies

`fbs-map` exposes the two mapping policies described in [README.md](../README.md).

- `basic` uses the same coefficient search as `search` at FBS size 2 and
  bootstraps after every gate composition. Constants and unary gates retain
  search's simplifications.
- `search` searches coefficient pairs for a valid composition minimizing a
  span bound on the FBS input encoding. This is a greedy local objective, not a
  guarantee of the smallest final circuit or evaluation cost.

## Bootstrap boundaries

In ordinary `search`, shared cones are retained until composition
requires a bootstrap; fanout alone does not force one. Input cones are ordered
by span, then squared coefficient norm, with the larger cone first. If
composition fails, that cone is bootstrapped and composition is retried. The
second input is bootstrapped only if the retry
also fails. Circuit outputs are bootstrapped eagerly in ordinary mode.

## Search ranking and linear forms

Ordinary `search` considers primitive combinations `a*L1 + b*L2`, with `a`
in `[0, s2]` and `b` in `[-s1, s1]` for input spans `s1` and `s2`. It
normalizes the merged coefficients, then minimizes their span, the squared
multivalue-table norm `sum_j(mvt[j]^2)`, and the squared coefficient norm.
Remaining ties prefer ascending coefficient pairs. The coefficient norm also
orders input cones and enforces `--max-norm2`. Cache keys retain the exact
input values and truth table.

Ordinary linear operations preserve term order and repeated terms when
flattening. Canonical sorting and coefficient aggregation are retained in
multi-output mode, where they enable physical bootstrap sharing.

An encoding can skip integer values within its range: for Boolean inputs,
`2*a + 2*b` produces only `0`, `2`, and `4`, so LUT entries at `1` and `3` are
unreachable. Their values do not affect the circuit's Boolean outputs, but
must still satisfy the FBS test-vector constraints. The mapper first fills
all unreachable entries with `0` and checks validity. If that fails, it fills
all of them with `1` and checks again, leaving reachable entries unchanged.
If neither uniform completion is valid, missing entries are treated as
wildcards and completed pair by pair to satisfy the antipodal constraint.
This mixed completion takes linear time. The emitted test vector still contains
only concrete `0` and `1` values.

## Limits and trials

For `search`, test vectors can span repeated `2p` periods when the
negacyclic conditions are satisfied; `--fbs-range 2p` limits them to `2p`
and `--fbs-range p` limits them to `p`. `--max-tt-size` bounds cone support before forced bootstrapping, and
`--max-norm2 N` limits `sum_i(c_i²)` of candidate cones to `N`. `basic`
always uses `p=2` regardless of `--fbs-size`, applies the same composition
limits, and bootstraps each gate before continuing. With `--fbs-range p`,
only two-entry tables are allowed, so some binary functions cannot be mapped.

`--trials N` runs the selected policy once in the circuit's encoded topological
order, then in `N` deterministic random topological orders derived from
`--seed`; `--exclude-input-order` skips the input-order run. The best mapping is retained by bootstrap count, then maximum LUT
size, maximum reported norm², and linear-operation count. Reported mapping time
is summed over all trials and excludes parsing, verification, and file output.

## Experimental multi-output mode

`--multioutput-fbs` groups Boolean outputs with the same materialized linear
bootstrap input and counts each group as one FBS. Search prefers encodings
compatible with existing bootstrap inputs, and cones can be realigned to such
encodings before bootstrapping. These zero-additional-FBS choices describe the
mapper's counting model, not a guarantee of zero encrypted-evaluation cost.
Among candidates with the same reuse cost, search minimizes the span bound,
then the squared multivalue-table norm, then the squared coefficient norm;
remaining ties prefer ascending coefficient pairs.

In `search`, multi-output mode defers eager circuit-output
bootstraps. Shared cones remain available until composition requires a
bootstrap. This can enable further sharing, but the heuristic is not guaranteed
to find every possible shared encoding.

## Verification

Generated mappings are verified exhaustively for circuits with at most 16
inputs. Larger circuits use deterministic samples, with 1,000 by default, as
selected by `--verify-samples` and `--seed`. These checks simulate Boolean and
linear/LUT operations; they do not perform encrypted execution.
