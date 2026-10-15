# Rust mapping policies

`fbs-map` exposes the three mapping policies described in [README.md](../README.md).

- `basic` maps each two-input Boolean gate independently to a functional
  bootstrap (FBS). Unary gates use linear operations without bootstrapping.
- `naive` composes gate-input cones with a fixed-radix linear combination,
  without searching coefficients. For independent Boolean inputs, repeated
  composition gives power-of-two weights such as `a + 2*b + 4*c + ...`, up to
  input order. More generally, it combines input encodings `L1` and `L2` as
  `s2*L1 + L2`, where `s2` is the range span of `L2`.
- `search` searches coefficient pairs for a valid composition minimizing a
  span bound on the FBS input encoding. This is a greedy local objective, not a
  guarantee of the smallest final circuit or evaluation cost.

## Bootstrap boundaries

In ordinary `naive` and `search`, shared cones are retained until composition
requires a bootstrap; fanout alone does not force one. Input cones are ordered
by span, then squared coefficient norm, with the larger cone first. If
composition fails, that cone is bootstrapped and composition is retried. The
second input is bootstrapped only if the retry
also fails. Circuit outputs are bootstrapped eagerly in ordinary mode.

## Search ranking and linear forms

Ordinary `search` considers combinations `a*L1 + b*L2`. For input spans `s1`
and `s2`, the inclusive integer coefficient ranges are `a` in `[-s2, s2]` and
`b` in `[0, s1]` when `s1 >= s2`; otherwise, `a` is in `[0, s2]` and `b` is in
`[-s1, s1]`. It minimizes the span bound
`|a|*(s1 - 1) + |b|*(s2 - 1)`, then the squared Euclidean norm of the
multivalue table: `sum_j(mvt[j]^2)`, including repeated table entries. Equal
scores prefer descending coefficient pairs. The coefficient norm
`sum_i(c_i²)` is used separately to order input cones and enforce
`--max-norm2`; it does not break ordinary search ties. Cache keys retain the
exact input tables and truth table so symmetry sharing cannot alter these tie
choices.

Ordinary linear operations preserve term order and repeated terms when
flattening. Canonical sorting and coefficient aggregation are retained in
multi-output mode, where they enable physical bootstrap sharing.

An encoding can skip integer values within its range: for Boolean inputs,
`2*a + 2*b` produces only `0`, `2`, and `4`, so LUT entries at `1` and `3` are
unreachable. Their values do not affect the circuit's Boolean outputs, but
must still satisfy the FBS test-vector constraints. The mapper first fills
all unreachable entries with `0` and checks validity. If that fails, it fills
all of them with `1` and checks again, leaving reachable entries unchanged.
If neither completion is valid, the candidate is rejected. Mixed fillings
(some unreachable entries set to `0`, others to `1`) are not supported.

## Limits and trials

For `naive` and `search`, test vectors can have up to `2p` entries when the
negacyclic-extension conditions are satisfied; `--strict-fbs-size` limits them
to `p`. `--max-tt-size` bounds cone support before forced bootstrapping, and
`--max-norm2 N` limits `sum_i(c_i²)` of candidate cones to `N`. `basic`
normally uses four-entry binary LUTs and does not apply the composition-size
or cone-support limits; its norm limit is checked on the mapped linear forms.

`--trials N` runs the selected policy once in the circuit's encoded topological
order, then in `N - 1` deterministic random topological orders derived from
`--seed`. The best mapping is retained by bootstrap count, then maximum LUT
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

In `naive` and `search`, multi-output mode defers eager circuit-output
bootstraps. Shared cones remain available until composition requires a
bootstrap. This can enable further sharing, but the heuristic is not guaranteed
to find every possible shared encoding.

## Verification

Generated mappings are verified exhaustively for circuits with at most 16
inputs. Larger circuits use deterministic samples, with 1,000 by default, as
selected by `--verify-samples` and `--seed`. These checks simulate Boolean and
linear/LUT operations; they do not perform encrypted execution.
