# Rust mapping policies

The command exposes three mapping policies.

- `basic` maps every Boolean gate to its own linear combination and functional
  bootstrap.
- `naive` composes two-input cones using the original fixed-coefficient rule and
  bootstraps when the resulting encoding is not valid.
- `search` examines coefficient pairs in increasing resulting FBS size and uses
  the smallest valid composition.

The `naive` and `search` policies use the Python mapper's bootstrap boundaries.
Shared cones are retained until composition requires a bootstrap; fanout alone
does not force one. Input cones are ordered by span, then squared coefficient
norm, with the larger cone first. If composition fails, that cone is bootstrapped
and composition is retried. The second input is bootstrapped only if the retry
also fails. Circuit outputs are bootstrapped eagerly in ordinary mode.

Ordinary `search` uses Python's coefficient ranges and minimizes the original
span bound, then the squared multivalue-table energy. Equal scores prefer
descending coefficient pairs; coefficient norm is only used for configured
limits. Cache keys retain the exact input tables and truth table so symmetry
sharing cannot alter these tie choices.

Ordinary linear operations preserve Python's term order and repeated terms when
flattening. Canonical sorting and coefficient aggregation are retained in
multi-output mode, where they enable physical bootstrap sharing.

Unreachable LUT entries are filled uniformly with zero, then uniformly with
one, as in Python. Mixed filling is not used to accept additional candidates.

All three policies support the FBS size, negacyclic-extension, truth-table-size,
and encoding-norm limits exposed by `fbs-map`. `--trials N` runs the selected
policy once in the circuit's encoded topological order and then in `N - 1`
deterministic random topological orders derived from `--seed`. The best mapping
is retained and the total mapping time is reported. `--multioutput-fbs` groups
Boolean outputs that use the same materialized linear bootstrap input. It also
defers shared and circuit-output bootstraps: if a later cone merge only scales a
predecessor's linear form, the predecessor's other functions remain available
as outputs of the merged cone's multi-output FBS.

Generated mappings are verified exhaustively for circuits with at most 16
inputs. Larger circuits use the deterministic sample count selected by
`--verify-samples`.
