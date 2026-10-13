# Rust mapping policies

The command exposes three mapping policies.

- `basic` maps every Boolean gate to its own linear combination and functional
  bootstrap.
- `naive` composes two-input cones using the original fixed-coefficient rule and
  bootstraps when the resulting encoding is not valid.
- `search` examines coefficient pairs in increasing resulting FBS size and uses
  the smallest valid composition. If composition fails, it bootstraps the
  predecessor that gives the smaller next cone.

Unreachable LUT entries are filled independently when needed to satisfy the
negacyclic constraints. Uniform zero or one filling is retained when valid.

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
