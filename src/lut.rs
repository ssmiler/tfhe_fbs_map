use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LutSignal {
    Const(i64),
    Node(usize),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LutNode {
    Input {
        name: String,
    },
    Linear {
        terms: Vec<(i64, usize)>,
        constant: i64,
    },
    Bootstrap {
        input: usize,
        table: Vec<u8>,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MappingStats {
    pub input_count: usize,
    pub linear_count: usize,
    pub bootstrap_count: usize,
    pub max_lut_size: usize,
    pub max_norm2: u128,
    pub output_count: usize,
}

impl MappingStats {
    pub fn quality_key(&self) -> (usize, usize, u128, usize) {
        (
            self.bootstrap_count,
            self.max_lut_size,
            self.max_norm2,
            self.linear_count,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct LutCircuit {
    pub nodes: Vec<LutNode>,
    pub outputs: Vec<(String, LutSignal)>,
    multioutput_fbs: bool,
    bounds: Vec<(i64, i64)>,
    norm2: Vec<u128>,
    cache: HashMap<LutNode, usize>,
    bootstrap_inputs: HashSet<usize>,
    bootstrap_forms: HashMap<Vec<usize>, Vec<Vec<i64>>>,
}

pub(crate) struct BootstrapLookup {
    basis: Vec<(usize, i64, i64)>,
    constants: (i64, i64),
    key: LutNode,
}

impl BootstrapLookup {
    pub(crate) fn has_bootstrap(
        &mut self,
        circuit: &LutCircuit,
        scales: (i64, i64),
        constant: i64,
    ) -> bool {
        let LutNode::Linear {
            terms,
            constant: offset,
        } = &mut self.key
        else {
            unreachable!()
        };
        terms.clear();
        for &(node, left, right) in &self.basis {
            let coefficient = scales.0 * left + scales.1 * right;
            if coefficient != 0 {
                terms.push((coefficient, node));
            }
        }
        *offset = constant + scales.0 * self.constants.0 + scales.1 * self.constants.1;
        let input = if terms.len() == 1 && terms[0].0 == 1 && *offset == 0 {
            Some(terms[0].1)
        } else {
            circuit.cache.get(&self.key).copied()
        };
        input.is_some_and(|input| circuit.bootstrap_inputs.contains(&input))
    }
}

impl LutCircuit {
    pub fn set_multioutput_fbs(&mut self, enabled: bool) {
        self.multioutput_fbs = enabled;
    }

    pub fn input(&mut self, name: &str) -> LutSignal {
        self.intern(
            LutNode::Input {
                name: name.to_owned(),
            },
            (0, 1),
            1,
        )
    }

    pub fn linear(&mut self, terms: &[(i64, LutSignal)], mut constant: i64) -> LutSignal {
        let terms = if self.multioutput_fbs {
            let (mut flattened, flattened_constant) = self.flatten_linear(terms, constant);
            constant = flattened_constant;
            flattened.retain(|_, coefficient| *coefficient != 0);
            if flattened.is_empty() {
                return LutSignal::Const(constant);
            }
            if flattened.len() == 1 && constant == 0 {
                let (&id, &coefficient) = flattened.first_key_value().unwrap();
                if coefficient == 1 {
                    return LutSignal::Node(id);
                }
            }

            flattened
                .into_iter()
                .map(|(id, coefficient)| (coefficient, id))
                .collect()
        } else {
            let (flattened, flattened_constant) = self.flatten_linear_ordered(terms, constant);
            constant = flattened_constant;
            if flattened.is_empty() {
                return LutSignal::Const(constant);
            }
            flattened
        };
        let mut minimum = constant;
        let mut maximum = constant;
        let mut norm2 = 0u128;
        for (coefficient, id) in &terms {
            let (input_minimum, input_maximum) = self.bounds[*id];
            if *coefficient >= 0 {
                minimum = minimum.saturating_add(coefficient.saturating_mul(input_minimum));
                maximum = maximum.saturating_add(coefficient.saturating_mul(input_maximum));
            } else {
                minimum = minimum.saturating_add(coefficient.saturating_mul(input_maximum));
                maximum = maximum.saturating_add(coefficient.saturating_mul(input_minimum));
            }
            norm2 = norm2.saturating_add(
                coefficient.unsigned_abs() as u128
                    * coefficient.unsigned_abs() as u128
                    * self.norm2[*id],
            );
        }
        self.intern(
            LutNode::Linear { terms, constant },
            (minimum, maximum),
            norm2,
        )
    }

    fn flatten_linear(
        &self,
        terms: &[(i64, LutSignal)],
        mut constant: i64,
    ) -> (BTreeMap<usize, i64>, i64) {
        let mut flattened = BTreeMap::new();
        for (coefficient, signal) in terms {
            if *coefficient == 0 {
                continue;
            }
            match signal {
                LutSignal::Const(value) => constant += coefficient * value,
                LutSignal::Node(id) => match &self.nodes[*id] {
                    LutNode::Linear {
                        terms,
                        constant: inner_constant,
                    } => {
                        constant += coefficient * inner_constant;
                        for (inner_coefficient, input) in terms {
                            *flattened.entry(*input).or_default() +=
                                coefficient * inner_coefficient;
                        }
                    }
                    _ => *flattened.entry(*id).or_default() += coefficient,
                },
            }
        }
        (flattened, constant)
    }

    fn flatten_linear_ordered(
        &self,
        terms: &[(i64, LutSignal)],
        mut constant: i64,
    ) -> (Vec<(i64, usize)>, i64) {
        let mut flattened = Vec::new();
        for &(coefficient, signal) in terms {
            match signal {
                LutSignal::Const(value) => constant += coefficient * value,
                LutSignal::Node(id) => match &self.nodes[id] {
                    LutNode::Linear {
                        terms,
                        constant: inner_constant,
                    } => {
                        constant += coefficient * inner_constant;
                        flattened.extend(
                            terms
                                .iter()
                                .map(|&(inner, input)| (coefficient * inner, input)),
                        );
                    }
                    _ => flattened.push((coefficient, id)),
                },
            }
        }
        (flattened, constant)
    }

    pub(crate) fn prepare_bootstrap_lookup(
        &self,
        basis: impl IntoIterator<Item = (usize, i64, i64)>,
    ) -> Option<BootstrapLookup> {
        let basis: Vec<_> = basis.into_iter().collect();
        if basis.iter().any(|(node, _, _)| *node >= self.nodes.len()) {
            return None;
        }
        let left_terms: Vec<_> = basis
            .iter()
            .map(|&(node, left, _)| (left, LutSignal::Node(node)))
            .collect();
        let right_terms: Vec<_> = basis
            .iter()
            .map(|&(node, _, right)| (right, LutSignal::Node(node)))
            .collect();
        let (left, left_constant) = self.flatten_linear(&left_terms, 0);
        let (right, right_constant) = self.flatten_linear(&right_terms, 0);
        let mut aligned: BTreeMap<_, _> = left
            .into_iter()
            .map(|(node, coefficient)| (node, (coefficient, 0)))
            .collect();
        for (node, coefficient) in right {
            aligned.entry(node).or_default().1 = coefficient;
        }
        let basis: Vec<_> = aligned
            .into_iter()
            .map(|(node, (left, right))| (node, left, right))
            .collect();
        Some(BootstrapLookup {
            key: LutNode::Linear {
                terms: Vec::with_capacity(basis.len()),
                constant: 0,
            },
            basis,
            constants: (left_constant, right_constant),
        })
    }

    pub(crate) fn reusable_linear_coefficients(&self, support: &[usize]) -> Vec<Vec<i64>> {
        let mut key = support.to_vec();
        key.sort_unstable();
        let Some(forms) = self.bootstrap_forms.get(&key) else {
            return Vec::new();
        };
        let positions: HashMap<_, _> = key
            .iter()
            .enumerate()
            .map(|(position, node)| (*node, position))
            .collect();
        forms
            .iter()
            .map(|form| support.iter().map(|node| form[positions[node]]).collect())
            .collect()
    }

    fn record_bootstrap_input(&mut self, input: usize) {
        self.bootstrap_inputs.insert(input);
        let (support, coefficients) = match &self.nodes[input] {
            LutNode::Linear { terms, .. } => (
                terms.iter().map(|(_, node)| *node).collect::<Vec<_>>(),
                terms
                    .iter()
                    .map(|(coefficient, _)| *coefficient)
                    .collect::<Vec<_>>(),
            ),
            _ => (vec![input], vec![1]),
        };
        let forms = self.bootstrap_forms.entry(support).or_default();
        if !forms.contains(&coefficients) {
            forms.push(coefficients);
        }
    }

    pub fn bootstrap(&mut self, signal: LutSignal, table: Vec<u8>) -> Result<LutSignal, String> {
        match signal {
            LutSignal::Const(value) => {
                let index = usize::try_from(value)
                    .map_err(|_| format!("negative bootstrap input {value}"))?;
                table
                    .get(index)
                    .copied()
                    .map(|v| LutSignal::Const(i64::from(v)))
                    .ok_or_else(|| {
                        format!(
                            "bootstrap input {value} outside table of size {}",
                            table.len()
                        )
                    })
            }
            LutSignal::Node(input) => {
                let (minimum, maximum) = self.bounds[input];
                if minimum != 0 || maximum < 0 || maximum as usize + 1 != table.len() {
                    return Err(format!(
                        "bootstrap input range [{minimum}, {maximum}] does not match table size {}",
                        table.len()
                    ));
                }
                if table.iter().any(|value| *value > 1) {
                    return Err("only Boolean bootstrap outputs are supported".to_owned());
                }
                self.record_bootstrap_input(input);
                Ok(self.intern(LutNode::Bootstrap { input, table }, (0, 1), 1))
            }
        }
    }

    pub fn add_output(&mut self, name: String, signal: LutSignal) {
        self.outputs.push((name, signal));
    }

    pub fn bounds(&self, signal: LutSignal) -> (i64, i64) {
        match signal {
            LutSignal::Const(value) => (value, value),
            LutSignal::Node(id) => self.bounds[id],
        }
    }

    pub fn signal_name(&self, signal: LutSignal) -> String {
        match signal {
            LutSignal::Const(value) => value.to_string(),
            LutSignal::Node(id) => self.node_name(id),
        }
    }

    pub fn node_name(&self, id: usize) -> String {
        match &self.nodes[id] {
            LutNode::Input { name } => name.clone(),
            _ => format!("m{}", id + 1),
        }
    }

    fn intern(&mut self, node: LutNode, bounds: (i64, i64), norm2: u128) -> LutSignal {
        if let Some(id) = self.cache.get(&node) {
            return LutSignal::Node(*id);
        }
        let id = self.nodes.len();
        self.cache.insert(node.clone(), id);
        self.nodes.push(node);
        self.bounds.push(bounds);
        self.norm2.push(norm2);
        LutSignal::Node(id)
    }

    pub fn evaluate(&self, inputs: &HashMap<String, bool>) -> Result<Vec<(String, bool)>, String> {
        let mut values = vec![0i64; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            values[id] = match node {
                LutNode::Input { name } => i64::from(
                    *inputs
                        .get(name)
                        .ok_or_else(|| format!("missing input '{name}'"))?,
                ),
                LutNode::Linear { terms, constant } => {
                    terms.iter().fold(*constant, |sum, (coefficient, input)| {
                        sum + coefficient * values[*input]
                    })
                }
                LutNode::Bootstrap { input, table } => {
                    let value = values[*input];
                    let index = usize::try_from(value)
                        .map_err(|_| format!("negative bootstrap input {value} at node {id}"))?;
                    i64::from(*table.get(index).ok_or_else(|| {
                        format!("bootstrap input {value} outside table at node {id}")
                    })?)
                }
            };
        }
        self.outputs
            .iter()
            .map(|(name, signal)| {
                let value = match signal {
                    LutSignal::Const(value) => *value,
                    LutSignal::Node(id) => values[*id],
                };
                match value {
                    0 => Ok((name.clone(), false)),
                    1 => Ok((name.clone(), true)),
                    _ => Err(format!("output '{name}' is not Boolean: {value}")),
                }
            })
            .collect()
    }

    pub fn prune(&mut self) {
        let mut live = HashSet::new();
        let mut stack: Vec<_> = self
            .outputs
            .iter()
            .filter_map(|(_, signal)| {
                if let LutSignal::Node(id) = signal {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        while let Some(id) = stack.pop() {
            if !live.insert(id) {
                continue;
            }
            match &self.nodes[id] {
                LutNode::Input { .. } => {}
                LutNode::Linear { terms, .. } => {
                    stack.extend(terms.iter().map(|(_, input)| *input))
                }
                LutNode::Bootstrap { input, .. } => stack.push(*input),
            }
        }

        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut nodes = Vec::with_capacity(live.len());
        let mut bounds = Vec::with_capacity(live.len());
        let mut norm2 = Vec::with_capacity(live.len());
        for (old, remapped) in remap.iter_mut().enumerate() {
            if live.contains(&old) {
                *remapped = nodes.len();
                nodes.push(self.nodes[old].clone());
                bounds.push(self.bounds[old]);
                norm2.push(self.norm2[old]);
            }
        }
        for node in &mut nodes {
            match node {
                LutNode::Input { .. } => {}
                LutNode::Linear { terms, .. } => {
                    for (_, input) in terms {
                        *input = remap[*input];
                    }
                }
                LutNode::Bootstrap { input, .. } => *input = remap[*input],
            }
        }
        for (_, signal) in &mut self.outputs {
            if let LutSignal::Node(id) = signal {
                *id = remap[*id];
            }
        }
        self.nodes = nodes;
        self.bounds = bounds;
        self.norm2 = norm2;
        self.cache = self
            .nodes
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, node)| (node, id))
            .collect();
        let bootstrap_inputs: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|node| {
                if let LutNode::Bootstrap { input, .. } = node {
                    Some(*input)
                } else {
                    None
                }
            })
            .collect();
        self.bootstrap_inputs.clear();
        self.bootstrap_forms.clear();
        for input in bootstrap_inputs {
            self.record_bootstrap_input(input);
        }
    }

    pub fn stats(&self) -> MappingStats {
        let mut stats = MappingStats {
            output_count: self.outputs.len(),
            ..MappingStats::default()
        };
        for (id, node) in self.nodes.iter().enumerate() {
            match node {
                LutNode::Input { .. } => stats.input_count += 1,
                LutNode::Linear { .. } => stats.linear_count += 1,
                LutNode::Bootstrap { table, .. } => {
                    stats.bootstrap_count += 1;
                    stats.max_lut_size = stats.max_lut_size.max(table.len());
                }
            }
            stats.max_norm2 = stats.max_norm2.max(self.norm2[id]);
        }
        if self.multioutput_fbs {
            // A programmable bootstrap can return several Boolean functions of
            // the same linear input. Count each such group as one physical FBS.
            stats.bootstrap_count = self.bootstrap_groups().len();
        }
        stats
    }

    fn bootstrap_groups(&self) -> HashMap<usize, Vec<usize>> {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut input_leaders = HashMap::new();
        for (id, node) in self.nodes.iter().enumerate() {
            if let LutNode::Bootstrap { input, .. } = node {
                let leader = if self.multioutput_fbs {
                    *input_leaders.entry(*input).or_insert(id)
                } else {
                    id
                };
                groups.entry(leader).or_default().push(id);
            }
        }
        groups
    }

    pub fn to_fbs(&self) -> String {
        let mut output = String::new();
        let bootstrap_groups = self.bootstrap_groups();
        for (id, node) in self.nodes.iter().enumerate() {
            match node {
                LutNode::Input { .. } => {}
                LutNode::Linear { terms, constant } => {
                    let expression = terms
                        .iter()
                        .map(|(coefficient, input)| {
                            format!("{coefficient} * {}", self.node_name(*input))
                        })
                        .collect::<Vec<_>>()
                        .join(" + ");
                    let suffix = if *constant == 0 {
                        String::new()
                    } else {
                        format!(" + {constant}")
                    };
                    let _ = writeln!(output, "{} = {expression}{suffix}", self.node_name(id));
                }
                LutNode::Bootstrap { input, .. } => {
                    let Some(group) = bootstrap_groups.get(&id) else {
                        continue;
                    };
                    let output_names = group
                        .iter()
                        .map(|output| self.node_name(*output))
                        .collect::<Vec<_>>()
                        .join(",");
                    let tables = group
                        .iter()
                        .map(|output| {
                            let LutNode::Bootstrap { table, .. } = &self.nodes[*output] else {
                                unreachable!()
                            };
                            let values = table
                                .iter()
                                .map(u8::to_string)
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("[{values}]")
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let _ = writeln!(
                        output,
                        "{output_names} = Bootstrap({}, {tables})",
                        self.node_name(*input)
                    );
                }
            }
        }
        for (name, signal) in &self.outputs {
            let _ = writeln!(output, "Output {name} = {}", self.signal_name(*signal));
        }
        output
    }

    pub fn to_lbf(&self) -> String {
        let mut output = String::new();
        let bootstrap_groups = self.bootstrap_groups();
        let inputs = self
            .nodes
            .iter()
            .filter_map(|node| {
                if let LutNode::Input { name } = node {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let outputs = self
            .outputs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(output, ".inputs {inputs}");
        let _ = writeln!(output, ".outputs {outputs}");
        for (id, node) in self.nodes.iter().enumerate() {
            match node {
                LutNode::Input { .. } => {}
                LutNode::Linear { terms, constant } => {
                    let names = terms
                        .iter()
                        .map(|(_, input)| self.node_name(*input))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let coefficients = terms
                        .iter()
                        .map(|(coefficient, _)| coefficient.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let _ = writeln!(output, ".lincomb {names} {}", self.node_name(id));
                    if *constant == 0 {
                        let _ = writeln!(output, "{coefficients}");
                    } else {
                        let _ = writeln!(output, "{coefficients} {constant}");
                    }
                }
                LutNode::Bootstrap { input, .. } => {
                    let Some(group) = bootstrap_groups.get(&id) else {
                        continue;
                    };
                    let output_names = group
                        .iter()
                        .map(|output| self.node_name(*output))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let _ = writeln!(
                        output,
                        ".bootstrap {} {output_names}",
                        self.node_name(*input)
                    );
                    for output_node in group {
                        let LutNode::Bootstrap { table, .. } = &self.nodes[*output_node] else {
                            unreachable!()
                        };
                        let values = table.iter().map(u8::to_string).collect::<String>();
                        let _ = writeln!(output, "{values}");
                    }
                }
            }
        }
        for (name, signal) in &self.outputs {
            match signal {
                LutSignal::Node(id) => {
                    let _ = writeln!(output, ".lincomb {} {name}", self.node_name(*id));
                    let _ = writeln!(output, "1");
                }
                LutSignal::Const(value) => {
                    let _ = writeln!(output, ".lincomb {name}");
                    let _ = writeln!(output, "{value}");
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_and_canonicalizes_linear_nodes() {
        let mut circuit = LutCircuit::default();
        circuit.set_multioutput_fbs(true);
        let a = circuit.input("a");
        let b = circuit.input("b");
        let x = circuit.linear(&[(2, a), (1, b)], 0);
        let y = circuit.linear(&[(3, x), (-1, b)], 2);
        let z = circuit.linear(&[(2, b), (6, a)], 2);
        assert_eq!(y, z);
        assert_eq!(circuit.bounds(y), (2, 10));
    }

    #[test]
    fn prepared_bootstrap_lookup_matches_linear_interning() {
        let mut circuit = LutCircuit::default();
        circuit.set_multioutput_fbs(true);
        let a = circuit.input("a");
        let b = circuit.input("b");
        let x = circuit.linear(&[(2, a), (-1, b)], 3);
        let y = circuit.linear(&[(-1, a), (3, b)], -2);
        let LutSignal::Node(x_id) = x else {
            unreachable!()
        };
        let LutSignal::Node(y_id) = y else {
            unreachable!()
        };
        for signal in [a, x, y, circuit.linear(&[(2, x), (-1, y)], 1)] {
            let (minimum, maximum) = circuit.bounds(signal);
            let normalized = circuit.linear(&[(1, signal)], -minimum);
            circuit
                .bootstrap(normalized, vec![0; (maximum - minimum + 1) as usize])
                .unwrap();
        }
        circuit.bootstrap(a, vec![0, 1]).unwrap();
        let basis = [(y_id, 0, 1), (x_id, 1, 0)];
        let mut lookup = circuit.prepare_bootstrap_lookup(basis).unwrap();
        for left in -3..=3 {
            for right in -3..=3 {
                for constant in -16..=16 {
                    let mut reference = circuit.clone();
                    let input = reference.linear(&[(left, x), (right, y)], constant);
                    let expected = matches!(input, LutSignal::Node(node) if circuit.bootstrap_inputs.contains(&node));
                    assert_eq!(
                        lookup.has_bootstrap(&circuit, (left, right), constant),
                        expected
                    );
                }
            }
        }
        assert!(
            circuit
                .prepare_bootstrap_lookup([(usize::MAX, 0, 0)])
                .is_none()
        );
        let mut identity = circuit.prepare_bootstrap_lookup([(0, 1, 0)]).unwrap();
        assert!(identity.has_bootstrap(&circuit, (1, 0), 0));
        let mut empty = circuit.prepare_bootstrap_lookup([]).unwrap();
        assert!(!empty.has_bootstrap(&circuit, (1, 1), 0));
        let mut fresh = LutCircuit::default();
        let a = fresh.input("a");
        let mut lookup = fresh.prepare_bootstrap_lookup([(0, 1, 0)]).unwrap();
        assert!(!lookup.has_bootstrap(&fresh, (1, 0), 0));
        fresh.bootstrap(a, vec![0, 1]).unwrap();
        assert!(lookup.has_bootstrap(&fresh, (1, 0), 0));
    }

    #[test]
    fn prunes_dangling_nodes() {
        let mut circuit = LutCircuit::default();
        let a = circuit.input("a");
        let b = circuit.input("b");
        let _dead = circuit.linear(&[(1, b)], 1);
        circuit.add_output("a".into(), a);
        circuit.prune();
        assert_eq!(circuit.nodes, vec![LutNode::Input { name: "a".into() }]);
    }

    #[test]
    fn groups_bootstraps_with_a_common_linear_input() {
        let mut circuit = LutCircuit::default();
        let a = circuit.input("a");
        let b = circuit.input("b");
        let cin = circuit.input("cin");
        let sum = circuit.linear(&[(1, a), (1, b), (1, cin)], 0);
        let parity = circuit
            .bootstrap(sum, vec![0, 1, 0, 1])
            .expect("valid parity bootstrap");
        let carry = circuit
            .bootstrap(sum, vec![0, 0, 1, 1])
            .expect("valid carry bootstrap");
        circuit.add_output("out".into(), parity);
        circuit.add_output("cout".into(), carry);

        assert_eq!(circuit.stats().bootstrap_count, 2);
        assert!(
            circuit
                .to_lbf()
                .contains(".bootstrap m4 m5\n0101\n.bootstrap m4 m6\n0011")
        );
        circuit.set_multioutput_fbs(true);
        assert_eq!(circuit.stats().bootstrap_count, 1);
        assert_eq!(
            circuit.to_fbs(),
            "m4 = 1 * a + 1 * b + 1 * cin\n\
             m5,m6 = Bootstrap(m4, [0, 1, 0, 1], [0, 0, 1, 1])\n\
             Output out = m5\n\
             Output cout = m6\n"
        );
        assert_eq!(
            circuit.to_lbf(),
            ".inputs a b cin\n\
             .outputs out cout\n\
             .lincomb a b cin m4\n\
             1 1 1\n\
             .bootstrap m4 m5 m6\n\
             0101\n\
             0011\n\
             .lincomb m5 out\n\
             1\n\
             .lincomb m6 cout\n\
             1\n"
        );
    }
}
