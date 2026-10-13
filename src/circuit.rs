use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Signal {
    Const(bool),
    Node(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BitNode {
    Input {
        name: String,
    },
    Lut {
        name: String,
        inputs: Vec<Signal>,
        table: Vec<u8>,
    },
}

impl BitNode {
    pub fn name(&self) -> &str {
        match self {
            Self::Input { name } | Self::Lut { name, .. } => name,
        }
    }

    pub fn inputs(&self) -> &[Signal] {
        match self {
            Self::Input { .. } => &[],
            Self::Lut { inputs, .. } => inputs,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BitCircuit {
    pub nodes: Vec<BitNode>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<(String, Signal)>,
}

impl BitCircuit {
    pub fn parse_file(path: impl AsRef<Path>, input_type: &str) -> Result<Self, String> {
        let path = path.as_ref();
        let source =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let input_type = if input_type == "auto" {
            match path.extension().and_then(|extension| extension.to_str()) {
                Some(extension) if extension.eq_ignore_ascii_case("bench") => "bench",
                _ => "blif",
            }
        } else {
            input_type
        };
        let circuit = match input_type {
            "blif" => Self::parse_blif(&source),
            "bench" => Self::parse_bench(&source),
            "bristol" => Self::parse_bristol(&source),
            value => Err(format!("unknown input type '{value}'")),
        };
        circuit.map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn parse_blif_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let source =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse_blif(&source).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse_blif(source: &str) -> Result<Self, String> {
        let lines = logical_lines(source);
        let mut circuit = Self::default();
        let mut wires: HashMap<String, Signal> = HashMap::from([
            ("0".to_owned(), Signal::Const(false)),
            ("1".to_owned(), Signal::Const(true)),
        ]);
        let mut output_names = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            let line = &lines[i];
            let words: Vec<_> = line.split_whitespace().collect();
            if words.is_empty() {
                i += 1;
                continue;
            }
            match words[0] {
                ".model" | ".end" => i += 1,
                ".inputs" => {
                    for name in &words[1..] {
                        if wires.contains_key(*name) {
                            return Err(format!("duplicate input '{name}'"));
                        }
                        let id = circuit.nodes.len();
                        circuit.nodes.push(BitNode::Input {
                            name: (*name).to_owned(),
                        });
                        circuit.inputs.push(id);
                        wires.insert((*name).to_owned(), Signal::Node(id));
                    }
                    i += 1;
                }
                ".outputs" => {
                    output_names.extend(words[1..].iter().map(|s| (*s).to_owned()));
                    i += 1;
                }
                ".names" => {
                    if words.len() < 2 {
                        return Err(".names requires an output".to_owned());
                    }
                    let output = words[words.len() - 1].to_owned();
                    if wires.contains_key(&output) {
                        return Err(format!("duplicate wire '{output}'"));
                    }
                    let input_names = &words[1..words.len() - 1];
                    if input_names.len() > 2 {
                        return Err(format!(
                            "gate '{output}' has {} inputs; map the circuit to 2-input gates first",
                            input_names.len()
                        ));
                    }
                    let inputs: Vec<_> = input_names
                        .iter()
                        .map(|name| {
                            wires.get(*name).copied().ok_or_else(|| {
                                format!("gate '{output}' references undefined wire '{name}'")
                            })
                        })
                        .collect::<Result<_, _>>()?;

                    i += 1;
                    let mut covers = Vec::new();
                    while i < lines.len() && !lines[i].starts_with('.') {
                        if !lines[i].is_empty() {
                            covers.push(lines[i].clone());
                        }
                        i += 1;
                    }
                    let table = parse_cover(input_names.len(), &covers)
                        .map_err(|e| format!("gate '{output}': {e}"))?;

                    let signal = if input_names.is_empty() {
                        Signal::Const(table[0] != 0)
                    } else {
                        let id = circuit.nodes.len();
                        circuit.nodes.push(BitNode::Lut {
                            name: output.clone(),
                            inputs,
                            table,
                        });
                        Signal::Node(id)
                    };
                    wires.insert(output, signal);
                }
                directive if directive.starts_with('.') => {
                    return Err(format!("unsupported BLIF directive '{directive}'"));
                }
                _ => return Err(format!("unexpected line '{line}'")),
            }
        }

        for name in output_names {
            let signal = wires
                .get(&name)
                .copied()
                .ok_or_else(|| format!("output '{name}' is undefined"))?;
            circuit.outputs.push((name, signal));
        }
        if circuit.outputs.is_empty() {
            return Err("circuit has no outputs".to_owned());
        }
        circuit.validate_topological()?;
        Ok(circuit)
    }

    pub fn parse_bench_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let source =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse_bench(&source).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse_bench(source: &str) -> Result<Self, String> {
        let lines = logical_lines(source);
        let mut circuit = Self::default();
        let mut wires = HashMap::<String, Signal>::from([
            ("FALSE".to_owned(), Signal::Const(false)),
            ("TRUE".to_owned(), Signal::Const(true)),
        ]);
        let mut output_names = Vec::new();

        for line in lines {
            if let Some((output, expression)) = line.split_once('=') {
                let output = output.trim();
                if output.is_empty() {
                    return Err(format!("gate has an empty output in '{line}'"));
                }
                if wires.contains_key(output) {
                    return Err(format!("duplicate wire '{output}'"));
                }
                let (operation, input_names) = parse_bench_invocation(expression)?;
                let inputs = input_names
                    .iter()
                    .map(|name| {
                        wires.get(*name).copied().ok_or_else(|| {
                            format!("gate '{output}' references undefined wire '{name}'")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let signal = lower_bench_gate(&mut circuit, output, &operation, &inputs)?;
                wires.insert(output.to_owned(), signal);
                continue;
            }

            let (declaration, names) = parse_bench_invocation(&line)?;
            if names.len() != 1 {
                return Err(format!("{declaration} requires exactly one wire name"));
            }
            let name = names[0];
            match declaration.as_str() {
                "INPUT" => {
                    if wires.contains_key(name) {
                        return Err(format!("duplicate input '{name}'"));
                    }
                    let id = circuit.nodes.len();
                    circuit.nodes.push(BitNode::Input {
                        name: name.to_owned(),
                    });
                    circuit.inputs.push(id);
                    wires.insert(name.to_owned(), Signal::Node(id));
                }
                "OUTPUT" => output_names.push(name.to_owned()),
                _ => return Err(format!("unexpected BENCH declaration '{declaration}'")),
            }
        }

        for name in output_names {
            let signal = wires
                .get(&name)
                .copied()
                .ok_or_else(|| format!("output '{name}' is undefined"))?;
            circuit.outputs.push((name, signal));
        }
        if circuit.outputs.is_empty() {
            return Err("circuit has no outputs".to_owned());
        }
        circuit.validate_topological()?;
        Ok(circuit)
    }

    pub fn parse_bristol_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let source =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse_bristol(&source).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse_bristol(source: &str) -> Result<Self, String> {
        let lines: Vec<_> = source
            .lines()
            .map(|line| line.split('#').next().unwrap_or("").trim())
            .filter(|line| !line.is_empty())
            .collect();
        if lines.len() < 3 {
            return Err("Bristol circuit is missing its three-line header".to_owned());
        }
        let first = parse_usize_words(lines[0])?;
        if first.len() != 2 {
            return Err("Bristol first line must contain gate and wire counts".to_owned());
        }
        let gate_count = first[0];
        let wire_count = first[1];
        let input_header = parse_usize_words(lines[1])?;
        let output_header = parse_usize_words(lines[2])?;
        let input_count = grouped_wire_count(&input_header, "input")?;
        let output_count = grouped_wire_count(&output_header, "output")?;
        if input_count + output_count > wire_count {
            return Err("Bristol input/output counts exceed the wire count".to_owned());
        }

        let mut circuit = Self::default();
        let mut wires = vec![None; wire_count];
        for (wire, slot) in wires.iter_mut().enumerate().take(input_count) {
            let id = circuit.nodes.len();
            circuit.nodes.push(BitNode::Input {
                name: format!("i_{wire}"),
            });
            circuit.inputs.push(id);
            *slot = Some(Signal::Node(id));
        }

        let mut parsed_gates = 0usize;
        for line in &lines[3..] {
            let words: Vec<_> = line.split_whitespace().collect();
            if words.len() < 4 {
                return Err(format!("invalid Bristol gate row '{line}'"));
            }
            let input_arity: usize = words[0]
                .parse()
                .map_err(|_| format!("invalid Bristol gate input count in '{line}'"))?;
            let output_arity: usize = words[1]
                .parse()
                .map_err(|_| format!("invalid Bristol gate output count in '{line}'"))?;
            let expected = 2 + input_arity + output_arity + 1;
            if words.len() != expected {
                return Err(format!("Bristol gate row has wrong width: '{line}'"));
            }
            if output_arity != 1 {
                return Err(format!(
                    "Bristol gate '{line}' has {output_arity} outputs; only one-output gates are supported"
                ));
            }
            let input_wires: Vec<_> = words[2..2 + input_arity]
                .iter()
                .map(|word| {
                    word.parse::<usize>()
                        .map_err(|_| format!("invalid wire '{word}' in '{line}'"))
                })
                .collect::<Result<_, _>>()?;
            let output_wire: usize = words[2 + input_arity]
                .parse()
                .map_err(|_| format!("invalid output wire in '{line}'"))?;
            if output_wire >= wire_count {
                return Err(format!(
                    "output wire {output_wire} is outside 0..{wire_count}"
                ));
            }
            let inputs: Vec<_> = input_wires
                .iter()
                .map(|wire| {
                    wires
                        .get(*wire)
                        .copied()
                        .flatten()
                        .ok_or_else(|| format!("gate references undefined wire {wire}"))
                })
                .collect::<Result<_, _>>()?;
            let operation = words[expected - 1].to_ascii_uppercase();
            let value = match operation.as_str() {
                "EQ" | "EQW" | "BUF" => {
                    if inputs.len() != 1 {
                        return Err(format!("{operation} requires one input"));
                    }
                    inputs[0]
                }
                _ => {
                    let table = bristol_truth_table(&operation, inputs.len())?;
                    let id = circuit.nodes.len();
                    circuit.nodes.push(BitNode::Lut {
                        name: format!("w_{output_wire}"),
                        inputs,
                        table,
                    });
                    Signal::Node(id)
                }
            };
            if wires[output_wire].replace(value).is_some() {
                return Err(format!("wire {output_wire} is assigned more than once"));
            }
            parsed_gates += 1;
        }
        if parsed_gates != gate_count {
            return Err(format!(
                "Bristol header declares {gate_count} gates, found {parsed_gates}"
            ));
        }
        for (wire, signal) in wires.iter().enumerate().skip(wire_count - output_count) {
            let signal = signal.ok_or_else(|| format!("output wire {wire} is undefined"))?;
            circuit.outputs.push((wire.to_string(), signal));
        }
        circuit.validate_topological()?;
        Ok(circuit)
    }

    pub fn validate_topological(&self) -> Result<(), String> {
        for (id, node) in self.nodes.iter().enumerate() {
            for signal in node.inputs() {
                if let Signal::Node(input) = signal
                    && *input >= id
                {
                    return Err(format!(
                        "wire '{}' is not in topological order (input node {input}, node {id})",
                        node.name()
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn gate_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| matches!(node, BitNode::Lut { .. }).then_some(id))
    }

    pub fn topological_gates(&self) -> Vec<usize> {
        self.gate_ids().collect()
    }

    pub fn scheduled_gates_by_key<F>(&self, mut key: F) -> Result<Vec<usize>, String>
    where
        F: FnMut(usize) -> (usize, usize, usize),
    {
        let mut indegree = vec![0usize; self.nodes.len()];
        let mut successors = vec![Vec::new(); self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            if !matches!(node, BitNode::Lut { .. }) {
                continue;
            }
            for input in node.inputs() {
                if let Signal::Node(input_id) = input
                    && matches!(self.nodes[*input_id], BitNode::Lut { .. })
                {
                    indegree[id] += 1;
                    successors[*input_id].push(id);
                }
            }
        }
        let mut ready: Vec<usize> = self.gate_ids().filter(|id| indegree[*id] == 0).collect();
        let mut order = Vec::with_capacity(self.gate_ids().count());
        while !ready.is_empty() {
            let (position, _) = ready
                .iter()
                .enumerate()
                .max_by_key(|(_, id)| key(**id))
                .unwrap();
            let id = ready.swap_remove(position);
            order.push(id);
            for succ in &successors[id] {
                indegree[*succ] -= 1;
                if indegree[*succ] == 0 {
                    ready.push(*succ);
                }
            }
        }
        if order.len() != self.gate_ids().count() {
            return Err("circuit contains a cycle".to_owned());
        }
        Ok(order)
    }

    pub fn evaluate(&self, inputs: &HashMap<String, bool>) -> Result<Vec<(String, bool)>, String> {
        let mut values = vec![false; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            values[id] = match node {
                BitNode::Input { name } => *inputs
                    .get(name)
                    .ok_or_else(|| format!("missing input '{name}'"))?,
                BitNode::Lut { inputs, table, .. } => {
                    let mut index = 0usize;
                    for signal in inputs {
                        index = (index << 1) | usize::from(signal_value(*signal, &values));
                    }
                    table[index] != 0
                }
            };
        }
        Ok(self
            .outputs
            .iter()
            .map(|(name, signal)| (name.clone(), signal_value(*signal, &values)))
            .collect())
    }

    pub fn exhaustive_inputs(&self) -> Result<Vec<HashMap<String, bool>>, String> {
        if self.inputs.len() > 20 {
            return Err(format!(
                "{} inputs are too many for exhaustive evaluation",
                self.inputs.len()
            ));
        }
        let mut result = Vec::with_capacity(1usize << self.inputs.len());
        for assignment in 0..(1usize << self.inputs.len()) {
            let mut values = HashMap::new();
            for (position, id) in self.inputs.iter().enumerate() {
                let BitNode::Input { name } = &self.nodes[*id] else {
                    unreachable!()
                };
                let bit = (assignment >> (self.inputs.len() - position - 1)) & 1 != 0;
                values.insert(name.clone(), bit);
            }
            result.push(values);
        }
        Ok(result)
    }
}

fn signal_value(signal: Signal, values: &[bool]) -> bool {
    match signal {
        Signal::Const(value) => value,
        Signal::Node(id) => values[id],
    }
}

fn logical_lines(source: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    for raw in source.lines() {
        let raw = raw.split('#').next().unwrap_or("").trim();
        if raw.is_empty() {
            continue;
        }
        let continued = raw.ends_with('\\');
        let part = raw.strip_suffix('\\').unwrap_or(raw).trim();
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(part);
        if !continued {
            result.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

fn parse_bench_invocation(expression: &str) -> Result<(String, Vec<&str>), String> {
    let expression = expression.trim();
    let open = expression
        .find('(')
        .ok_or_else(|| format!("expected an invocation, got '{expression}'"))?;
    let close = expression
        .rfind(')')
        .ok_or_else(|| format!("missing ')' in '{expression}'"))?;
    if close < open || !expression[close + 1..].trim().is_empty() {
        return Err(format!("invalid invocation '{expression}'"));
    }
    let operation = expression[..open].trim().to_ascii_uppercase();
    if operation.is_empty() {
        return Err(format!("missing operation in '{expression}'"));
    }
    let arguments = expression[open + 1..close].trim();
    let inputs = if arguments.is_empty() {
        Vec::new()
    } else {
        arguments
            .split(',')
            .map(str::trim)
            .map(|argument| {
                if argument.is_empty() {
                    Err(format!("empty argument in '{expression}'"))
                } else {
                    Ok(argument)
                }
            })
            .collect::<Result<_, _>>()?
    };
    Ok((operation, inputs))
}

fn lower_bench_gate(
    circuit: &mut BitCircuit,
    output: &str,
    operation: &str,
    inputs: &[Signal],
) -> Result<Signal, String> {
    match operation {
        "BUFF" | "BUF" => {
            if inputs.len() != 1 {
                return Err(format!("{operation} gate '{output}' requires one input"));
            }
            Ok(inputs[0])
        }
        "NOT" | "INV" => {
            if inputs.len() != 1 {
                return Err(format!("{operation} gate '{output}' requires one input"));
            }
            Ok(push_bench_lut(
                circuit,
                output.to_owned(),
                vec![inputs[0]],
                vec![1, 0],
            ))
        }
        "AND" | "NAND" | "OR" | "NOR" | "XOR" | "XNOR" => {
            if inputs.len() < 2 {
                return Err(format!(
                    "{operation} gate '{output}' requires at least two inputs"
                ));
            }
            let associative_operation = match operation {
                "NAND" => "AND",
                "NOR" => "OR",
                "XNOR" => "XOR",
                value => value,
            };
            let mut signal = inputs[0];
            for (position, input) in inputs[1..].iter().enumerate() {
                let is_last = position + 2 == inputs.len();
                let step_operation = if is_last {
                    operation
                } else {
                    associative_operation
                };
                let name = if is_last {
                    output.to_owned()
                } else {
                    format!("{output}$bench${}", position + 1)
                };
                let table = bench_binary_truth_table(step_operation).ok_or_else(|| {
                    format!("unsupported BENCH operation '{step_operation}' at gate '{output}'")
                })?;
                signal = push_bench_lut(circuit, name, vec![signal, *input], table);
            }
            Ok(signal)
        }
        _ => Err(format!(
            "unsupported BENCH operation '{operation}' at gate '{output}'"
        )),
    }
}

fn push_bench_lut(
    circuit: &mut BitCircuit,
    name: String,
    inputs: Vec<Signal>,
    table: Vec<u8>,
) -> Signal {
    let id = circuit.nodes.len();
    circuit.nodes.push(BitNode::Lut {
        name,
        inputs,
        table,
    });
    Signal::Node(id)
}

fn bench_binary_truth_table(operation: &str) -> Option<Vec<u8>> {
    match operation {
        "XOR" => Some(vec![0, 1, 1, 0]),
        "XNOR" => Some(vec![1, 0, 0, 1]),
        "AND" => Some(vec![0, 0, 0, 1]),
        "NAND" => Some(vec![1, 1, 1, 0]),
        "OR" => Some(vec![0, 1, 1, 1]),
        "NOR" => Some(vec![1, 0, 0, 0]),
        _ => None,
    }
}

fn parse_cover(input_count: usize, covers: &[String]) -> Result<Vec<u8>, String> {
    if input_count == 0 {
        if covers.is_empty() {
            return Ok(vec![0]);
        }
        let value = covers[0].split_whitespace().last().unwrap();
        return match value {
            "0" => Ok(vec![0]),
            "1" => Ok(vec![1]),
            _ => Err(format!("invalid constant cover '{value}'")),
        };
    }
    if covers.is_empty() {
        return Ok(vec![0; 1 << input_count]);
    }

    let first: Vec<_> = covers[0].split_whitespace().collect();
    let first_value = if first.len() == 1 {
        1
    } else {
        parse_bit(first[1])?
    };
    let mut table = vec![1 - first_value; 1 << input_count];
    for cover in covers {
        let words: Vec<_> = cover.split_whitespace().collect();
        if words.is_empty() || words.len() > 2 {
            return Err(format!("invalid cover row '{cover}'"));
        }
        let pattern = words[0].as_bytes();
        if pattern.len() != input_count {
            return Err(format!("cover pattern '{}' has wrong width", words[0]));
        }
        let value = if words.len() == 1 {
            1
        } else {
            parse_bit(words[1])?
        };
        for (assignment, slot) in table.iter_mut().enumerate() {
            let matches = pattern.iter().enumerate().all(|(position, byte)| {
                let bit = ((assignment >> (input_count - position - 1)) & 1) as u8;
                *byte == b'-' || *byte == b'0' + bit
            });
            if matches {
                *slot = value;
            }
        }
    }
    Ok(table)
}

fn parse_bit(value: &str) -> Result<u8, String> {
    match value {
        "0" => Ok(0),
        "1" => Ok(1),
        _ => Err(format!("expected Boolean value, got '{value}'")),
    }
}

fn parse_usize_words(line: &str) -> Result<Vec<usize>, String> {
    line.split_whitespace()
        .map(|word| {
            word.parse()
                .map_err(|_| format!("expected an integer, got '{word}'"))
        })
        .collect()
}

fn grouped_wire_count(header: &[usize], kind: &str) -> Result<usize, String> {
    let Some((&groups, widths)) = header.split_first() else {
        return Err(format!("missing Bristol {kind} header"));
    };
    if groups != widths.len() {
        return Err(format!(
            "Bristol {kind} header declares {groups} groups but contains {} widths",
            widths.len()
        ));
    }
    Ok(widths.iter().sum())
}

fn bristol_truth_table(operation: &str, arity: usize) -> Result<Vec<u8>, String> {
    let table = match (operation, arity) {
        ("INV" | "NOT", 1) => vec![1, 0],
        ("XOR", 2) => vec![0, 1, 1, 0],
        ("XNOR", 2) => vec![1, 0, 0, 1],
        ("AND", 2) => vec![0, 0, 0, 1],
        ("NAND", 2) => vec![1, 1, 1, 0],
        ("OR", 2) => vec![0, 1, 1, 1],
        ("NOR", 2) => vec![1, 0, 0, 0],
        _ => {
            return Err(format!(
                "unsupported Bristol operation '{operation}' with arity {arity}"
            ));
        }
    };
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_covers_constants_and_dont_cares() {
        let source = r#"
            .model sample
            .inputs a b
            .outputs and or one
            .names a b and
            11 1
            .names a b or
            00 0
            .names one
            1
            .end
        "#;
        let circuit = BitCircuit::parse_blif(source).unwrap();
        assert_eq!(circuit.nodes.len(), 4);
        let assignments = circuit.exhaustive_inputs().unwrap();
        let outputs: Vec<_> = assignments
            .iter()
            .map(|v| circuit.evaluate(v).unwrap())
            .collect();
        assert_eq!(
            outputs[0],
            vec![
                ("and".into(), false),
                ("or".into(), false),
                ("one".into(), true)
            ]
        );
        assert_eq!(
            outputs[3],
            vec![
                ("and".into(), true),
                ("or".into(), true),
                ("one".into(), true)
            ]
        );
    }

    #[test]
    fn parses_line_continuations() {
        let source = ".inputs a \\\nb\n.outputs z\n.names a b z\n1- 1\n-1 1\n.end\n";
        let circuit = BitCircuit::parse_blif(source).unwrap();
        let BitNode::Lut { table, .. } = &circuit.nodes[2] else {
            panic!()
        };
        assert_eq!(table, &[0, 1, 1, 1]);
    }

    #[test]
    fn parses_iscas_bench_format() {
        let source = r#"
            # ISCAS c17
            INPUT(1)
            INPUT(2)
            INPUT(3)
            INPUT(6)
            INPUT(7)
            OUTPUT(22)
            OUTPUT(23)
            10 = NAND(1, 3)
            11 = NAND(3, 6)
            16 = NAND(2, 11)
            19 = NAND(11, 7)
            22 = NAND(10, 16)
            23 = NAND(16, 19)
        "#;
        let circuit = BitCircuit::parse_bench(source).unwrap();
        assert_eq!(circuit.inputs.len(), 5);
        assert_eq!(circuit.outputs.len(), 2);
        assert_eq!(circuit.nodes.len(), 11);
        assert!(circuit.nodes.iter().all(|node| node.inputs().len() <= 2));

        for inputs in circuit.exhaustive_inputs().unwrap() {
            let n10 = !(inputs["1"] && inputs["3"]);
            let n11 = !(inputs["3"] && inputs["6"]);
            let n16 = !(inputs["2"] && n11);
            let n19 = !(n11 && inputs["7"]);
            let expected = vec![
                ("22".to_owned(), !(n10 && n16)),
                ("23".to_owned(), !(n16 && n19)),
            ];
            assert_eq!(circuit.evaluate(&inputs).unwrap(), expected);
        }
    }

    #[test]
    fn lowers_nary_bench_gates_to_binary_gates() {
        let source = r#"
            INPUT(a)
            INPUT(b)
            INPUT(c)
            INPUT(d)
            OUTPUT(and4)
            OUTPUT(nand4)
            OUTPUT(or4)
            OUTPUT(nor4)
            OUTPUT(xor4)
            OUTPUT(xnor4)
            OUTPUT(buffered)
            and4 = AND(a, b, c, d)
            nand4 = NAND(a, b, c, d)
            or4 = OR(a, b, c, d)
            nor4 = NOR(a, b, c, d)
            xor4 = XOR(a, b, c, d)
            xnor4 = XNOR(a, b, c, d)
            inverted = NOT(a)
            buffered = BUFF(inverted)
        "#;
        let circuit = BitCircuit::parse_bench(source).unwrap();
        assert!(circuit.nodes.iter().all(|node| node.inputs().len() <= 2));

        for inputs in circuit.exhaustive_inputs().unwrap() {
            let values = [inputs["a"], inputs["b"], inputs["c"], inputs["d"]];
            let conjunction = values.into_iter().all(|value| value);
            let disjunction = values.into_iter().any(|value| value);
            let parity = values.into_iter().fold(false, |left, right| left ^ right);
            let expected = vec![
                ("and4".to_owned(), conjunction),
                ("nand4".to_owned(), !conjunction),
                ("or4".to_owned(), disjunction),
                ("nor4".to_owned(), !disjunction),
                ("xor4".to_owned(), parity),
                ("xnor4".to_owned(), !parity),
                ("buffered".to_owned(), !inputs["a"]),
            ];
            assert_eq!(circuit.evaluate(&inputs).unwrap(), expected);
        }
    }

    #[test]
    fn parses_bench_constants() {
        let source = "OUTPUT(zero)\nOUTPUT(one)\nzero = BUFF(FALSE)\none = BUF(TRUE)\n";
        let circuit = BitCircuit::parse_bench(source).unwrap();
        assert_eq!(
            circuit.evaluate(&HashMap::new()).unwrap(),
            vec![("zero".to_owned(), false), ("one".to_owned(), true)]
        );
    }

    #[test]
    fn parses_bristol_format() {
        let source = r#"
            3 7
            2 2 1
            1 1

            2 1 0 2 3 XOR
            2 1 1 3 4 AND
            1 1 4 6 INV
        "#;
        let circuit = BitCircuit::parse_bristol(source).unwrap();
        assert_eq!(circuit.inputs.len(), 3);
        assert_eq!(circuit.outputs.len(), 1);
        for inputs in circuit.exhaustive_inputs().unwrap() {
            let output = circuit.evaluate(&inputs).unwrap()[0].1;
            let expected = !(inputs["i_1"] && (inputs["i_0"] ^ inputs["i_2"]));
            assert_eq!(output, expected);
        }
    }
}
