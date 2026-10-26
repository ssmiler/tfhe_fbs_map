use crate::circuit::{BoolCircuit, BoolNode, BoolWire};
use crate::lut::{LutCircuit, LutSignal};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoefficientPolicy {
    Basic,
    Naive,
    MinSpan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulePolicy {
    Original,
    Random { seed: u64 },
}

/// Controls mapping limits, candidate selection, scheduling, and reuse.
#[derive(Clone, Debug)]
pub struct MapConfig {
    pub fbs_size: usize,
    pub max_fbs_size: usize,
    pub max_truth_table_inputs: usize,
    pub max_norm2: Option<u128>,
    pub coefficient_policy: CoefficientPolicy,
    pub schedule_policy: SchedulePolicy,
    pub multioutput_fbs: bool,
}

impl Default for MapConfig {
    /// Return the default search mapper configuration.
    fn default() -> Self {
        Self {
            fbs_size: 3,
            max_fbs_size: 6,
            max_truth_table_inputs: 16,
            max_norm2: None,
            coefficient_policy: CoefficientPolicy::MinSpan,
            schedule_policy: SchedulePolicy::Original,
            multioutput_fbs: false,
        }
    }
}

/// A Boolean cone together with its linear encoding and attainable values.
#[derive(Clone, Debug)]
struct Cone {
    support: Vec<usize>,        // N variables
    coefficients: Vec<i64>,     // N coefficients, one per support variable
    truth_table: Vec<u8>,       // 2^N Booleans
    multivalue_table: Vec<i64>, // 2^N integers
}

impl Cone {
    // Build a support-free cone for one Boolean constant.
    fn constant(value: u8) -> Self {
        Self {
            support: Vec::new(),
            coefficients: Vec::new(),
            truth_table: vec![value],
            multivalue_table: vec![0],
        }
    }

    // Build an identity cone over one source node.
    fn identity(node: usize) -> Self {
        Self {
            support: vec![node],
            coefficients: vec![1],
            truth_table: vec![0, 1],
            multivalue_table: vec![0, 1],
        }
    }

    // Return a value only for cones using the canonical constant representation.
    fn constant_value(&self) -> Option<u8> {
        (self.support.is_empty() && self.truth_table.len() == 1).then(|| self.truth_table[0])
    }

    // Return the inclusive range width of the cone's linear values.
    fn span(&self) -> usize {
        table_span(&self.multivalue_table)
    }

    // Return the sum of squared coefficients in the cone's linear form.
    fn norm2(&self) -> u128 {
        coefficients_norm2(&self.coefficients)
    }
}

/// A normalized composition ready to be merged into the output cone.
#[derive(Clone, Debug)]
struct Candidate {
    scales: (i64, i64),
    coefficients: Vec<i64>,
    multivalue_table: Vec<i64>,
}

/// Lexicographic heuristic metrics for selecting a candidate scale pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Score {
    reuse_cost: u8,
    span: u128,
    energy: u128,
    norm2: u128,
    a: i64,
    b: i64,
}

impl PartialOrd for Score {
    // Expose the total candidate ranking through PartialOrd.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.ordering(other))
    }

    // Keep `<` explicit and consistent with the complete score ordering.
    fn lt(&self, other: &Self) -> bool {
        self.ordering(other).is_lt()
    }
}

impl Score {
    // Compare fields lexicographically in heuristic priority order.
    fn ordering(&self, other: &Self) -> Ordering {
        self.reuse_cost
            .cmp(&other.reuse_cost)
            .then_with(|| self.span.cmp(&other.span))
            .then_with(|| self.energy.cmp(&other.energy))
            .then_with(|| self.norm2.cmp(&other.norm2))
            .then_with(|| self.a.cmp(&other.a))
            .then_with(|| self.b.cmp(&other.b))
    }
}

type CompositionTables = (Vec<(i64, i64)>, Vec<u8>);
type ConeUpdates = [Option<Cone>; 2];

/// A compacted input-value pair, output bit, and assignment multiplicity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CandidateRow {
    x: i64,
    y: i64,
    truth: u8,
    count: usize,
}

// Expand paired cone values and truth bits into candidate rows.
fn candidate_rows<'a>(
    xy: &'a [(i64, i64)],
    truth_table: &'a [u8],
) -> impl Iterator<Item = CandidateRow> + 'a {
    xy.iter()
        .zip(truth_table)
        .map(|(&(x, y), &truth)| CandidateRow {
            x,
            y,
            truth,
            count: 1,
        })
}

// Collapse repeated candidate rows while preserving assignment multiplicity.
fn compact_candidate_rows(
    xy: &[(i64, i64)],
    truth_table: &[u8],
    ranges: [(i64, usize); 2],
) -> Option<Vec<CandidateRow>> {
    if xy.len() != truth_table.len() || xy.is_empty() {
        return None;
    }
    let dense_size = ranges[0]
        .1
        .checked_mul(ranges[1].1)
        .and_then(|size| size.checked_mul(2))
        .filter(|size| *size <= xy.len().saturating_mul(2));
    // Use direct indexing for compact encoding ranges; avoid huge sparse grids.
    let mut positions = vec![usize::MAX; dense_size.unwrap_or(0)];
    let mut sparse_positions = HashMap::new();
    let mut rows: Vec<CandidateRow> = Vec::new();
    for row in candidate_rows(xy, truth_table) {
        if row.truth > 1 {
            return None;
        }
        let position = if positions.is_empty() {
            *sparse_positions
                .entry((row.x, row.y, row.truth))
                .or_insert(rows.len())
        } else {
            let x = usize::try_from(row.x.checked_sub(ranges[0].0)?).ok()?;
            let y = usize::try_from(row.y.checked_sub(ranges[1].0)?).ok()?;
            if x >= ranges[0].1 || y >= ranges[1].1 {
                return None;
            }
            let position = &mut positions[2 * (x * ranges[1].1 + y) + row.truth as usize];
            if *position == usize::MAX {
                *position = rows.len();
            }
            *position
        };
        if position == rows.len() {
            rows.push(row);
        } else {
            rows[position].count += 1;
        }
    }
    Some(rows)
}

/// Weighted moments used to evaluate each candidate's energy in constant time.
struct EnergyMoments {
    xx: i128,
    xy: i128,
    yy: i128,
}

impl EnergyMoments {
    // Precompute weighted quadratic moments for candidate rows.
    fn new(rows: &[CandidateRow]) -> Option<Self> {
        let mut moments = Self {
            xx: 0,
            xy: 0,
            yy: 0,
        };
        for row in rows {
            let (x, y, count) = (i128::from(row.x), i128::from(row.y), row.count as i128);
            moments.xx = moments
                .xx
                .checked_add(x.checked_mul(x)?.checked_mul(count)?)?;
            moments.xy = moments
                .xy
                .checked_add(x.checked_mul(y)?.checked_mul(count)?)?;
            moments.yy = moments
                .yy
                .checked_add(y.checked_mul(y)?.checked_mul(count)?)?;
        }
        Some(moments)
    }

    // Evaluate the quadratic energy for one pair of scales.
    fn energy(&self, scales: (i64, i64)) -> Option<u128> {
        let (a, b) = (i128::from(scales.0), i128::from(scales.1));
        let xx = a.checked_mul(a)?.checked_mul(self.xx)?;
        let xy = a.checked_mul(b)?.checked_mul(2)?.checked_mul(self.xy)?;
        let yy = b.checked_mul(b)?.checked_mul(self.yy)?;
        u128::try_from(xx.checked_add(xy)?.checked_add(yy)?).ok()
    }
}

// Compute candidate energy from moments, with a checked row-wise fallback.
fn candidate_energy(
    rows: &[CandidateRow],
    moments: Option<&EnergyMoments>,
    scales: (i64, i64),
) -> Option<u128> {
    moments
        .and_then(|moments| moments.energy(scales))
        .or_else(|| {
            // Large cancelling terms can overflow the moment formula despite a small result.
            rows.iter().try_fold(0u128, |energy, row| {
                let value = i128::from(scales.0)
                    .checked_mul(i128::from(row.x))?
                    .checked_add(i128::from(scales.1).checked_mul(i128::from(row.y))?)?;
                let absolute = value.unsigned_abs();
                energy.checked_add(
                    absolute
                        .checked_mul(absolute)?
                        .checked_mul(row.count as u128)?,
                )
            })
        })
}

/// Reusable buffers for evaluating scale pairs over the same two input cones.
struct CandidateWorkspace {
    coefficient_basis: Vec<(i64, i64)>,
    coefficients: Vec<i64>,
    known: Vec<u8>,
}

impl CandidateWorkspace {
    // Build the merged-support coefficient basis used by candidate evaluation.
    fn new(left: &Cone, right: &Cone) -> Self {
        let support = union_support(&left.support, &right.support);
        let coefficient_basis = support
            .iter()
            .map(|node| coefficient_pair(left, right, *node))
            .collect();
        Self {
            coefficient_basis,
            coefficients: Vec::with_capacity(support.len()),
            known: Vec::new(),
        }
    }

    // Merge and canonicalize the scaled input coefficients once per candidate.
    fn prepare_coefficients(&mut self, scales: (i64, i64)) -> Option<i64> {
        let (a, b) = scales;
        self.coefficients.clear();
        for &(left, right) in &self.coefficient_basis {
            self.coefficients
                .push(a.checked_mul(left)?.checked_add(b.checked_mul(right)?)?);
        }
        let divisor = coefficient_normalization_divisor(&self.coefficients);
        for coefficient in &mut self.coefficients {
            *coefficient = coefficient.checked_div(divisor)?;
        }
        Some(divisor)
    }

    // Validate prepared coefficients and return their complete heuristic score.
    fn evaluate_prepared(
        &mut self,
        mapper: &Mapper,
        rows: impl IntoIterator<Item = CandidateRow>,
        scales: (i64, i64),
        divisor: i64,
        energy: u128,
        reuse_cost: impl FnOnce(i64) -> u8,
    ) -> Option<Score> {
        let mut rows = rows.into_iter().peekable();
        rows.peek()?;

        let mut minimum = 0i64;
        let mut span = 1u128;
        let mut norm2 = 0u128;
        for &coefficient in &self.coefficients {
            let absolute = coefficient.unsigned_abs() as u128;
            span += absolute;
            if span > mapper.config.max_fbs_size as u128 {
                return None;
            }
            minimum += coefficient.min(0);
            norm2 += absolute * absolute;
        }
        if mapper.config.max_norm2.is_some_and(|limit| norm2 > limit) {
            return None;
        }

        self.known.clear();
        self.known.resize(span as usize, 2);
        for row in rows {
            let raw_value = scales
                .0
                .checked_mul(row.x)?
                .checked_add(scales.1.checked_mul(row.y)?)?;
            let value = raw_value.checked_div(divisor)?;
            if value.checked_mul(divisor)? != raw_value {
                return None;
            }
            let index = usize::try_from(value.checked_sub(minimum)?).ok()?;
            let entry = self.known.get_mut(index)?;
            if row.truth > 1 || (*entry != 2 && *entry != row.truth) {
                return None;
            }
            *entry = row.truth;
        }
        mapper.is_known_lut_valid(&self.known).then(|| Score {
            reuse_cost: reuse_cost(minimum),
            span,
            energy,
            norm2,
            a: scales.0,
            b: scales.1,
        })
    }

    // Prepare and score a candidate without multi-output reuse information.
    fn evaluate(
        &mut self,
        mapper: &Mapper,
        rows: impl IntoIterator<Item = CandidateRow>,
        scales: (i64, i64),
    ) -> Option<(Score, i64)> {
        let rows: Vec<_> = rows.into_iter().collect();
        let divisor = self.prepare_coefficients(scales)?;
        let divisor_squared = (divisor.unsigned_abs() as u128).checked_pow(2)?;
        let energy = candidate_energy(&rows, EnergyMoments::new(&rows).as_ref(), scales)?
            .checked_div(divisor_squared)?;
        let score = self.evaluate_prepared(mapper, rows, scales, divisor, energy, |_| 1)?;
        Some((score, divisor))
    }

    // Test whether the coefficient basis spans two independent directions.
    fn independent_basis(&self) -> bool {
        let Some(&(first, second)) = self.coefficient_basis.iter().find(|pair| **pair != (0, 0))
        else {
            return false;
        };
        self.coefficient_basis.iter().any(|&(left, right)| {
            i128::from(first) * i128::from(right) != i128::from(second) * i128::from(left)
        })
    }
}

/// Gate and input-basis identity used to cache selected scale pairs.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CandidateCacheKey {
    truth_table: Vec<u8>,
    basis: Vec<(i64, i64)>,
}

type CandidateCache = HashMap<CandidateCacheKey, Option<(i64, i64)>>;

/// Maps Boolean circuits into FBS LUT circuits according to its configuration.
pub struct Mapper {
    pub config: MapConfig,
}

impl Mapper {
    /// Validate and construct a mapper configuration.
    pub fn new(config: MapConfig) -> Result<Self, String> {
        if config.fbs_size < 2 {
            return Err("FBS size must be at least 2".to_owned());
        }
        let negacyclic_limit = config
            .fbs_size
            .checked_mul(2)
            .ok_or_else(|| "FBS size is too large".to_owned())?;
        if config.max_fbs_size < config.fbs_size || config.max_fbs_size > negacyclic_limit {
            return Err("maximum FBS size must be between p and 2p".to_owned());
        }
        if config.max_fbs_size > i64::MAX as usize {
            return Err("FBS size exceeds the supported integer range".to_owned());
        }
        if config.max_truth_table_inputs == 0 || config.max_truth_table_inputs > 20 {
            return Err("max truth-table inputs must be between 1 and 20".to_owned());
        }
        Ok(Self { config })
    }

    // Trace the complete symbolic state of one cone.
    fn trace_cone(&self, label: fmt::Arguments<'_>, cone: &Cone) {
        log::trace!(
            target: "mapper",
            "{label}: support={:?} coefficients={:?} tt={:?} mvt={:?} span={} norm2={}",
            cone.support,
            cone.coefficients,
            cone.truth_table,
            cone.multivalue_table,
            cone.span(),
            cone.norm2()
        );
    }

    // Trace the complete result of one candidate-selection attempt.
    fn trace_candidate(&self, attempt: usize, candidate: Option<&Candidate>, truth_table: &[u8]) {
        match candidate {
            Some(candidate) => log::trace!(
                target: "mapper",
                "candidate attempt={attempt} scales={:?} coefficients={:?} tt={truth_table:?} mvt={:?}",
                candidate.scales, candidate.coefficients, candidate.multivalue_table
            ),
            None => log::trace!(
                target: "mapper",
                "candidate attempt={attempt} scales=None tt={truth_table:?}"
            ),
        }
    }

    /// Map a Boolean circuit and prune nodes unreachable from its outputs.
    pub fn map(&self, circuit: &BoolCircuit) -> Result<LutCircuit, String> {
        if self.config.coefficient_policy == CoefficientPolicy::Basic {
            let mapped = self.map_basic(circuit)?;
            if self
                .config
                .max_norm2
                .is_some_and(|limit| mapped.stats().max_norm2 > limit)
            {
                return Err(format!(
                    "basic mapping requires norm² {}, above the configured limit",
                    mapped.stats().max_norm2
                ));
            }
            return Ok(mapped);
        }
        self.map_once(circuit)
    }

    // Select the configured deterministic or randomized gate schedule.
    fn schedule(
        &self,
        circuit: &BoolCircuit,
        policy: SchedulePolicy,
    ) -> Result<Vec<usize>, String> {
        match policy {
            SchedulePolicy::Original => Ok(circuit.topological_gates()),
            SchedulePolicy::Random { seed } => circuit.scheduled_gates_by_key(8, |id| {
                (splitmix64(seed ^ id as u64) as usize, 0, usize::MAX - id)
            }),
        }
    }

    // Perform one complete optimized mapping pass over a scheduled circuit.
    fn map_once(&self, circuit: &BoolCircuit) -> Result<LutCircuit, String> {
        let order = self.schedule(circuit, self.config.schedule_policy)?;
        let mut environment = LutCircuit::default();
        environment.set_multioutput_fbs(self.config.multioutput_fbs);
        let mut wires: Vec<Option<Cone>> = vec![None; circuit.nodes.len()];
        let mut candidate_cache = CandidateCache::new();
        for id in &circuit.inputs {
            let BoolNode::Input { name } = &circuit.nodes[*id] else {
                unreachable!()
            };
            let LutSignal::Node(node) = environment.input(name) else {
                unreachable!()
            };
            wires[*id] = Some(Cone::identity(node));
        }

        let mut forced_outputs = vec![false; circuit.nodes.len()];
        for (_, output) in &circuit.outputs {
            if let BoolWire::Node(id) = output {
                forced_outputs[*id] = true;
            }
        }

        for gate_id in order {
            let BoolNode::Lut {
                name,
                inputs,
                table,
            } = &circuit.nodes[gate_id]
            else {
                unreachable!()
            };
            let mut input_cones = Vec::with_capacity(inputs.len());
            for input in inputs {
                input_cones.push(match input {
                    BoolWire::Const(value) => Cone::constant(u8::from(*value)),
                    BoolWire::Node(id) => wires[*id]
                        .clone()
                        .ok_or_else(|| format!("gate {gate_id} scheduled before input {id}"))?,
                });
            }

            if log::log_enabled!(target: "mapper", log::Level::Debug) {
                let input_names: Vec<_> = inputs
                    .iter()
                    .map(|input| bool_wire_name(circuit, *input))
                    .collect();
                log::debug!(
                    target: "mapper",
                    "gate id={gate_id} name={name} inputs={input_names:?} table={table:?}"
                );
                for (position, cone) in input_cones.iter().enumerate() {
                    self.trace_cone(format_args!("gate {name} input[{position}]"), cone);
                }
            }

            let (cone, updates) =
                self.map_gate(&mut environment, input_cones, table, &mut candidate_cache)?;
            self.trace_cone(format_args!("gate {name} result"), &cone);
            for (position, replacement) in updates.into_iter().enumerate() {
                if let Some(replacement) = replacement
                    && let BoolWire::Node(id) = inputs[position]
                {
                    self.trace_cone(
                        format_args!("gate {name} updates input {}", circuit.nodes[id].name()),
                        &replacement,
                    );
                    wires[id] = Some(replacement);
                }
            }

            let cone = if !self.config.multioutput_fbs && forced_outputs[gate_id] {
                log::debug!(
                    target: "mapper",
                    "bootstrap reason=forced-output gate={name} position={gate_id}"
                );
                self.bootstrap_cone(&mut environment, cone)?
            } else {
                cone
            };
            wires[gate_id] = Some(cone);
        }

        for (name, output) in &circuit.outputs {
            let cone = match output {
                BoolWire::Const(value) => Cone::constant(u8::from(*value)),
                BoolWire::Node(id) => wires[*id]
                    .clone()
                    .ok_or_else(|| format!("output '{name}' was not mapped"))?,
            };
            log::debug!(
                target: "mapper",
                "output name={name} source={}",
                bool_wire_name(circuit, *output)
            );
            self.trace_cone(format_args!("output {name} cone"), &cone);
            let signal = self.output_signal(&mut environment, cone)?;
            log::debug!(target: "mapper", "output name={name} signal={signal:?}");
            environment.add_output(name.clone(), signal);
        }
        environment.prune();
        Ok(environment)
    }

    // Map each Boolean LUT independently without cone composition search.
    fn map_basic(&self, circuit: &BoolCircuit) -> Result<LutCircuit, String> {
        let mut environment = LutCircuit::default();
        environment.set_multioutput_fbs(self.config.multioutput_fbs);
        let mut wires = vec![LutSignal::Const(0); circuit.nodes.len()];
        for id in &circuit.inputs {
            let BoolNode::Input { name } = &circuit.nodes[*id] else {
                unreachable!()
            };
            wires[*id] = environment.input(name);
        }
        for id in self.schedule(circuit, self.config.schedule_policy)? {
            if let BoolNode::Lut {
                name,
                inputs,
                table,
            } = &circuit.nodes[id]
                && log::log_enabled!(target: "mapper", log::Level::Debug)
            {
                let input_names: Vec<_> = inputs
                    .iter()
                    .map(|input| bool_wire_name(circuit, *input))
                    .collect();
                log::debug!(
                    target: "mapper",
                    "basic gate id={id} name={name} inputs={input_names:?} table={table:?}"
                );
            }
            wires[id] = match &circuit.nodes[id] {
                BoolNode::Input { .. } => unreachable!(),
                BoolNode::Lut { inputs, table, .. } => {
                    let signals: Vec<_> = inputs
                        .iter()
                        .map(|input| match input {
                            BoolWire::Const(value) => LutSignal::Const(i64::from(*value)),
                            BoolWire::Node(input) => wires[*input],
                        })
                        .collect();
                    match signals.as_slice() {
                        [input] => match table.as_slice() {
                            [0, 1] => *input,
                            [1, 0] => environment.linear(&[(-1, *input)], 1),
                            _ => return Err("invalid unary truth table".to_owned()),
                        },
                        [left, right] => {
                            let linear = environment.linear(&[(2, *left), (1, *right)], 0);
                            let (minimum, maximum) = environment.bounds(linear);
                            if self.config.multioutput_fbs
                                && minimum == 0
                                && maximum == 3
                                && let (LutSignal::Node(left), LutSignal::Node(right)) =
                                    (*left, *right)
                                && left != right
                            {
                                let cone = Cone {
                                    support: vec![left, right],
                                    coefficients: vec![2, 1],
                                    truth_table: table.clone(),
                                    multivalue_table: vec![0, 1, 2, 3],
                                };
                                if self.is_lut_valid(&cone.truth_table, &cone.multivalue_table)
                                    || self
                                        .reusable_cone_coefficients(&environment, &cone)
                                        .is_some()
                                {
                                    let bootstrapped =
                                        self.bootstrap_cone(&mut environment, cone)?;
                                    LutSignal::Node(bootstrapped.support[0])
                                } else {
                                    // Preserve basic's fixed encoding when no valid reuse is available.
                                    environment.bootstrap(linear, table.clone())?
                                }
                            } else {
                                let minimum = usize::try_from(minimum).map_err(|_| {
                                    "binary Boolean encoding has a negative minimum".to_owned()
                                })?;
                                let maximum = usize::try_from(maximum).map_err(|_| {
                                    "binary Boolean encoding has a negative maximum".to_owned()
                                })?;
                                let restricted_table = table
                                .get(minimum..=maximum)
                                .ok_or_else(|| {
                                    format!(
                                        "binary Boolean encoding range [{minimum}, {maximum}] is outside its truth table"
                                    )
                                })?
                                .to_vec();
                                let shifted = environment.linear(&[(1, linear)], -(minimum as i64));
                                environment.bootstrap(shifted, restricted_table)?
                            }
                        }
                        _ => {
                            return Err(
                                "basic mapper supports only unary and binary gates".to_owned()
                            );
                        }
                    }
                }
            };
            log::debug!(
                target: "mapper",
                "basic gate id={id} name={} signal={:?}",
                circuit.nodes[id].name(),
                wires[id]
            );
        }
        for (name, output) in &circuit.outputs {
            let signal = match output {
                BoolWire::Const(value) => LutSignal::Const(i64::from(*value)),
                BoolWire::Node(id) => wires[*id],
            };
            log::debug!(
                target: "mapper",
                "output name={name} source={} signal={signal:?}",
                bool_wire_name(circuit, *output)
            );
            environment.add_output(name.clone(), signal);
        }
        environment.prune();
        Ok(environment)
    }

    // Dispatch unary, constant-input, and general binary gates.
    fn map_gate(
        &self,
        environment: &mut LutCircuit,
        mut inputs: Vec<Cone>,
        gate_table: &[u8],
        candidate_cache: &mut CandidateCache,
    ) -> Result<(Cone, ConeUpdates), String> {
        match inputs.len() {
            1 => self.map_unary_gate(inputs.pop().unwrap(), gate_table),
            2 => {
                let right = inputs.pop().unwrap();
                let left = inputs.pop().unwrap();
                if gate_table.len() != 4 {
                    return Err("binary gate must have a four-entry table".to_owned());
                }
                if let Some(constant) = left.constant_value() {
                    let offset = 2 * constant as usize;
                    log::debug!(
                        target: "mapper",
                        "dispatch binary gate to unary constant_position=0 constant={constant}"
                    );
                    return self.map_unary_gate(right, &gate_table[offset..offset + 2]);
                }
                if let Some(constant) = right.constant_value() {
                    let constant = constant as usize;
                    let unary_table = [gate_table[constant], gate_table[2 + constant]];
                    log::debug!(
                        target: "mapper",
                        "dispatch binary gate to unary constant_position=1 constant={constant}"
                    );
                    return self.map_unary_gate(left, &unary_table);
                }
                self.map_binary_gate(
                    environment,
                    left,
                    right,
                    gate_table.to_vec(),
                    candidate_cache,
                )
            }
            _ => Err("mapper supports only unary and binary gates".to_owned()),
        }
    }

    // Apply a unary LUT directly, without searching for a binary composition.
    fn map_unary_gate(&self, cone: Cone, gate_table: &[u8]) -> Result<(Cone, ConeUpdates), String> {
        if gate_table.len() != 2 {
            return Err("unary gate must have a two-entry table".to_owned());
        }
        let truth_table: Vec<u8> = cone
            .truth_table
            .iter()
            .map(|value| gate_table[*value as usize])
            .collect();
        if all_equal(&truth_table) {
            return Ok((Cone::constant(truth_table[0]), [None, None]));
        }
        let result = Cone {
            truth_table,
            ..cone
        };
        self.validate_cone(&result)?;
        Ok((result, [None, None]))
    }

    // Map a binary LUT by composing cones or bootstrapping inputs as needed.
    fn map_binary_gate(
        &self,
        environment: &mut LutCircuit,
        mut left: Cone,
        mut right: Cone,
        mut gate_table: Vec<u8>,
        candidate_cache: &mut CandidateCache,
    ) -> Result<(Cone, ConeUpdates), String> {
        let mut left_position = 0usize;
        let mut right_position = 1usize;
        if cone_order_key(&left) < cone_order_key(&right) {
            switch_inputs(
                &mut left,
                &mut right,
                &mut left_position,
                &mut right_position,
                &mut gate_table,
            );
            log::debug!(
                target: "mapper",
                "switch binary inputs left_position={left_position} right_position={right_position} table={gate_table:?}"
            );
        }
        let mut updates = [None, None];

        let support_size = union_support_size(&left.support, &right.support);
        if support_size > self.config.max_truth_table_inputs {
            log::debug!(
                target: "mapper",
                "bootstrap reason=support-limit input_position={left_position} support_size={support_size} limit={}",
                self.config.max_truth_table_inputs
            );
            left = self.bootstrap_cone(environment, left)?;
            updates[left_position] = Some(left.clone());
            switch_inputs(
                &mut left,
                &mut right,
                &mut left_position,
                &mut right_position,
                &mut gate_table,
            );
            if union_support_size(&left.support, &right.support)
                > self.config.max_truth_table_inputs
            {
                log::debug!(
                    target: "mapper",
                    "bootstrap reason=support-limit input_position={left_position} support_size={} limit={}",
                    union_support_size(&left.support, &right.support),
                    self.config.max_truth_table_inputs
                );
                left = self.bootstrap_cone(environment, left)?;
                updates[left_position] = Some(left.clone());
            }
        }

        let (xy, truth_table) = cone_composition_tables(&left, &right, &gate_table)?;
        if all_equal(&truth_table) {
            log::debug!(
                target: "mapper",
                "candidate attempt=1 constant={}",
                truth_table[0]
            );
            return Ok((Cone::constant(truth_table[0]), updates));
        }
        let candidate = self.find_candidate(
            environment,
            &left,
            &right,
            &xy,
            &truth_table,
            candidate_cache,
        );
        self.trace_candidate(1, candidate.as_ref(), &truth_table);
        if let Some(candidate) = candidate {
            let cone = self.merge_cones(left, right, candidate, truth_table)?;
            return Ok((cone, updates));
        }

        log::debug!(
            target: "mapper",
            "bootstrap reason=no-valid-candidate attempt=1 input_position={left_position}"
        );
        left = self.bootstrap_cone(environment, left)?;
        updates[left_position] = Some(left.clone());
        let (xy, truth_table) = cone_composition_tables(&left, &right, &gate_table)?;
        if all_equal(&truth_table) {
            log::debug!(
                target: "mapper",
                "candidate attempt=2 constant={}",
                truth_table[0]
            );
            return Ok((Cone::constant(truth_table[0]), updates));
        }
        let candidate = self.find_candidate(
            environment,
            &left,
            &right,
            &xy,
            &truth_table,
            candidate_cache,
        );
        self.trace_candidate(2, candidate.as_ref(), &truth_table);
        if let Some(candidate) = candidate {
            let cone = self.merge_cones(left, right, candidate, truth_table)?;
            return Ok((cone, updates));
        }

        log::debug!(
            target: "mapper",
            "bootstrap reason=no-valid-candidate attempt=2 input_position={right_position}"
        );
        right = self.bootstrap_cone(environment, right)?;
        updates[right_position] = Some(right.clone());
        let (xy, truth_table) = cone_composition_tables(&left, &right, &gate_table)?;
        if all_equal(&truth_table) {
            log::debug!(
                target: "mapper",
                "candidate attempt=3 constant={}",
                truth_table[0]
            );
            return Ok((Cone::constant(truth_table[0]), updates));
        }
        let candidate = self.find_candidate(
            environment,
            &left,
            &right,
            &xy,
            &truth_table,
            candidate_cache,
        );
        self.trace_candidate(3, candidate.as_ref(), &truth_table);
        let candidate = candidate.ok_or_else(|| {
            "no valid composition after bootstrapping both inputs as needed".to_owned()
        })?;
        let cone = self.merge_cones(left, right, candidate, truth_table)?;
        Ok((cone, updates))
    }

    // Reuse a known composition or search and cache a new scale pair.
    fn find_candidate(
        &self,
        environment: &LutCircuit,
        left: &Cone,
        right: &Cone,
        xy: &[(i64, i64)],
        truth_table: &[u8],
        cache: &mut CandidateCache,
    ) -> Option<Candidate> {
        if self.config.coefficient_policy == CoefficientPolicy::Naive {
            let scales = (right.span() as i64, 1);
            return self.make_candidate(left, right, xy, truth_table, scales);
        }

        if self.config.multioutput_fbs {
            return self.search_candidate(environment, left, right, xy, truth_table);
        }

        let key = CandidateCacheKey {
            truth_table: truth_table.to_vec(),
            basis: xy.to_vec(),
        };
        if let Some(scales) = cache.get(&key) {
            return scales
                .and_then(|scales| self.make_candidate(left, right, xy, truth_table, scales));
        }
        let candidate = self.search_candidate(environment, left, right, xy, truth_table);
        let cached_scales = candidate.as_ref().map(|candidate| candidate.scales);
        cache.insert(key, cached_scales);
        candidate
    }

    // Search primitive scale pairs and retain the best valid heuristic score.
    fn search_candidate(
        &self,
        environment: &LutCircuit,
        left: &Cone,
        right: &Cone,
        xy: &[(i64, i64)],
        truth_table: &[u8],
    ) -> Option<Candidate> {
        let left_size = left.span() as i64;
        let right_size = right.span() as i64;

        let rows = compact_candidate_rows(
            xy,
            truth_table,
            [
                (*left.multivalue_table.iter().min()?, left_size as usize),
                (*right.multivalue_table.iter().min()?, right_size as usize),
            ],
        )?;
        let mut best: Option<(Score, (i64, i64))> = None;
        let moments = EnergyMoments::new(&rows);
        let mut workspace = CandidateWorkspace::new(left, right);
        let mut bootstrap_lookup = if self.config.multioutput_fbs {
            environment.prepare_bootstrap_lookup(
                union_support(&left.support, &right.support)
                    .into_iter()
                    .zip(&workspace.coefficient_basis)
                    .map(|(node, &(left, right))| (node, left, right)),
            )
        } else {
            None
        };
        let independent_basis = workspace.independent_basis();
        let mut seen_coefficients = HashSet::new();
        for a in 0..=right_size {
            for b in -left_size..=left_size {
                if !is_candidate_scale_pair(a, b) {
                    continue;
                }
                let Some(divisor) = workspace.prepare_coefficients((a, b)) else {
                    continue;
                };
                // Primitive scale pairs are already unique on independent bases.
                if !independent_basis && !seen_coefficients.insert(workspace.coefficients.clone()) {
                    continue;
                }
                let divisor_squared = (divisor.unsigned_abs() as u128).checked_pow(2)?;
                let energy = candidate_energy(&rows, moments.as_ref(), (a, b))?
                    .checked_div(divisor_squared)?;
                let Some(score) = workspace.evaluate_prepared(
                    self,
                    rows.iter().copied(),
                    (a, b),
                    divisor,
                    energy,
                    |minimum| {
                        u8::from(!bootstrap_lookup.as_mut().is_some_and(|lookup| {
                            lookup.has_bootstrap(environment, (a, b), divisor, -minimum)
                        }))
                    },
                ) else {
                    continue;
                };
                // A larger span can still win by reuse until the best reuse cost is zero.
                if best.as_ref().is_some_and(|(best_score, _)| {
                    (!self.config.multioutput_fbs || best_score.reuse_cost == 0)
                        && score.span > best_score.span
                }) {
                    continue;
                }
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _)| score < *best_score)
                {
                    best = Some((score, (a, b)));
                }
            }
        }
        best.and_then(|(_, scales)| self.make_candidate(left, right, xy, truth_table, scales))
    }

    // Materialize the normalized coefficients and values for selected scales.
    fn make_candidate(
        &self,
        left: &Cone,
        right: &Cone,
        xy: &[(i64, i64)],
        truth_table: &[u8],
        scales: (i64, i64),
    ) -> Option<Candidate> {
        if xy.len() != truth_table.len() || xy.is_empty() {
            return None;
        }
        let mut workspace = CandidateWorkspace::new(left, right);
        let (_, divisor) = workspace.evaluate(self, candidate_rows(xy, truth_table), scales)?;
        let multivalue_table = xy
            .iter()
            .map(|(x, y)| {
                scales
                    .0
                    .checked_mul(*x)?
                    .checked_add(scales.1.checked_mul(*y)?)?
                    .checked_div(divisor)
            })
            .collect::<Option<_>>()?;
        Some(Candidate {
            scales,
            coefficients: workspace.coefficients,
            multivalue_table,
        })
    }

    // Merge a selected candidate into a simplified output cone.
    fn merge_cones(
        &self,
        left: Cone,
        right: Cone,
        candidate: Candidate,
        truth_table: Vec<u8>,
    ) -> Result<Cone, String> {
        let support = union_support(&left.support, &right.support);
        debug_assert_eq!(support.len(), candidate.coefficients.len());
        let cone = simplify_cone(Cone {
            support,
            coefficients: candidate.coefficients,
            truth_table,
            multivalue_table: candidate.multivalue_table,
        })?;
        self.validate_cone(&cone)?;
        Ok(cone)
    }

    // Find a reusable coefficient form equivalent to the requested cone.
    fn reusable_cone_coefficients(
        &self,
        environment: &LutCircuit,
        cone: &Cone,
    ) -> Option<Vec<i64>> {
        if !self.config.multioutput_fbs || cone.support.len() <= 1 {
            return None;
        }
        let mut best: Option<((u128, u128), Vec<i64>)> = None;
        for coefficients in environment.reusable_linear_coefficients(&cone.support) {
            let norm2 = coefficients_norm2(&coefficients);
            if self.config.max_norm2.is_some_and(|limit| norm2 > limit) {
                continue;
            }
            let multivalue_table = linear_table(&coefficients);
            if !self.is_lut_valid(&cone.truth_table, &multivalue_table) {
                continue;
            }
            let span = table_span(&multivalue_table) as u128;
            let score = (span, norm2);
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best = Some((score, coefficients));
            }
        }
        best.map(|(_, coefficients)| coefficients)
    }

    // Emit a bootstrap for a cone and replace it with an identity cone.
    fn bootstrap_cone(&self, environment: &mut LutCircuit, mut cone: Cone) -> Result<Cone, String> {
        if cone.support.len() <= 1 {
            self.trace_cone(format_args!("bootstrap skipped"), &cone);
            return Ok(cone);
        }
        self.trace_cone(format_args!("bootstrap input"), &cone);
        if let Some(coefficients) = self.reusable_cone_coefficients(environment, &cone) {
            log::debug!(
                target: "mapper",
                "bootstrap realign coefficients={coefficients:?}"
            );
            cone.multivalue_table = linear_table(&coefficients);
            cone.coefficients = coefficients;
            self.validate_cone(&cone)?;
        }
        let minimum = *cone.multivalue_table.iter().min().unwrap();
        for value in &mut cone.multivalue_table {
            *value -= minimum;
        }
        let terms: Vec<_> = cone
            .coefficients
            .iter()
            .zip(&cone.support)
            .map(|(coefficient, node)| (*coefficient, LutSignal::Node(*node)))
            .collect();
        let linear = environment.linear(&terms, -minimum);
        let table = self.bootstrap_table(&cone.truth_table, &cone.multivalue_table)?;
        log::trace!(
            target: "mapper",
            "bootstrap materialize linear={linear:?} table={table:?}"
        );
        let LutSignal::Node(node) = environment.bootstrap(linear, table)? else {
            return Err("non-constant cone unexpectedly bootstrapped to a constant".to_owned());
        };
        log::debug!(target: "mapper", "bootstrap result node={node}");
        Ok(Cone::identity(node))
    }

    // Convert a final cone to a constant, node, or linear output signal.
    fn output_signal(&self, environment: &mut LutCircuit, cone: Cone) -> Result<LutSignal, String> {
        match cone.support.as_slice() {
            [] => Ok(LutSignal::Const(i64::from(cone.truth_table[0]))),
            [node] => match cone.truth_table.as_slice() {
                [0, 1] => Ok(LutSignal::Node(*node)),
                [1, 0] => Ok(environment.linear(&[(-1, LutSignal::Node(*node))], 1)),
                [value] => Ok(LutSignal::Const(i64::from(*value))),
                table => Err(format!("unexpected one-input output table {table:?}")),
            },
            _ => {
                log::debug!(target: "mapper", "bootstrap reason=output-materialization");
                let bootstrapped = self.bootstrap_cone(environment, cone)?;
                self.output_signal(environment, bootstrapped)
            }
        }
    }

    // Check structural consistency and configured cone complexity limits.
    fn validate_cone(&self, cone: &Cone) -> Result<(), String> {
        let expected = 1usize
            .checked_shl(cone.support.len() as u32)
            .ok_or_else(|| "cone support is too large".to_owned())?;
        if cone.support.len() != cone.coefficients.len()
            || cone.truth_table.len() != expected
            || cone.multivalue_table.len() != expected
        {
            return Err("inconsistent cone dimensions".to_owned());
        }
        if !self.is_lut_valid(&cone.truth_table, &cone.multivalue_table) {
            return Err("invalid cone encoding".to_owned());
        }
        if self
            .config
            .max_norm2
            .is_some_and(|limit| cone.norm2() > limit)
        {
            return Err(format!(
                "cone norm {} exceeds configured norm² limit",
                cone.norm2()
            ));
        }
        Ok(())
    }

    // Test whether a full multivalue table admits the requested Boolean LUT.
    fn is_lut_valid(&self, truth_table: &[u8], multivalue_table: &[i64]) -> bool {
        self.is_lut_valid_with_scratch(truth_table, multivalue_table, &mut Vec::new())
    }

    // Validate a LUT using caller-provided scratch storage.
    fn is_lut_valid_with_scratch(
        &self,
        truth_table: &[u8],
        multivalue_table: &[i64],
        known: &mut Vec<u8>,
    ) -> bool {
        if truth_table.len() != multivalue_table.len() || multivalue_table.is_empty() {
            return false;
        }
        let minimum = *multivalue_table.iter().min().unwrap();
        let maximum = *multivalue_table.iter().max().unwrap();
        let Some(size) = maximum
            .checked_sub(minimum)
            .and_then(|span| span.checked_add(1))
            .and_then(|size| usize::try_from(size).ok())
        else {
            return false;
        };
        if size > self.config.max_fbs_size {
            return false;
        }
        known.clear();
        known.resize(size, 2);
        for (&truth, &value) in truth_table.iter().zip(multivalue_table) {
            let entry = &mut known[(value - minimum) as usize];
            if truth > 1 || (*entry != 2 && *entry != truth) {
                return false;
            }
            *entry = truth;
        }
        self.is_known_lut_valid(known)
    }

    // Test whether known sparse LUT entries admit a valid completion.
    fn is_known_lut_valid(&self, known: &[u8]) -> bool {
        let size = known.len();
        if size <= self.config.fbs_size {
            return true;
        }

        let mut antipodal = true;
        let mut all_zero = true;
        let mut all_one = true;
        for (&left, &right) in known[..size - self.config.fbs_size]
            .iter()
            .zip(&known[self.config.fbs_size..])
        {
            antipodal &= left == 2 || right == 2 || left != right;
            all_zero &= left != 1 && right != 1;
            all_one &= left != 0 && right != 0;
        }
        antipodal || all_zero || all_one
    }

    // Build the bootstrap table for one cone truth table.
    fn bootstrap_table(
        &self,
        truth_table: &[u8],
        multivalue_table: &[i64],
    ) -> Result<Vec<u8>, String> {
        self.composed_bootstrap_table(truth_table, multivalue_table)
            .ok_or_else(|| "cannot build a valid FBS test vector".to_owned())
    }

    // Build a bootstrap table for a binary composition of two cones.
    fn composed_bootstrap_table(
        &self,
        truth_table: &[u8],
        multivalue_table: &[i64],
    ) -> Option<Vec<u8>> {
        if truth_table.len() != multivalue_table.len() {
            return None;
        }
        let minimum = *multivalue_table.iter().min()?;
        let maximum = *multivalue_table.iter().max()?;
        let size = usize::try_from(maximum - minimum + 1).ok()?;
        if size > self.config.max_fbs_size {
            return None;
        }
        let mut known = vec![None; size];
        for (value, truth) in multivalue_table.iter().zip(truth_table) {
            let index = usize::try_from(*value - minimum).ok()?;
            if *truth > 1 || known[index].is_some_and(|previous| previous != *truth) {
                return None;
            }
            known[index] = Some(*truth);
        }
        for missing in [0, 1] {
            let table: Vec<_> = known.iter().map(|value| value.unwrap_or(missing)).collect();
            if self.is_test_vector_valid(&table) {
                return Some(table);
            }
        }

        let overlap = size.checked_sub(self.config.fbs_size)?;
        let mut table: Vec<_> = known.iter().map(|value| value.unwrap_or(0)).collect();
        for index in 0..overlap {
            let opposite = index + self.config.fbs_size;
            match (known[index], known[opposite]) {
                (Some(left), Some(right)) if left != right => {}
                (Some(_), Some(_)) => return None,
                (Some(left), None) => table[opposite] = 1 - left,
                (None, Some(right)) => table[index] = 1 - right,
                (None, None) => table[opposite] = 1,
            }
        }
        self.is_test_vector_valid(&table).then_some(table)
    }

    // Check one completed lookup table against negacyclic FBS constraints.
    fn is_test_vector_valid(&self, table: &[u8]) -> bool {
        if table.len() <= self.config.fbs_size {
            return true;
        }
        if table.len() > self.config.max_fbs_size {
            return false;
        }
        let overlap = table.len() - self.config.fbs_size;
        let left = &table[..overlap];
        let right = &table[self.config.fbs_size..];
        left.iter().zip(right).all(|(x, y)| x != y)
            || (left.iter().all(|x| *x == 0) && right.iter().all(|x| *x == 0))
            || (left.iter().all(|x| *x == 1) && right.iter().all(|x| *x == 1))
    }
}

// Return a readable source name for a Boolean wire.
fn bool_wire_name(circuit: &BoolCircuit, wire: BoolWire) -> String {
    match wire {
        BoolWire::Const(value) => u8::from(value).to_string(),
        BoolWire::Node(id) => circuit.nodes[id].name().to_owned(),
    }
}

#[cfg(test)]
impl Cone {
    fn identity_dummy() -> Self {
        Self::identity(usize::MAX)
    }
}

// Rank cones by span and coefficient norm for deterministic input ordering.
fn cone_order_key(cone: &Cone) -> (usize, u128) {
    (cone.span(), cone.norm2())
}

// Swap binary inputs, positions, and truth-table axes together.
fn switch_inputs(
    left: &mut Cone,
    right: &mut Cone,
    left_position: &mut usize,
    right_position: &mut usize,
    gate_table: &mut [u8],
) {
    std::mem::swap(left, right);
    std::mem::swap(left_position, right_position);
    gate_table.swap(1, 2);
}

// Count the sorted union of two cone supports without allocating it.
fn union_support_size(left: &[usize], right: &[usize]) -> usize {
    left.len() + right.iter().filter(|node| !left.contains(node)).count()
}

// Return the sorted union of two cone supports.
fn union_support(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut support = left.to_vec();
    support.extend(right.iter().filter(|node| !left.contains(node)).copied());
    support
}

// Expand two cone value tables and a binary gate into aligned rows.
fn cone_composition_tables(
    left: &Cone,
    right: &Cone,
    gate_table: &[u8],
) -> Result<CompositionTables, String> {
    let support = union_support(&left.support, &right.support);
    let count = 1usize
        .checked_shl(support.len() as u32)
        .ok_or_else(|| "truth table is too large".to_owned())?;
    let mut xy = Vec::with_capacity(count);
    let mut truth_table = Vec::with_capacity(count);
    let mut right_toggle_masks = vec![0usize; support.len() + 1];
    for (position, node) in right.support.iter().enumerate() {
        let shift = support.len()
            - support
                .iter()
                .position(|candidate| candidate == node)
                .unwrap()
            - 1;
        right_toggle_masks[shift] |= 1 << (right.support.len() - position - 1);
    }
    for shift in 1..right_toggle_masks.len() {
        right_toggle_masks[shift] |= right_toggle_masks[shift - 1];
    }
    let mut right_index = 0;
    for assignment in 0..count {
        let left_index = assignment >> (support.len() - left.support.len());
        let x = left.multivalue_table[left_index];
        let y = right.multivalue_table[right_index];
        let gate_index =
            2 * left.truth_table[left_index] as usize + right.truth_table[right_index] as usize;
        xy.push((x, y));
        truth_table.push(gate_table[gate_index]);
        // Binary increment flips the low bits through the carry position.
        right_index ^= right_toggle_masks[(assignment + 1).trailing_zeros() as usize];
    }
    Ok((xy, truth_table))
}

#[cfg(test)]
fn combined_coefficients_for_support(
    left: &Cone,
    right: &Cone,
    scales: (i64, i64),
    support: &[usize],
) -> Vec<i64> {
    support
        .iter()
        .map(|node| {
            let (left_coefficient, right_coefficient) = coefficient_pair(left, right, *node);
            scales.0 * left_coefficient + scales.1 * right_coefficient
        })
        .collect()
}

// Return both input-cone coefficients for one support node.
fn coefficient_pair(left: &Cone, right: &Cone, node: usize) -> (i64, i64) {
    let coefficient = |cone: &Cone| {
        cone.support
            .iter()
            .position(|candidate| *candidate == node)
            .map(|position| cone.coefficients[position])
            .unwrap_or(0)
    };
    (coefficient(left), coefficient(right))
}

// Remove unused variables and canonicalize a validated cone's linear form.
fn simplify_cone(mut cone: Cone) -> Result<Cone, String> {
    let old_support_len = cone.support.len();
    let retained: Vec<_> = cone
        .coefficients
        .iter()
        .enumerate()
        .filter_map(|(position, coefficient)| (*coefficient != 0).then_some(position))
        .collect();
    if retained.len() != old_support_len {
        let count = 1usize
            .checked_shl(retained.len() as u32)
            .ok_or_else(|| "simplified truth table is too large".to_owned())?;
        let mut truth_table = Vec::with_capacity(count);
        let mut multivalue_table = Vec::with_capacity(count);
        for assignment in 0..count {
            let mut old_assignment = 0usize;
            for old_position in 0..old_support_len {
                old_assignment <<= 1;
                if let Some(new_position) = retained
                    .iter()
                    .position(|position| *position == old_position)
                {
                    old_assignment |= (assignment >> (retained.len() - new_position - 1)) & 1;
                }
            }
            truth_table.push(cone.truth_table[old_assignment]);
            multivalue_table.push(cone.multivalue_table[old_assignment]);
        }
        cone.support = retained
            .iter()
            .map(|position| cone.support[*position])
            .collect();
        cone.coefficients = retained
            .iter()
            .map(|position| cone.coefficients[*position])
            .collect();
        cone.truth_table = truth_table;
        cone.multivalue_table = multivalue_table;
    }

    if !cone.coefficients.is_empty() {
        let divisor = coefficient_normalization_divisor(&cone.coefficients);
        if divisor != 1 {
            for coefficient in &mut cone.coefficients {
                *coefficient /= divisor;
            }
            for value in &mut cone.multivalue_table {
                *value /= divisor;
            }
        }
    }
    Ok(cone)
}

// Enumerate a linear form over all Boolean support assignments.
fn linear_table(coefficients: &[i64]) -> Vec<i64> {
    let count = 1usize << coefficients.len();
    let mut table = Vec::with_capacity(count);
    table.push(0);
    // Expand low-order bits first to preserve Boolean assignment order.
    for coefficient in coefficients.iter().rev() {
        for index in 0..table.len() {
            table.push(table[index] + coefficient);
        }
    }
    table
}

// Keep only primitive scale pairs with a canonical global sign.
fn is_candidate_scale_pair(first: i64, second: i64) -> bool {
    (first > 0 || (first == 0 && second > 0)) && gcd_i64(first, second) == 1
}

// Return a signed GCD that makes the first nonzero coefficient positive.
fn coefficient_normalization_divisor(coefficients: &[i64]) -> i64 {
    let divisor = coefficients
        .iter()
        .fold(0i64, |divisor, coefficient| gcd_i64(divisor, *coefficient))
        .max(1);
    if coefficients
        .iter()
        .find(|coefficient| **coefficient != 0)
        .is_some_and(|coefficient| *coefficient < 0)
    {
        -divisor
    } else {
        divisor
    }
}

// Return the nonnegative greatest common divisor of two coefficients.
fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs()
}

// Return the inclusive integer range covered by a value table.
fn table_span(table: &[i64]) -> usize {
    let minimum = *table.iter().min().unwrap();
    let maximum = *table.iter().max().unwrap();
    usize::try_from(maximum - minimum + 1).unwrap()
}

#[cfg(test)]
// Return the sum of squared values in a table.
fn raw_energy(table: &[i64]) -> u128 {
    table
        .iter()
        .map(|value| {
            let absolute = value.unsigned_abs() as u128;
            absolute.saturating_mul(absolute)
        })
        .sum()
}

// Return the sum of squared linear-form coefficients.
fn coefficients_norm2(coefficients: &[i64]) -> u128 {
    coefficients
        .iter()
        .map(|coefficient| {
            let absolute = coefficient.unsigned_abs() as u128;
            absolute.saturating_mul(absolute)
        })
        .sum()
}

// Test whether every value in a slice equals the first one.
fn all_equal<T: PartialEq>(values: &[T]) -> bool {
    values.windows(2).all(|window| window[0] == window[1])
}

// Mix a seed into one deterministic pseudorandom 64-bit value.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_source(
        source: &str,
        fbs_size: usize,
        policy: CoefficientPolicy,
    ) -> (BoolCircuit, LutCircuit) {
        let circuit = BoolCircuit::parse_blif(source).unwrap();
        let config = MapConfig {
            fbs_size,
            max_fbs_size: 2 * fbs_size,
            coefficient_policy: policy,
            ..MapConfig::default()
        };
        let mapped = Mapper::new(config).unwrap().map(&circuit).unwrap();
        (circuit, mapped)
    }

    fn assert_equivalent(bit: &BoolCircuit, lut: &LutCircuit) {
        for inputs in bit.exhaustive_inputs().unwrap() {
            assert_eq!(
                bit.evaluate(&inputs).unwrap(),
                lut.evaluate(&inputs).unwrap(),
                "inputs: {inputs:?}"
            );
        }
    }

    #[test]
    fn score_order_matches_candidate_ranking() {
        let score = Score {
            reuse_cost: 1,
            span: 1,
            energy: 1,
            norm2: 1,
            a: 1,
            b: 1,
        };

        assert!(
            Score {
                reuse_cost: 0,
                span: 2,
                ..score
            } < score
        );
        assert!(
            Score {
                span: 0,
                energy: 2,
                ..score
            } < score
        );
        assert!(
            Score {
                energy: 0,
                norm2: 2,
                ..score
            } < score
        );
        assert!(
            Score {
                norm2: 0,
                a: 2,
                ..score
            } < score
        );
        assert!(
            Score {
                a: 0,
                b: 2,
                ..score
            } < score
        );
        assert!(Score { b: 0, ..score } < score);
    }

    #[test]
    fn maps_full_adder_with_all_coefficient_policies() {
        let source = r#"
            .inputs a b cin
            .outputs sum cout
            .names a b x1
            01 1
            10 1
            .names x1 cin sum
            01 1
            10 1
            .names a b ab
            11 1
            .names x1 cin xc
            11 1
            .names ab xc cout
            1- 1
            -1 1
            .end
        "#;
        for policy in [
            CoefficientPolicy::Basic,
            CoefficientPolicy::Naive,
            CoefficientPolicy::MinSpan,
        ] {
            let (bit, lut) = map_source(source, 3, policy);
            assert_equivalent(&bit, &lut);
        }
    }

    #[test]
    fn basic_mapper_handles_reconvergent_inversions() {
        let source = r#"
            INPUT(a)
            OUTPUT(y)
            inverted = NOT(a)
            y = XOR(inverted, a)
        "#;
        let circuit = BoolCircuit::parse_bench(source).unwrap();
        let mapped = Mapper::new(MapConfig {
            fbs_size: 3,
            max_fbs_size: 6,
            coefficient_policy: CoefficientPolicy::Basic,
            ..MapConfig::default()
        })
        .unwrap()
        .map(&circuit)
        .unwrap();
        assert_equivalent(&circuit, &mapped);
    }

    #[test]
    fn optimized_and_uses_size_three_table() {
        let source = ".inputs a b\n.outputs y\n.names a b y\n11 1\n.end\n";
        let (bit, lut) = map_source(source, 3, CoefficientPolicy::MinSpan);
        assert_equivalent(&bit, &lut);
        assert_eq!(lut.stats().bootstrap_count, 1);
        assert_eq!(lut.stats().max_lut_size, 3);
    }

    #[test]
    fn binary_gate_with_constant_input_dispatches_to_unary() {
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        let mut environment = LutCircuit::default();
        let mut candidate_cache = CandidateCache::new();
        let gate_table = [0, 1, 0, 0];

        let (left_constant, updates) = mapper
            .map_gate(
                &mut environment,
                vec![Cone::constant(0), Cone::identity(7)],
                &gate_table,
                &mut candidate_cache,
            )
            .unwrap();
        assert_eq!(left_constant.support, vec![7]);
        assert_eq!(left_constant.truth_table, vec![0, 1]);
        assert!(updates.iter().all(Option::is_none));

        let (right_constant, updates) = mapper
            .map_gate(
                &mut environment,
                vec![Cone::identity(7), Cone::constant(1)],
                &gate_table,
                &mut candidate_cache,
            )
            .unwrap();
        assert_eq!(right_constant.support, vec![7]);
        assert_eq!(right_constant.truth_table, vec![1, 0]);
        assert!(updates.iter().all(Option::is_none));
        assert!(candidate_cache.is_empty());
    }

    #[test]
    fn fused_candidate_evaluation_matches_full_table_reference() {
        let cone = |support: Vec<usize>, coefficients: Vec<i64>, truth_table: Vec<u8>| Cone {
            multivalue_table: linear_table(&coefficients),
            support,
            coefficients,
            truth_table,
        };
        let cases = [
            (
                cone(vec![1, 2], vec![1, 2], vec![0, 1, 1, 0]),
                Cone::identity(3),
            ),
            (
                cone(vec![1, 2], vec![2, -1], vec![0, 1, 1, 0]),
                cone(vec![2, 1], vec![-2, 1], vec![1, 0, 0, 0]),
            ),
            (
                cone(vec![1, 2], vec![1, -1], vec![0, 1, 1, 0]),
                cone(vec![1, 2], vec![2, -2], vec![0, 1, 1, 0]),
            ),
            (Cone::constant(1), Cone::identity(1)),
            (Cone::constant(0), Cone::constant(1)),
            (
                cone(vec![1, 2], vec![4, 1], vec![0, 0, 1, 1]),
                Cone::constant(0),
            ),
        ];
        for fbs_size in [2, 3, 4, 8] {
            for max_fbs_size in [fbs_size, 2 * fbs_size] {
                for max_norm2 in [None, Some(5)] {
                    let mapper = Mapper::new(MapConfig {
                        fbs_size,
                        max_fbs_size,
                        max_norm2,
                        ..MapConfig::default()
                    })
                    .unwrap();
                    for (left, right) in &cases {
                        let support = union_support(&left.support, &right.support);
                        for gate in 0..16 {
                            let gate_table: Vec<_> =
                                (0..4).map(|i| ((gate >> i) & 1) as u8).collect();
                            let (xy, truth) =
                                cone_composition_tables(left, right, &gate_table).unwrap();
                            let mut workspace = CandidateWorkspace::new(left, right);
                            let rows = compact_candidate_rows(
                                &xy,
                                &truth,
                                [
                                    (*left.multivalue_table.iter().min().unwrap(), left.span()),
                                    (*right.multivalue_table.iter().min().unwrap(), right.span()),
                                ],
                            )
                            .unwrap();
                            let moments = EnergyMoments::new(&rows);
                            let mut compact_workspace = CandidateWorkspace::new(left, right);
                            for a in -3..=3 {
                                for b in -3..=3 {
                                    let values: Vec<_> =
                                        xy.iter().map(|(x, y)| a * x + b * y).collect();
                                    let mut coefficients = combined_coefficients_for_support(
                                        left,
                                        right,
                                        (a, b),
                                        &support,
                                    );
                                    let divisor = coefficient_normalization_divisor(&coefficients);
                                    for coefficient in &mut coefficients {
                                        *coefficient /= divisor;
                                    }
                                    let values: Vec<_> =
                                        values.into_iter().map(|value| value / divisor).collect();
                                    let span = 1 + coefficients
                                        .iter()
                                        .map(|coefficient| coefficient.unsigned_abs() as u128)
                                        .sum::<u128>();
                                    let norm2 = coefficients_norm2(&coefficients);
                                    let expected = (mapper.is_lut_valid(&truth, &values)
                                        && max_norm2.is_none_or(|limit| norm2 <= limit))
                                    .then(|| {
                                        (
                                            Score {
                                                reuse_cost: 1,
                                                span,
                                                energy: raw_energy(&values),
                                                norm2,
                                                a,
                                                b,
                                            },
                                            divisor,
                                        )
                                    });
                                    let actual = workspace.evaluate(
                                        &mapper,
                                        candidate_rows(&xy, &truth),
                                        (a, b),
                                    );
                                    assert_eq!(
                                        actual, expected,
                                        "p={fbs_size}, max={max_fbs_size}, norm={max_norm2:?}, gate={gate}, scales=({a},{b}), left={left:?}, right={right:?}"
                                    );
                                    let compact_actual = compact_workspace.evaluate(
                                        &mapper,
                                        rows.iter().copied(),
                                        (a, b),
                                    );
                                    assert_eq!(
                                        compact_actual, expected,
                                        "weighted rows, p={fbs_size}, max={max_fbs_size}, norm={max_norm2:?}, gate={gate}, scales=({a},{b})"
                                    );
                                    if actual.is_some() {
                                        assert_eq!(
                                            candidate_energy(&rows, moments.as_ref(), (a, b))
                                                .map(|energy| energy
                                                    / (divisor.unsigned_abs() as u128).pow(2)),
                                            Some(raw_energy(&values))
                                        );
                                        let candidate = mapper
                                            .make_candidate(left, right, &xy, &truth, (a, b))
                                            .unwrap();
                                        assert_eq!(candidate.coefficients, coefficients);
                                        assert_eq!(candidate.multivalue_table, values);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn energy_moments_preserve_signed_weights_and_handle_overflow() {
        let rows = [
            CandidateRow {
                x: -3,
                y: 2,
                truth: 0,
                count: 4,
            },
            CandidateRow {
                x: 2,
                y: -1,
                truth: 1,
                count: 3,
            },
            CandidateRow {
                x: 0,
                y: 4,
                truth: 1,
                count: 2,
            },
        ];
        let moments = EnergyMoments::new(&rows).unwrap();
        for a in -4..=4 {
            for b in -4..=4 {
                let values: Vec<_> = rows
                    .iter()
                    .flat_map(|row| std::iter::repeat_n(a * row.x + b * row.y, row.count))
                    .collect();
                assert_eq!(moments.energy((a, b)), Some(raw_energy(&values)));
            }
        }
        let rows = [CandidateRow {
            x: i64::MAX,
            y: i64::MAX - 1,
            truth: 0,
            count: 2,
        }];
        let moments = EnergyMoments::new(&rows).unwrap();
        assert!(moments.energy((2, -2)).is_none());
        assert_eq!(candidate_energy(&rows, Some(&moments), (2, -2)), Some(8));
        let rows = [CandidateRow {
            x: i64::MAX,
            y: i64::MAX,
            truth: 0,
            count: 3,
        }];
        assert!(EnergyMoments::new(&rows).is_none());
        assert_eq!(candidate_energy(&rows, None, (1, -1)), Some(0));
        assert_eq!(
            candidate_energy(&rows, None, (1, 0)),
            Some(3 * (i64::MAX as u128).pow(2))
        );
        assert_eq!(candidate_energy(&rows, None, (2, 0)), None);
    }

    #[test]
    fn incremental_linear_tables_match_assignment_sums() {
        for length in 0..=5 {
            for encoding in 0..5usize.pow(length) {
                let coefficients: Vec<_> = (0..length)
                    .map(|position| ((encoding / 5usize.pow(position)) % 5) as i64 - 2)
                    .collect();
                let expected: Vec<i64> = (0..1usize << length)
                    .map(|assignment| {
                        coefficients
                            .iter()
                            .enumerate()
                            .map(|(position, coefficient)| {
                                coefficient
                                    * ((assignment >> (coefficients.len() - position - 1)) & 1)
                                        as i64
                            })
                            .sum()
                    })
                    .collect();
                assert_eq!(linear_table(&coefficients), expected);
            }
        }
    }

    #[test]
    fn pruned_search_matches_unpruned_full_table_reference() {
        fn reference(
            mapper: &Mapper,
            environment: &LutCircuit,
            left: &Cone,
            right: &Cone,
            xy: &[(i64, i64)],
            truth: &[u8],
        ) -> Option<(i64, i64)> {
            let support = union_support(&left.support, &right.support);
            let independent = CandidateWorkspace::new(left, right).independent_basis();
            let mut seen = HashSet::new();
            let mut best: Option<(Score, (i64, i64))> = None;
            for a in 0..=right.span() as i64 {
                for b in -(left.span() as i64)..=left.span() as i64 {
                    if !is_candidate_scale_pair(a, b) {
                        continue;
                    }
                    let mut coefficients =
                        combined_coefficients_for_support(left, right, (a, b), &support);
                    let divisor = coefficient_normalization_divisor(&coefficients);
                    for coefficient in &mut coefficients {
                        *coefficient /= divisor;
                    }
                    if !independent && !seen.insert(coefficients.clone()) {
                        continue;
                    }
                    let values: Vec<_> =
                        xy.iter().map(|&(x, y)| (a * x + b * y) / divisor).collect();
                    let norm2 = coefficients_norm2(&coefficients);
                    if !mapper.is_lut_valid(truth, &values)
                        || mapper.config.max_norm2.is_some_and(|limit| norm2 > limit)
                    {
                        continue;
                    }
                    let reuse = if mapper.config.multioutput_fbs
                        && support.iter().all(|node| *node < environment.nodes.len())
                    {
                        let terms: Vec<_> = support
                            .iter()
                            .zip(&coefficients)
                            .map(|(&node, &coefficient)| (coefficient, LutSignal::Node(node)))
                            .collect();
                        let mut probe = environment.clone();
                        probe.set_multioutput_fbs(true);
                        let input = probe.linear(&terms, -*values.iter().min().unwrap());
                        environment.nodes.iter().any(|node| matches!(node, crate::lut::LutNode::Bootstrap { input: node, .. } if LutSignal::Node(*node) == input))
                    } else {
                        false
                    };
                    let span = 1 + coefficients
                        .iter()
                        .map(|coefficient| coefficient.unsigned_abs() as u128)
                        .sum::<u128>();
                    let score = Score {
                        reuse_cost: u8::from(!reuse),
                        span,
                        energy: raw_energy(&values),
                        norm2,
                        a,
                        b,
                    };
                    if best.as_ref().is_none_or(|(previous, _)| score < *previous) {
                        best = Some((score, (a, b)));
                    }
                }
            }
            best.map(|(_, scales)| scales)
        }
        let cone = |support: Vec<usize>, coefficients: Vec<i64>, truth_table: Vec<u8>| Cone {
            multivalue_table: linear_table(&coefficients),
            support,
            coefficients,
            truth_table,
        };
        let cases = [
            (Cone::identity(0), Cone::identity(1)),
            (
                cone(vec![0, 1], vec![1, 1], vec![0, 1, 1, 0]),
                Cone::identity(2),
            ),
            (
                cone(vec![1, 0], vec![2, -1], vec![0, 1, 1, 0]),
                cone(vec![0, 1], vec![-2, 1], vec![0, 0, 0, 1]),
            ),
            (
                cone(vec![0, 1], vec![1, -1], vec![0, 1, 1, 0]),
                cone(vec![0, 1], vec![2, -2], vec![0, 1, 1, 0]),
            ),
            (Cone::constant(0), Cone::identity(0)),
            (Cone::identity_dummy(), Cone::identity(0)),
        ];
        let mut environment = LutCircuit::default();
        let a = environment.input("a");
        let b = environment.input("b");
        environment.input("c");
        for with_reuse in [false, true] {
            if with_reuse {
                let input = environment.linear(&[(2, a), (1, b)], 0);
                environment.bootstrap(input, vec![0, 0, 0, 1]).unwrap();
                environment.bootstrap(a, vec![0, 1]).unwrap();
                let mapper = Mapper::new(MapConfig {
                    fbs_size: 4,
                    max_fbs_size: 8,
                    multioutput_fbs: true,
                    ..MapConfig::default()
                })
                .unwrap();
                let (left, right) = &cases[0];
                let (xy, truth) = cone_composition_tables(left, right, &[0, 0, 0, 1]).unwrap();
                assert_eq!(
                    mapper
                        .search_candidate(&environment, left, right, &xy, &truth)
                        .unwrap()
                        .scales,
                    (2, 1)
                );
            }
            for fbs_size in [2, 4, 8] {
                for max_fbs_size in [fbs_size, 2 * fbs_size] {
                    for max_norm2 in [None, Some(5)] {
                        for multioutput_fbs in [false, true] {
                            let mapper = Mapper::new(MapConfig {
                                fbs_size,
                                max_fbs_size,
                                max_norm2,
                                multioutput_fbs,
                                ..MapConfig::default()
                            })
                            .unwrap();
                            for (left, right) in &cases {
                                for gate in 0..16 {
                                    let table: Vec<_> =
                                        (0..4).map(|index| ((gate >> index) & 1) as u8).collect();
                                    let (xy, truth) =
                                        cone_composition_tables(left, right, &table).unwrap();
                                    assert_eq!(
                                        mapper
                                            .search_candidate(
                                                &environment,
                                                left,
                                                right,
                                                &xy,
                                                &truth,
                                            )
                                            .map(|candidate| candidate.scales),
                                        reference(&mapper, &environment, left, right, &xy, &truth),
                                        "p={fbs_size}, max={max_fbs_size}, norm={max_norm2:?}, multi={multioutput_fbs}, reuse={with_reuse}, gate={gate}, left={left:?}, right={right:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simplifies_and_canonicalizes_candidates_before_validation() {
        let left = Cone {
            support: vec![0, 1],
            coefficients: vec![1, 1],
            truth_table: vec![0, 0, 0, 1],
            multivalue_table: vec![0, 1, 1, 2],
        };
        let right = Cone {
            support: vec![0, 1],
            coefficients: vec![1, -1],
            truth_table: vec![0, 0, 1, 0],
            multivalue_table: vec![0, -1, 1, 0],
        };
        let mapper = Mapper::new(MapConfig {
            fbs_size: 2,
            max_fbs_size: 2,
            ..MapConfig::default()
        })
        .unwrap();
        let (xy, truth) = cone_composition_tables(&left, &right, &[0, 1, 1, 1]).unwrap();

        for scales in [(1, 1), (-1, -1)] {
            let candidate = mapper
                .make_candidate(&left, &right, &xy, &truth, scales)
                .unwrap();
            assert_eq!(candidate.coefficients, [1, 0]);
            assert_eq!(candidate.multivalue_table, [0, 0, 1, 1]);
        }

        let candidate = mapper
            .make_candidate(&left, &right, &xy, &truth, (-1, -1))
            .unwrap();
        let cone = mapper.merge_cones(left, right, candidate, truth).unwrap();
        assert_eq!(cone.support, [0]);
        assert_eq!(cone.coefficients, [1]);
        assert_eq!(cone.multivalue_table, [0, 1]);
    }

    #[test]
    fn search_scores_merged_span_after_shared_variable_cancellation() {
        let left = Cone {
            support: vec![0, 1],
            coefficients: vec![1, 1],
            truth_table: vec![0, 1, 1, 0],
            multivalue_table: vec![0, 1, 1, 2],
        };
        let right = Cone {
            support: vec![1, 2],
            coefficients: vec![1, 1],
            truth_table: vec![0, 1, 1, 0],
            multivalue_table: vec![0, 1, 1, 2],
        };
        let mapper = Mapper::new(MapConfig {
            fbs_size: 4,
            max_fbs_size: 8,
            ..MapConfig::default()
        })
        .unwrap();
        let (xy, truth) = cone_composition_tables(&left, &right, &[0, 1, 1, 0]).unwrap();

        let candidate = mapper
            .search_candidate(&LutCircuit::default(), &left, &right, &xy, &truth)
            .unwrap();
        assert_eq!(candidate.scales, (1, -1));
        assert_eq!(candidate.coefficients, [1, 0, -1]);
    }

    #[test]
    fn candidate_rejections_stop_before_row_validation() {
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        let left = Cone::identity(1);
        let right = Cone::identity(2);
        let (xy, truth) = cone_composition_tables(&left, &right, &[0, 0, 0, 1]).unwrap();
        let mut workspace = CandidateWorkspace::new(&left, &right);
        assert!(
            workspace
                .evaluate(&mapper, candidate_rows(&xy, &truth), (6, 1))
                .is_none()
        );
        assert!(workspace.known.is_empty());
        assert_eq!(
            workspace.evaluate(&mapper, candidate_rows(&xy, &truth), (2, 1)),
            Some((
                Score {
                    reuse_cost: 1,
                    span: 4,
                    energy: 14,
                    norm2: 5,
                    a: 2,
                    b: 1,
                },
                1,
            ))
        );
        assert_eq!(workspace.known, [0, 0, 0, 1]);
        assert!(
            workspace
                .evaluate(&mapper, candidate_rows(&xy, &truth), (0, 0))
                .is_none()
        );
        assert_eq!(workspace.known, [0]);
        assert!(workspace.evaluate(&mapper, [], (1, 1)).is_none());
    }

    #[test]
    fn compact_rows_preserve_weights_and_conflicting_truths_in_dense_and_sparse_ranges() {
        let xy = [(0, 0), (1, 0), (1, 0), (1, 0)];
        let truth = [0, 0, 1, 0];
        let dense = compact_candidate_rows(&xy, &truth, [(0, 2), (0, 1)]).unwrap();
        let sparse = compact_candidate_rows(&xy, &truth, [(0, 1000), (0, 1000)]).unwrap();
        assert_eq!(dense, sparse);
        assert_eq!(
            dense.iter().map(|row| row.count).collect::<Vec<_>>(),
            [1, 2, 1]
        );
        assert_eq!(dense[1].truth, 0);
        assert_eq!(dense[2].truth, 1);
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        let mut workspace = CandidateWorkspace::new(&Cone::identity(1), &Cone::constant(0));
        assert!(
            workspace
                .evaluate(&mapper, dense.iter().copied(), (1, 0))
                .is_none()
        );
        assert!(compact_candidate_rows(&xy, &[0], [(0, 2), (0, 1)]).is_none());
        assert!(compact_candidate_rows(&[], &[], [(0, 2), (0, 1)]).is_none());
        assert!(compact_candidate_rows(&xy, &[0, 0, 2, 0], [(0, 2), (0, 1)]).is_none());
    }

    #[test]
    fn selected_candidate_expands_weighted_rows_in_assignment_order() {
        let left = Cone {
            support: vec![0, 1],
            coefficients: vec![1, 1],
            truth_table: vec![0, 0, 0, 1],
            multivalue_table: vec![0, 1, 1, 2],
        };
        let right = Cone::identity(2);
        let (xy, truth) = cone_composition_tables(&left, &right, &[0, 1, 1, 0]).unwrap();
        let rows = compact_candidate_rows(&xy, &truth, [(0, 3), (0, 2)]).unwrap();
        assert_eq!(xy.len(), 8);
        assert_eq!(rows.len(), 6);
        assert_eq!(rows.iter().map(|row| row.count).sum::<usize>(), 8);
        let environment = LutCircuit::default();
        for multioutput_fbs in [false, true] {
            let mapper = Mapper::new(MapConfig {
                fbs_size: 4,
                max_fbs_size: 8,
                multioutput_fbs,
                ..MapConfig::default()
            })
            .unwrap();
            let candidate = mapper
                .search_candidate(&environment, &left, &right, &xy, &truth)
                .unwrap();
            assert_eq!(
                candidate.multivalue_table,
                linear_table(&candidate.coefficients)
            );
            assert_eq!(candidate.multivalue_table.len(), xy.len());
            let moments = EnergyMoments::new(&rows).unwrap();
            let support = union_support(&left.support, &right.support);
            let raw_coefficients =
                combined_coefficients_for_support(&left, &right, candidate.scales, &support);
            let divisor = coefficient_normalization_divisor(&raw_coefficients);
            assert_eq!(
                candidate_energy(&rows, Some(&moments), candidate.scales)
                    .map(|energy| energy / (divisor.unsigned_abs() as u128).pow(2)),
                Some(raw_energy(&candidate.multivalue_table))
            );
        }
    }

    #[test]
    fn completes_sparse_test_vectors_with_mixed_filling() {
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        let truth = [0, 0, 1, 1];
        let values = [-2, -1, 2, 3];
        assert!(mapper.is_lut_valid(&truth, &values));
        assert_eq!(
            mapper.bootstrap_table(&truth, &values).unwrap(),
            [0, 0, 0, 1, 1, 1]
        );
    }

    #[test]
    fn sparse_test_vector_completion_matches_exhaustive_search() {
        for fbs_size in 2..=3 {
            let mapper = Mapper::new(MapConfig {
                fbs_size,
                max_fbs_size: 2 * fbs_size,
                ..MapConfig::default()
            })
            .unwrap();
            for size in 1..=2 * fbs_size {
                for pattern in 0..3usize.pow(size as u32) {
                    let mut pattern = pattern;
                    let known: Vec<_> = (0..size)
                        .map(|_| {
                            let digit = pattern % 3;
                            pattern /= 3;
                            digit.checked_sub(1).map(|value| value as u8)
                        })
                        .collect();
                    if known[0].is_none() || known[size - 1].is_none() {
                        continue;
                    }
                    let (values, truth): (Vec<_>, Vec<_>) = known
                        .iter()
                        .enumerate()
                        .filter_map(|(index, value)| value.map(|value| (index as i64, value)))
                        .unzip();
                    let missing: Vec<_> = known
                        .iter()
                        .enumerate()
                        .filter_map(|(index, value)| value.is_none().then_some(index))
                        .collect();
                    let can_complete = (0..1usize << missing.len()).any(|assignment| {
                        let mut table: Vec<_> =
                            known.iter().map(|value| value.unwrap_or(0)).collect();
                        for (bit, index) in missing.iter().enumerate() {
                            table[*index] = ((assignment >> bit) & 1) as u8;
                        }
                        mapper.is_test_vector_valid(&table)
                    });
                    let completed = mapper.composed_bootstrap_table(&truth, &values);
                    assert_eq!(
                        mapper.is_lut_valid(&truth, &values),
                        can_complete,
                        "p={fbs_size}, known={known:?}"
                    );
                    assert_eq!(
                        completed.is_some(),
                        can_complete,
                        "p={fbs_size}, known={known:?}"
                    );
                    if let Some(table) = completed {
                        assert!(mapper.is_test_vector_valid(&table));
                        for (index, value) in known.iter().enumerate() {
                            if let Some(value) = value {
                                assert_eq!(table[index], *value);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_test_vectors_reject_conflicting_reachable_values_and_limits() {
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        assert!(!mapper.is_lut_valid(&[], &[]));
        assert!(!mapper.is_lut_valid(&[0], &[0, 1]));
        assert!(!mapper.is_lut_valid(&[2], &[0]));
        assert!(!mapper.is_lut_valid(&[0, 1], &[i64::MIN, i64::MAX]));
        assert!(!mapper.is_lut_valid(&[0, 1], &[0, 0]));
        assert!(mapper.bootstrap_table(&[0, 1], &[0, 0]).is_err());
        assert!(!mapper.is_lut_valid(&[0, 1, 0, 1], &[0, 1, 3, 4]));
        assert!(!mapper.is_lut_valid(&[0, 1], &[0, 6]));
        let strict = Mapper::new(MapConfig {
            max_fbs_size: 3,
            ..MapConfig::default()
        })
        .unwrap();
        assert!(!strict.is_lut_valid(&[0, 0, 1, 1], &[-2, -1, 2, 3]));
    }

    #[test]
    fn multioutput_mode_realigns_a_cone_to_an_existing_bootstrap_input() {
        let mapper = Mapper::new(MapConfig {
            fbs_size: 3,
            max_fbs_size: 6,
            multioutput_fbs: true,
            ..MapConfig::default()
        })
        .unwrap();
        let mut mapped = LutCircuit::default();
        mapped.set_multioutput_fbs(true);
        let LutSignal::Node(a) = mapped.input("a") else {
            unreachable!()
        };
        let LutSignal::Node(b) = mapped.input("b") else {
            unreachable!()
        };
        let support = vec![a, b];
        let first = Cone {
            support: support.clone(),
            coefficients: vec![2, 1],
            truth_table: vec![0, 0, 0, 1],
            multivalue_table: vec![0, 1, 2, 3],
        };
        let first = mapper.bootstrap_cone(&mut mapped, first).unwrap();
        let second = Cone {
            support,
            coefficients: vec![1, 1],
            truth_table: vec![0, 1, 1, 0],
            multivalue_table: vec![0, 1, 1, 2],
        };

        assert!(
            mapper
                .reusable_cone_coefficients(&mapped, &second)
                .is_some()
        );
        let second = mapper.bootstrap_cone(&mut mapped, second).unwrap();
        mapped.add_output("and".into(), LutSignal::Node(first.support[0]));
        mapped.add_output("xor".into(), LutSignal::Node(second.support[0]));
        assert_eq!(mapped.stats().bootstrap_count, 1);
        assert!(mapped.to_lbf().contains(".bootstrap m3 m4 m5"));
    }

    #[test]
    fn multioutput_mode_keeps_scaled_predecessor_outputs_in_the_merged_cone() {
        let source = r#"
            .inputs a b
            .outputs carry any
            .names a b parity
            01 1
            10 1
            .names a b carry
            11 1
            .names parity carry any
            1- 1
            -1 1
            .end
        "#;
        let circuit = BoolCircuit::parse_blif(source).unwrap();
        let mapped = Mapper::new(MapConfig {
            fbs_size: 3,
            max_fbs_size: 6,
            multioutput_fbs: true,
            ..MapConfig::default()
        })
        .unwrap()
        .map(&circuit)
        .unwrap();
        assert_equivalent(&circuit, &mapped);
        assert_eq!(mapped.stats().bootstrap_count, 1);
        assert_eq!(mapped.to_lbf().matches(".bootstrap ").count(), 1);
    }

    #[test]
    fn basic_multioutput_mode_supports_all_binary_truth_tables_at_size_two() {
        let mut circuit = BoolCircuit::default();
        for name in ["a", "b"] {
            circuit.inputs.push(circuit.nodes.len());
            circuit.nodes.push(BoolNode::Input { name: name.into() });
        }
        for truth in 0..16 {
            let id = circuit.nodes.len();
            let name = format!("y{truth}");
            circuit.nodes.push(BoolNode::Lut {
                name: name.clone(),
                inputs: vec![BoolWire::Node(0), BoolWire::Node(1)],
                table: (0..4).map(|index| ((truth >> index) & 1) as u8).collect(),
            });
            circuit.outputs.push((name, BoolWire::Node(id)));
        }
        for multioutput_fbs in [false, true] {
            let mapped = Mapper::new(MapConfig {
                fbs_size: 2,
                max_fbs_size: 4,
                coefficient_policy: CoefficientPolicy::Basic,
                multioutput_fbs,
                ..MapConfig::default()
            })
            .unwrap()
            .map(&circuit)
            .unwrap();
            assert_equivalent(&circuit, &mapped);
            assert_eq!(
                mapped.stats().bootstrap_count,
                if multioutput_fbs { 1 } else { 16 }
            );
            assert_eq!(mapped.stats().max_lut_size, 4);
            assert_eq!(mapped.stats().max_norm2, 5);
        }
    }

    #[test]
    fn basic_mapper_uses_multioutput_cost_when_encodings_can_be_aligned() {
        let source = r#"
            .inputs a b
            .outputs y z
            .names a b y
            10 1
            .names b a z
            01 1
            10 1
            .end
        "#;
        let circuit = BoolCircuit::parse_blif(source).unwrap();
        let mapped = Mapper::new(MapConfig {
            fbs_size: 3,
            max_fbs_size: 6,
            coefficient_policy: CoefficientPolicy::Basic,
            multioutput_fbs: true,
            ..MapConfig::default()
        })
        .unwrap()
        .map(&circuit)
        .unwrap();
        assert_equivalent(&circuit, &mapped);
        assert_eq!(mapped.stats().bootstrap_count, 1);
        assert!(mapped.to_lbf().contains(".bootstrap m3 m4 m5"));
    }

    #[test]
    fn schedules_remain_equivalent() {
        let source = r#"
            .inputs a b c d
            .outputs y z
            .names a b x
            01 1
            10 1
            .names c d w
            11 1
            .names x w y
            1- 1
            -1 1
            .names x c z
            11 1
            .end
        "#;
        let bit = BoolCircuit::parse_blif(source).unwrap();
        for schedule_policy in [SchedulePolicy::Original, SchedulePolicy::Random { seed: 7 }] {
            let mapper = Mapper::new(MapConfig {
                fbs_size: 3,
                max_fbs_size: 6,
                schedule_policy,
                ..MapConfig::default()
            })
            .unwrap();
            let lut = mapper.map(&bit).unwrap();
            assert_equivalent(&bit, &lut);
        }
    }

    #[test]
    fn random_scheduler_changes_the_order_of_ready_gates() {
        let source = r#"
            .inputs a b c d
            .outputs y z
            .names a b x
            01 1
            10 1
            .names c d w
            11 1
            .names x w y
            1- 1
            -1 1
            .names x c z
            11 1
            .end
        "#;
        let circuit = BoolCircuit::parse_blif(source).unwrap();
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        let original = mapper.schedule(&circuit, SchedulePolicy::Original).unwrap();
        assert!((0..8).any(|seed| {
            mapper
                .schedule(&circuit, SchedulePolicy::Random { seed })
                .unwrap()
                != original
        }));
    }

    #[test]
    fn random_schedule_window_is_bounded_and_reproducible() {
        let mut source = ".inputs a b\n.outputs g31\n".to_owned();
        for i in 0..32 {
            source.push_str(&format!(".names a b g{i}\n11 1\n"));
        }
        let circuit = BoolCircuit::parse_blif(&source).unwrap();
        let mapper = Mapper::new(MapConfig::default()).unwrap();
        for seed in 0..8 {
            let policy = SchedulePolicy::Random { seed };
            let order = mapper.schedule(&circuit, policy).unwrap();
            assert_eq!(order, mapper.schedule(&circuit, policy).unwrap());
            let mut remaining = circuit.topological_gates();
            for id in order {
                assert!(remaining.iter().take(8).any(|node| *node == id));
                remaining.retain(|node| *node != id);
            }
            assert!(remaining.is_empty());
        }
    }

    #[test]
    fn composition_preserves_support_order_with_shared_and_permuted_inputs() {
        fn supports(prefix: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
            result.push(prefix.clone());
            for node in [8, 3, 7, 2] {
                if !prefix.contains(&node) {
                    prefix.push(node);
                    supports(prefix, result);
                    prefix.pop();
                }
            }
        }
        let cone = |support: Vec<usize>| {
            let coefficients: Vec<_> = (0..support.len()).map(|i| 1 << i).collect();
            let multivalue_table = linear_table(&coefficients);
            let truth_table = multivalue_table
                .iter()
                .map(|value| (value % 2) as u8)
                .collect();
            Cone {
                support,
                coefficients,
                multivalue_table,
                truth_table,
            }
        };
        let mut cases = Vec::new();
        supports(&mut Vec::new(), &mut cases);
        for left_support in &cases {
            let left = cone(left_support.clone());
            for right_support in &cases {
                let right = cone(right_support.clone());
                let union = union_support(&left.support, &right.support);
                let project = |assignment: usize, subset: &[usize]| {
                    subset.iter().fold(0, |index, node| {
                        let position = union
                            .iter()
                            .position(|candidate| candidate == node)
                            .unwrap();
                        (index << 1) | ((assignment >> (union.len() - position - 1)) & 1)
                    })
                };
                let (xy, truth) = cone_composition_tables(&left, &right, &[0, 1, 1, 0]).unwrap();
                for assignment in 0..xy.len() {
                    let left_index = project(assignment, &left.support);
                    let right_index = project(assignment, &right.support);
                    assert_eq!(
                        xy[assignment],
                        (
                            left.multivalue_table[left_index],
                            right.multivalue_table[right_index]
                        )
                    );
                    assert_eq!(
                        truth[assignment],
                        left.truth_table[left_index] ^ right.truth_table[right_index]
                    );
                }
            }
        }
    }

    #[test]
    fn candidate_scale_filter_removes_projective_symmetries() {
        assert!(is_candidate_scale_pair(1, -2));
        assert!(is_candidate_scale_pair(1, 0));
        assert!(is_candidate_scale_pair(0, 1));
        assert!(!is_candidate_scale_pair(0, -1));
        assert!(!is_candidate_scale_pair(2, 2));
        assert!(!is_candidate_scale_pair(0, 0));
    }

    #[test]
    fn independent_bases_need_no_normalized_coefficient_deduplication() {
        for basis in [
            vec![(1, 0), (0, 1)],
            vec![(2, -1), (3, 2)],
            vec![(0, 0), (-2, 3), (1, 4)],
        ] {
            let workspace = CandidateWorkspace {
                coefficient_basis: basis,
                coefficients: Vec::new(),
                known: Vec::new(),
            };
            assert!(workspace.independent_basis());
            let mut seen = HashSet::new();
            for a in 0..=8 {
                for b in -8..=8 {
                    if is_candidate_scale_pair(a, b) {
                        let mut key: Vec<_> = workspace
                            .coefficient_basis
                            .iter()
                            .map(|(left, right)| a * left + b * right)
                            .collect();
                        let divisor = coefficient_normalization_divisor(&key);
                        for coefficient in &mut key {
                            *coefficient /= divisor;
                        }
                        assert!(seen.insert(key), "duplicate scales ({a}, {b})");
                    }
                }
            }
        }
        for basis in [
            vec![],
            vec![(0, 0)],
            vec![(1, 2), (-2, -4)],
            vec![(0, 1), (0, -2)],
        ] {
            assert!(
                !CandidateWorkspace {
                    coefficient_basis: basis,
                    coefficients: Vec::new(),
                    known: Vec::new(),
                }
                .independent_basis()
            );
        }
    }

    #[test]
    fn respects_norm_limit_by_bootstrapping() {
        let source = r#"
            .inputs a b c
            .outputs y
            .names a b x
            01 1
            10 1
            .names x c y
            11 1
            .end
        "#;
        let bit = BoolCircuit::parse_blif(source).unwrap();
        let mapped = Mapper::new(MapConfig {
            fbs_size: 8,
            max_fbs_size: 16,
            max_norm2: Some(2),
            ..MapConfig::default()
        })
        .unwrap()
        .map(&bit)
        .unwrap();
        assert_equivalent(&bit, &mapped);
        assert!(mapped.stats().max_norm2 <= 2);
    }

    #[test]
    fn random_dags_are_equivalent_under_supported_policies() {
        let mut state = 0x5eed_cafe_f00d_beefu64;
        for circuit_index in 0..12 {
            let mut circuit = BoolCircuit::default();
            for input in 0..6 {
                let id = circuit.nodes.len();
                circuit.nodes.push(BoolNode::Input {
                    name: format!("i{input}"),
                });
                circuit.inputs.push(id);
            }
            for gate in 0..18 {
                state = splitmix64(state);
                let left = state as usize % circuit.nodes.len();
                state = splitmix64(state);
                let mut right = state as usize % circuit.nodes.len();
                if right == left {
                    right = (right + 1) % circuit.nodes.len();
                }
                state = splitmix64(state);
                let mut table = vec![0; 4];
                for (index, value) in table.iter_mut().enumerate() {
                    *value = ((state >> index) & 1) as u8;
                }
                if all_equal(&table) {
                    table[3] ^= 1;
                }
                circuit.nodes.push(BoolNode::Lut {
                    name: format!("c{circuit_index}_g{gate}"),
                    inputs: vec![BoolWire::Node(left), BoolWire::Node(right)],
                    table,
                });
            }
            for output in 0..4 {
                let id = circuit.nodes.len() - 1 - output * 3;
                circuit
                    .outputs
                    .push((format!("o{output}"), BoolWire::Node(id)));
            }

            for (coefficient_policy, schedule_policy) in [
                (CoefficientPolicy::Basic, SchedulePolicy::Original),
                (CoefficientPolicy::MinSpan, SchedulePolicy::Original),
                (
                    CoefficientPolicy::MinSpan,
                    SchedulePolicy::Random {
                        seed: circuit_index,
                    },
                ),
            ] {
                let mapped = Mapper::new(MapConfig {
                    fbs_size: 3 + circuit_index as usize % 4,
                    max_fbs_size: 2 * (3 + circuit_index as usize % 4),
                    max_truth_table_inputs: 10,
                    coefficient_policy,
                    schedule_policy,
                    multioutput_fbs: true,
                    ..MapConfig::default()
                })
                .unwrap()
                .map(&circuit)
                .unwrap();
                assert_equivalent(&circuit, &mapped);
            }
        }
    }
}
