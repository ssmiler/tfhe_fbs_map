use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use tfhe_fbs_map::{BitCircuit, BitNode, CoefficientPolicy, MapConfig, Mapper, SchedulePolicy};

#[derive(Debug)]
struct Arguments {
    filename: PathBuf,
    input_type: String,
    mapper: String,
    fbs_size: usize,
    strict_fbs_size: bool,
    max_tt_size: usize,
    max_norm2: Option<u128>,
    trials: usize,
    seed: u64,
    verify_samples: usize,
    multioutput_fbs: bool,
    output: Option<PathBuf>,
    output_lbf: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let bit_circuit = BitCircuit::parse_file(&arguments.filename, &arguments.input_type)?;
    let max_fbs_size = if arguments.strict_fbs_size {
        arguments.fbs_size
    } else {
        arguments
            .fbs_size
            .checked_mul(2)
            .ok_or_else(|| "FBS size is too large".to_owned())?
    };
    let coefficient_policy = match arguments.mapper.as_str() {
        "basic" => CoefficientPolicy::Basic,
        "naive" => CoefficientPolicy::Naive,
        "search" => CoefficientPolicy::MinSpan,
        mapper => return Err(format!("unknown mapper '{mapper}'")),
    };
    let base_config = MapConfig {
        fbs_size: arguments.fbs_size,
        max_fbs_size,
        max_truth_table_inputs: arguments.max_tt_size,
        max_norm2: arguments.max_norm2,
        coefficient_policy,
        schedule_policy: SchedulePolicy::Original,
        multioutput_fbs: arguments.multioutput_fbs,
    };

    let mut best = None;
    let mut total_duration = 0.0;
    for trial in 0..arguments.trials {
        let mapper = Mapper::new(MapConfig {
            schedule_policy: trial_schedule(trial, arguments.seed),
            ..base_config.clone()
        })?;
        let start = Instant::now();
        let mapped = mapper.map(&bit_circuit)?;
        total_duration += start.elapsed().as_secs_f64();
        if best
            .as_ref()
            .is_none_or(|current: &tfhe_fbs_map::LutCircuit| {
                mapped.stats().quality_key() < current.stats().quality_key()
            })
        {
            best = Some(mapped);
        }
    }
    let lut_circuit = best.unwrap();
    verify(
        &bit_circuit,
        &lut_circuit,
        arguments.verify_samples,
        arguments.seed,
    )?;

    if let Some(path) = &arguments.output {
        fs::write(path, lut_circuit.to_fbs())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    if let Some(path) = &arguments.output_lbf {
        fs::write(path, lut_circuit.to_lbf())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }

    let stats = lut_circuit.stats();
    println!(
        "{{'nb_inp': {}, 'nb_linprod': {}, 'nb_bootstrap': {}, 'max_lut_size': {}, \
         'norm2_linprod': {}, 'nb_out': {}, 'filename': '{}', 'fbs_size': {}, \
         'mapper': '{}', 'multioutput_fbs': {}, 'strict_fbs_size': {}, 'max_tt_size': {}, 'max_fbs_size': {}, \
         'max_norm2': {}, 'trials': {}, 'time': {:.9}}}",
        stats.input_count,
        stats.linear_count,
        stats.bootstrap_count,
        stats.max_lut_size,
        stats.max_norm2,
        stats.output_count,
        escape_single_quotes(&arguments.filename.to_string_lossy()),
        arguments.fbs_size,
        arguments.mapper,
        python_bool(arguments.multioutput_fbs),
        python_bool(arguments.strict_fbs_size),
        arguments.max_tt_size,
        max_fbs_size,
        arguments
            .max_norm2
            .map(|value| value.to_string())
            .unwrap_or_else(|| "None".to_owned()),
        arguments.trials,
        total_duration,
    );
    Ok(())
}

fn trial_schedule(trial: usize, seed: u64) -> SchedulePolicy {
    if trial == 0 {
        SchedulePolicy::Original
    } else {
        SchedulePolicy::Random {
            seed: seed.wrapping_add((trial - 1) as u64),
        }
    }
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut arguments = env::args().skip(1);
    let mut filename = None;
    let mut mapper = "search".to_owned();
    let mut input_type = "auto".to_owned();
    let mut fbs_size = 3usize;
    let mut strict_fbs_size = false;
    let mut max_tt_size = 16usize;
    let mut max_norm2 = None;
    let mut trials = 1usize;
    let mut seed = 42u64;
    let mut verify_samples = 1000usize;
    let mut multioutput_fbs = false;
    let mut output = None;
    let mut output_lbf = None;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--mapper" => mapper = next_value(&mut arguments, "--mapper")?,
            "--fbs-size" | "--fbs_size" => fbs_size = parse_value(&mut arguments, &argument)?,
            "--strict-fbs-size" | "--strict_fbs_size" => strict_fbs_size = true,
            "--max-tt-size" | "--max_tt_size" => {
                max_tt_size = parse_value(&mut arguments, &argument)?
            }
            "--max-norm2" | "--max_norm2" => {
                max_norm2 = Some(parse_value(&mut arguments, &argument)?)
            }
            "--trials" => trials = parse_value(&mut arguments, &argument)?,
            "--seed" => seed = parse_value(&mut arguments, &argument)?,
            "--verify-samples" => verify_samples = parse_value(&mut arguments, &argument)?,
            "--multioutput-fbs" | "--multioutput_fbs" => multioutput_fbs = true,
            "--output" => output = Some(PathBuf::from(next_value(&mut arguments, "--output")?)),
            "--output-lbf" | "--output_lbf" => {
                output_lbf = Some(PathBuf::from(next_value(&mut arguments, &argument)?));
            }
            "--type" => {
                input_type = next_value(&mut arguments, "--type")?;
                if !matches!(input_type.as_str(), "auto" | "blif" | "bench" | "bristol") {
                    return Err(format!(
                        "unknown input type '{input_type}'; expected auto, blif, bench, or bristol"
                    ));
                }
            }
            value if value.starts_with('-') => return Err(format!("unknown option '{value}'")),
            value => {
                if filename.replace(PathBuf::from(value)).is_some() {
                    return Err("only one input circuit may be supplied".to_owned());
                }
            }
        }
    }

    if trials == 0 {
        return Err("--trials must be positive".to_owned());
    }

    Ok(Arguments {
        filename: filename.ok_or_else(|| "missing input circuit (try --help)".to_owned())?,
        input_type,
        mapper,
        fbs_size,
        strict_fbs_size,
        max_tt_size,
        max_norm2,
        trials,
        seed,
        verify_samples,
        multioutput_fbs,
        output,
        output_lbf,
    })
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_value<T: std::str::FromStr>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    let value = next_value(arguments, option)?;
    value
        .parse()
        .map_err(|_| format!("invalid value '{value}' for {option}"))
}

fn print_help() {
    println!(
        r#"Map a Boolean circuit to TFHE functional bootstraps.

Usage: fbs-map [OPTIONS] <FILE>

Options:
  --mapper <NAME>          basic, naive, or search (default: search)
  --type <FORMAT>          auto, blif, bench, or bristol (default: auto)
  --fbs-size <P>           FBS plaintext size (default: 3)
  --strict-fbs-size        Disable the negacyclic extension from p to 2p
  --max-tt-size <N>        Maximum cone support before forced bootstrap (default: 16)
  --max-norm2 <N>          Maximum squared Euclidean norm of a cone encoding
  --trials <N>             Original order once, then N-1 random orders (default: 1)
  --seed <N>               Random-order and verification seed (default: 42)
  --verify-samples <N>     Random equivalence samples above 16 inputs (default: 1000)
  --multioutput-fbs        Group bootstrap outputs with the same linear input
  --output <FILE>          Write the human-readable FBS circuit
  --output-lbf <FILE>      Write an LBF circuit
  -h, --help               Show this help
"#
    );
}

fn verify(
    bit_circuit: &BitCircuit,
    lut_circuit: &tfhe_fbs_map::LutCircuit,
    samples: usize,
    seed: u64,
) -> Result<(), String> {
    if bit_circuit.inputs.len() <= 16 {
        for inputs in bit_circuit.exhaustive_inputs()? {
            compare_once(bit_circuit, lut_circuit, &inputs)?;
        }
        return Ok(());
    }

    let mut state = seed;
    for _ in 0..samples {
        let mut inputs = HashMap::with_capacity(bit_circuit.inputs.len());
        for id in &bit_circuit.inputs {
            let BitNode::Input { name } = &bit_circuit.nodes[*id] else {
                unreachable!()
            };
            state = xorshift64(state);
            inputs.insert(name.clone(), state & 1 != 0);
        }
        compare_once(bit_circuit, lut_circuit, &inputs)?;
    }
    Ok(())
}

fn compare_once(
    bit_circuit: &BitCircuit,
    lut_circuit: &tfhe_fbs_map::LutCircuit,
    inputs: &HashMap<String, bool>,
) -> Result<(), String> {
    let expected = bit_circuit.evaluate(inputs)?;
    let actual = lut_circuit.evaluate(inputs)?;
    if expected != actual {
        return Err(format!(
            "mapping verification failed for inputs {inputs:?}: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn xorshift64(mut value: u64) -> u64 {
    if value == 0 {
        value = 0x4d59_5df4_d0f3_3173;
    }
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn escape_single_quotes(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trials_use_original_then_distinct_random_seeds() {
        assert_eq!(trial_schedule(0, 42), SchedulePolicy::Original);
        assert_eq!(trial_schedule(1, 42), SchedulePolicy::Random { seed: 42 });
        assert_eq!(trial_schedule(2, 42), SchedulePolicy::Random { seed: 43 });
    }
}
