pub mod circuit;
pub mod lut;
pub mod mapper;

use circuit::{BoolCircuit, BoolNode};
use lut::LutCircuit;
use mapper::{CoefficientPolicy, MapConfig, Mapper, SchedulePolicy};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Minimal logger that writes enabled records to standard error.
struct StderrLogger;

impl log::Log for StderrLogger {
    // Accept records up to the process-wide maximum log level.
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level().to_level_filter() <= log::max_level()
    }

    // Write one level-and-target-prefixed record to standard error.
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{} {}: {}", record.level(), record.target(), record.args());
        }
    }

    // Standard error requires no explicit flushing here.
    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

// Install the process logger with warnings enabled by default.
fn init_logger() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
}

// Map repeated verbosity flags through the standard logging levels.
fn verbosity_level(verbosity: usize) -> log::LevelFilter {
    match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    }
}

/// Allowed LUT span relative to the FBS size.
#[derive(Debug, PartialEq, Eq)]
enum FbsRange {
    Strict,
    Negacyclic,
    Extended,
}

impl FbsRange {
    // Return the command-line name of the range.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Strict => "p",
            Self::Negacyclic => "2p",
            Self::Extended => "ext",
        }
    }
}

/// Validated command-line settings for one mapping run.
#[derive(Debug)]
struct Arguments {
    filename: PathBuf,
    input_type: String,
    mapper: String,
    fbs_size: usize,
    fbs_range: FbsRange,
    max_tt_size: usize,
    max_norm2: Option<u128>,
    trials: usize,
    exclude_input_order: bool,
    seed: u64,
    verify_samples: usize,
    multioutput_fbs: bool,
    verbosity: usize,
    output: Option<PathBuf>,
    output_lbf: Option<PathBuf>,
}

impl Arguments {
    // Compute the configured maximum LUT span.
    fn max_fbs_size(&self) -> Result<usize, String> {
        match self.fbs_range {
            FbsRange::Strict => Ok(self.fbs_size),
            FbsRange::Negacyclic => self
                .fbs_size
                .checked_mul(2)
                .ok_or_else(|| "FBS size is too large".to_owned()),
            FbsRange::Extended => Ok(i64::MAX as usize),
        }
    }
}

// Initialize logging, run the mapper, and translate errors to an exit status.
fn main() -> ExitCode {
    init_logger();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

// Parse inputs, run mapping trials, verify the result, and emit outputs.
fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    log::set_max_level(verbosity_level(arguments.verbosity));
    let bit_circuit = parse_file(&arguments.filename, &arguments.input_type)?;
    let max_fbs_size = arguments.max_fbs_size()?;
    let coefficient_policy = match arguments.mapper.as_str() {
        "basic" => CoefficientPolicy::Basic,
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

    let include_input_order = !arguments.exclude_input_order;
    let mapping_count = arguments
        .trials
        .checked_add(usize::from(include_input_order))
        .ok_or_else(|| "too many mapping trials".to_owned())?;
    let mut best = None;
    let mut total_duration = 0.0;
    for trial in 0..mapping_count {
        let schedule_policy = trial_schedule(trial, arguments.seed, include_input_order);
        log::info!(
            target: "mapper",
            "trial {}/{} schedule={schedule_policy:?}",
            trial + 1,
            mapping_count
        );
        let mapper = Mapper::new(MapConfig {
            schedule_policy,
            ..base_config.clone()
        })?;
        let start = Instant::now();
        let mapped = mapper.map(&bit_circuit)?;
        total_duration += start.elapsed().as_secs_f64();
        let replace_best = best.as_ref().is_none_or(|current: &LutCircuit| {
            mapped.stats().quality_key() < current.stats().quality_key()
        });
        if log::log_enabled!(target: "mapper", log::Level::Info) {
            log::info!(
                target: "mapper",
                "trial {}/{} result={:?} selected={replace_best}",
                trial + 1,
                mapping_count,
                mapped.stats()
            );
        }
        if replace_best {
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
        "{{'nb_inp': {}, 'nb_out': {}, 'nb_linprod': {}, 'nb_bootstrap': {}, 'execution_fbs_size': {}, \
         'max_lut_size': {}, 'norm2_linprod': {}, 'filename': '{}', 'fbs_size': {}, \
         'mapper': '{}', 'multioutput_fbs': {}, 'fbs_range': '{}', 'max_tt_size': {}, \
         'max_norm2': {}, 'trials': {}, 'exclude_input_order': {}, 'seed': {}, 'time': {:.9}}}",
        stats.input_count,
        stats.output_count,
        stats.linear_count,
        stats.bootstrap_count,
        lut_circuit.execution_fbs_size(arguments.fbs_range == FbsRange::Strict),
        stats.max_lut_size,
        stats.max_norm2,
        escape_single_quotes(&arguments.filename.to_string_lossy()),
        arguments.fbs_size,
        arguments.mapper,
        python_bool(arguments.multioutput_fbs),
        arguments.fbs_range.as_str(),
        arguments.max_tt_size,
        arguments
            .max_norm2
            .map(|value| value.to_string())
            .unwrap_or_else(|| "None".to_owned()),
        arguments.trials,
        python_bool(arguments.exclude_input_order),
        arguments.seed,
        total_duration,
    );
    Ok(())
}

// Read a circuit and parse it using the requested or inferred format.
fn parse_file(path: impl AsRef<Path>, input_type: &str) -> Result<BoolCircuit, String> {
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
        "blif" => BoolCircuit::parse_blif(&source),
        "bench" => BoolCircuit::parse_bench(&source),
        "bristol" => BoolCircuit::parse_bristol(&source),
        value => Err(format!("unknown input type '{value}'")),
    };
    circuit.map_err(|error| format!("{}: {error}", path.display()))
}

// Use the input order first when enabled, then deterministic random orders.
fn trial_schedule(trial: usize, seed: u64, include_input_order: bool) -> SchedulePolicy {
    if include_input_order && trial == 0 {
        SchedulePolicy::Original
    } else {
        SchedulePolicy::Random {
            seed: seed.wrapping_add((trial - usize::from(include_input_order)) as u64),
        }
    }
}

// Parse command-line arguments supplied by the process.
fn parse_arguments() -> Result<Arguments, String> {
    parse_arguments_from(env::args().skip(1))
}

// Parse and validate command-line arguments from an arbitrary iterator.
fn parse_arguments_from(mut arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let mut filename = None;
    let mut mapper = "search".to_owned();
    let mut input_type = "auto".to_owned();
    let mut fbs_size = 3usize;
    let mut fbs_range = FbsRange::Extended;
    let mut max_tt_size = 16usize;
    let mut max_norm2 = None;
    let mut trials = 0usize;
    let mut exclude_input_order = false;
    let mut seed = 42u64;
    let mut verify_samples = 1000usize;
    let mut multioutput_fbs = false;
    let mut verbosity = 0usize;
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
            "--fbs-range" | "--fbs_range" => {
                let value = next_value(&mut arguments, &argument)?;
                fbs_range = match value.as_str() {
                    "p" => FbsRange::Strict,
                    "2p" => FbsRange::Negacyclic,
                    "ext" => FbsRange::Extended,
                    _ => {
                        return Err(format!(
                            "unknown FBS range '{value}'; expected p, 2p, or ext"
                        ));
                    }
                };
            }
            "--max-tt-size" | "--max_tt_size" => {
                max_tt_size = parse_value(&mut arguments, &argument)?
            }
            "--max-norm2" | "--max_norm2" => {
                max_norm2 = Some(parse_value(&mut arguments, &argument)?)
            }
            "--trials" => trials = parse_value(&mut arguments, &argument)?,
            "--exclude-input-order" | "--exclude_input_order" => exclude_input_order = true,
            "--seed" => seed = parse_value(&mut arguments, &argument)?,
            "--verify-samples" => verify_samples = parse_value(&mut arguments, &argument)?,
            "--multioutput-fbs" | "--multioutput_fbs" => multioutput_fbs = true,
            "-v" | "--verbose" => verbosity += 1,
            value
                if value.len() > 2
                    && value.starts_with('-')
                    && value[1..].chars().all(|character| character == 'v') =>
            {
                verbosity += value.len() - 1;
            }
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

    if exclude_input_order && trials == 0 {
        return Err("--exclude-input-order requires at least one random trial".to_owned());
    }
    if !matches!(mapper.as_str(), "basic" | "search") {
        return Err(format!("unknown mapper '{mapper}'"));
    }

    Ok(Arguments {
        filename: filename.ok_or_else(|| "missing input circuit (try --help)".to_owned())?,
        input_type,
        mapper,
        fbs_size,
        fbs_range,
        max_tt_size,
        max_norm2,
        trials,
        exclude_input_order,
        seed,
        verify_samples,
        multioutput_fbs,
        verbosity,
        output,
        output_lbf,
    })
}

// Consume the value following an option or report it as missing.
fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

// Parse a typed option value from the next command-line argument.
fn parse_value<T: std::str::FromStr>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    let value = next_value(arguments, option)?;
    value
        .parse()
        .map_err(|_| format!("invalid value '{value}' for {option}"))
}

// Print command usage and supported options.
fn print_help() {
    println!(
        r#"Map a Boolean circuit to TFHE functional bootstraps.

Usage: fbs-map [OPTIONS] <FILE>

Options:
  --mapper <NAME>          basic or search (default: search)
  --type <FORMAT>          auto, blif, bench, or bristol (default: auto)
  --fbs-size <P>           FBS plaintext size (default: 3)
  --fbs-range <MODE>       p, 2p, or ext (default: ext)
  --max-tt-size <N>        Maximum cone support before forced bootstrap (default: 16)
  --max-norm2 <N>          Maximum sum of squared linear-combination coefficients
  --trials <N>             Add N random-order mappings (default: 0)
  --exclude-input-order    Skip the input-order mapping
  --seed <N>               Random-order and verification seed (default: 42)
  --verify-samples <N>     Random equivalence samples above 16 inputs (default: 1000)
  --multioutput-fbs        Group bootstrap outputs with the same linear input
  -v, --verbose            Increase logging from Warn to Info, Debug, then Trace
  --output <FILE>          Write the human-readable FBS circuit
  --output-lbf <FILE>      Write an LBF circuit
  -h, --help               Show this help
"#
    );
}

// Verify mapping equivalence exhaustively or with deterministic samples.
fn verify(
    bit_circuit: &BoolCircuit,
    lut_circuit: &LutCircuit,
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
            let BoolNode::Input { name } = &bit_circuit.nodes[*id] else {
                unreachable!()
            };
            state = xorshift64(state);
            inputs.insert(name.clone(), state & 1 != 0);
        }
        compare_once(bit_circuit, lut_circuit, &inputs)?;
    }
    Ok(())
}

// Compare both circuits on one named input assignment.
fn compare_once(
    bit_circuit: &BoolCircuit,
    lut_circuit: &LutCircuit,
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

// Advance a deterministic nonzero xorshift pseudorandom state.
fn xorshift64(mut value: u64) -> u64 {
    if value == 0 {
        value = 0x4d59_5df4_d0f3_3173;
    }
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

// Format a Boolean using Python literal spelling.
fn python_bool(value: bool) -> &'static str {
    if value { "1" } else { "0" }
}

// Escape a value embedded in a single-quoted Python-style field.
fn escape_single_quotes(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trials_use_input_order_then_distinct_random_seeds() {
        assert_eq!(trial_schedule(0, 42, true), SchedulePolicy::Original);
        assert_eq!(
            trial_schedule(1, 42, true),
            SchedulePolicy::Random { seed: 42 }
        );
        assert_eq!(
            trial_schedule(2, 42, true),
            SchedulePolicy::Random { seed: 43 }
        );
        assert_eq!(
            trial_schedule(0, 42, false),
            SchedulePolicy::Random { seed: 42 }
        );
        assert_eq!(
            trial_schedule(1, 42, false),
            SchedulePolicy::Random { seed: 43 }
        );
    }

    #[test]
    fn excluding_input_order_requires_a_random_trial() {
        let error = parse_arguments_from(
            ["--exclude-input-order", "circuit.blif"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert_eq!(
            error,
            "--exclude-input-order requires at least one random trial"
        );

        let arguments = parse_arguments_from(
            ["--trials", "2", "--exclude-input-order", "circuit.blif"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(arguments.trials, 2);
        assert!(arguments.exclude_input_order);
    }

    #[test]
    fn lut_span_range_selects_the_expected_limit() {
        let parse = |args: &[&str]| {
            parse_arguments_from(args.iter().map(|argument| (*argument).to_owned())).unwrap()
        };
        assert_eq!(
            parse(&["circuit.blif"]).max_fbs_size().unwrap(),
            i64::MAX as usize
        );
        assert_eq!(
            parse(&["--fbs-size", "3", "--fbs-range", "2p", "circuit.blif"])
                .max_fbs_size()
                .unwrap(),
            6
        );
        assert_eq!(
            parse(&["--fbs-size", "3", "--fbs-range", "p", "circuit.blif"])
                .max_fbs_size()
                .unwrap(),
            3
        );
        assert_eq!(
            parse(&["--fbs-range", "ext", "circuit.blif"])
                .max_fbs_size()
                .unwrap(),
            i64::MAX as usize
        );
        assert!(
            parse_arguments_from(
                ["--fbs-range", "invalid", "circuit.blif"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn removed_mapper_is_rejected() {
        let error = parse_arguments_from(
            ["--mapper", "naive", "circuit.blif"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();
        assert_eq!(error, "unknown mapper 'naive'");
    }

    #[test]
    fn verbose_flags_select_standard_logging_levels() {
        let compact =
            parse_arguments_from(["-vvv", "circuit.blif"].into_iter().map(str::to_owned)).unwrap();
        assert_eq!(compact.verbosity, 3);

        let repeated = parse_arguments_from(
            ["--verbose", "-v", "--verbose", "circuit.blif"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(repeated.verbosity, 3);
        assert_eq!(verbosity_level(0), log::LevelFilter::Warn);
        assert_eq!(verbosity_level(1), log::LevelFilter::Info);
        assert_eq!(verbosity_level(2), log::LevelFilter::Debug);
        assert_eq!(verbosity_level(compact.verbosity), log::LevelFilter::Trace);
        assert_eq!(verbosity_level(usize::MAX), log::LevelFilter::Trace);
    }
}
