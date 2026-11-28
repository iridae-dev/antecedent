//! The execution trace that links evidence to the units it ran.
//!
//! With the test-only `evidence-trace` feature and `ANTECEDENT_EVIDENCE_TRACE=<file>`, every
//! support decision appends one tab-separated line to `<file>`:
//! `unit <binary> <thread> <kind> <record id> <status>`. A test thread is named after its test,
//! so `scripts/evidence_links.py` maps each scientific pass and calibration record to the units
//! its test executed. Binding a result to its compiled route appends
//! `route <binary> <thread> <route> <kind> <record id> [<kind> <record id>]...`: the unit the
//! result carries (`- -` for none), then every unit decided on the thread since the previous
//! route line, refusals included. Without the feature this compiles to nothing.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cell::CellStatus;
use super::unit::UnitRef;

/// The environment variable naming the trace file.
pub const TRACE_ENV: &str = "ANTECEDENT_EVIDENCE_TRACE";

/// A thread named after the thread that spawns it, so the evidence trace attributes the work
/// it runs to the calling test. Every thread a test or a route spawns is built from this.
#[doc(hidden)]
pub fn caller_thread() -> std::thread::Builder {
    let builder = std::thread::Builder::new();
    match std::thread::current().name() {
        Some(name) => builder.name(name.to_owned()),
        None => builder,
    }
}

/// Record that the unit `unit` was decided with `status` on this thread.
#[inline]
pub(crate) fn executed(unit: &UnitRef, status: &CellStatus) {
    #[cfg(feature = "evidence-trace")]
    executed_record(unit.kind(), &unit.record_id(), status);
    #[cfg(not(feature = "evidence-trace"))]
    let _ = (unit, status);
}

/// [`executed`] for a unit known only by its kind and record id, such as a capability id the
/// embedded status does not list.
#[inline]
pub(crate) fn executed_record(kind: &str, id: &str, status: &CellStatus) {
    #[cfg(feature = "evidence-trace")]
    if let Some(path) = std::env::var_os(TRACE_ENV) {
        write_unit(std::path::Path::new(&path), kind, id, status);
    }
    #[cfg(not(feature = "evidence-trace"))]
    let _ = (kind, id, status);
}

#[cfg(feature = "evidence-trace")]
fn write_unit(path: &std::path::Path, kind: &str, id: &str, status: &CellStatus) {
    write_line(path, &format!("unit\t{}\t{kind}\t{id}\t{}", origin(), status.as_str()));
    decided().entry(test_name()).or_default().push(format!("{kind}\t{id}"));
}

/// Units decided since the test's last route line, by test: a worker thread carries its
/// caller's name, so its decisions reach the caller's route line, and no test sees another's.
#[cfg(feature = "evidence-trace")]
fn decided() -> std::sync::MutexGuard<'static, std::collections::HashMap<String, Vec<String>>> {
    static DECIDED: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>,
    > = std::sync::OnceLock::new();
    DECIDED.get_or_init(Default::default).lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Record that the compiled route `route` produced a result decided by `unit` (`- -` when none),
/// with the units decided on this thread since the previous route line.
pub(crate) fn route_ran(route: &str, unit: Option<&UnitRef>) {
    #[cfg(feature = "evidence-trace")]
    if let Some(path) = std::env::var_os(TRACE_ENV) {
        write_route(std::path::Path::new(&path), route, unit);
    }
    #[cfg(not(feature = "evidence-trace"))]
    let _ = (route, unit);
}

#[cfg(feature = "evidence-trace")]
fn write_route(path: &std::path::Path, route: &str, unit: Option<&UnitRef>) {
    let (kind, id) =
        unit.map_or_else(|| ("-", "-".to_owned()), |unit| (unit.kind(), unit.record_id()));
    let mut line = format!("route\t{}\t{route}\t{kind}\t{id}", origin());
    let mut units = decided().remove(&test_name()).unwrap_or_default();
    units.sort_unstable();
    units.dedup();
    for unit in units {
        line.push('\t');
        line.push_str(&unit);
    }
    write_line(path, &line);
}

/// Record that a result decided by `unit` reported its interval with `method` (`none` for no
/// interval), so a unit stating it reports none is checked against what it returned.
pub(crate) fn interval_reported(unit: &UnitRef, method: &str) {
    #[cfg(feature = "evidence-trace")]
    if let Some(path) = std::env::var_os(TRACE_ENV) {
        write_line(
            std::path::Path::new(&path),
            &format!("interval\t{}\t{}\t{}\t{method}", origin(), unit.kind(), unit.record_id()),
        );
    }
    #[cfg(not(feature = "evidence-trace"))]
    let _ = (unit, method);
}

/// Record the typed answer of a result, or mark a shape without a complete digest.
#[cfg(feature = "evidence-trace")]
pub(crate) fn output_produced(values: impl FnOnce() -> Option<Vec<u8>>) {
    if let Some(path) = std::env::var_os(TRACE_ENV) {
        write_output(std::path::Path::new(&path), values().as_deref());
    }
}

#[cfg(feature = "evidence-trace")]
fn write_output(path: &std::path::Path, values: Option<&[u8]>) {
    let digest = values
        .map_or_else(|| "unsupported".into(), |values| blake3::hash(values).to_hex().to_string());
    write_line(path, &format!("output-v2\t{}\t{digest}", origin()));
}

/// Declare, for a test that executes several units, a unit `kind`/`id` its assertions are
/// about. Reached through [`crate::evidence_unit!`].
#[doc(hidden)]
pub fn declare_unit(kind: &str, id: &str) {
    #[cfg(feature = "evidence-trace")]
    if let Some(path) = std::env::var_os(TRACE_ENV) {
        write_line(std::path::Path::new(&path), &format!("declare\t{}\t{kind}\t{id}", origin()));
    }
    #[cfg(not(feature = "evidence-trace"))]
    let _ = (kind, id);
}

/// Declare an evidence unit a test that executes several units asserts about, once per unit:
/// `evidence_unit!("matrix", "AverageEffect/Dag/explicit/Frequentist/none")`. The kind is one of
/// `matrix`, `construction`, `graphless` or `capability`, and the id is the unit's record id
/// ([`crate::support::UnitRef::record_id`]). This is the only hand-written evidence link;
/// `scripts/evidence_links.py` rejects a declaration of a unit the test did not execute.
#[macro_export]
macro_rules! evidence_unit {
    ($kind:literal, $id:expr) => {
        $crate::support::declare_unit($kind, $id)
    };
}

/// `<binary>\t<test>`: the test binary's target name and the running test.
#[cfg(feature = "evidence-trace")]
fn origin() -> String {
    format!("{}\t{}", binary_name(), test_name())
}

/// The running test: the name of the thread, which libtest names after its test and
/// [`caller_thread`] passes on.
#[cfg(feature = "evidence-trace")]
fn test_name() -> String {
    std::thread::current().name().unwrap_or("").to_owned()
}

/// The test binary's target name: its file stem without Cargo's `-<16 hex>` suffix.
#[doc(hidden)]
#[must_use]
pub fn binary_name() -> String {
    let stem = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.file_stem().map(|stem| stem.to_string_lossy().into_owned()))
        .unwrap_or_default();
    match stem.rsplit_once('-') {
        Some((name, hash))
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
        {
            name.to_owned()
        }
        _ => stem,
    }
}

#[cfg(feature = "evidence-trace")]
fn write_line(path: &std::path::Path, line: &str) {
    // One write per line: concurrent test threads append to the same file.
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes());
    }
}

#[cfg(all(test, feature = "evidence-trace"))]
mod tests {
    use super::*;
    use crate::support::UnitId;

    #[test]
    fn a_line_names_the_test_and_the_decided_unit() {
        let path = std::env::temp_dir()
            .join(format!("antecedent-evidence-trace-{}.tsv", std::process::id()));
        write_unit(&path, "capability", "no.such.unit", &CellStatus::NotSupported);
        let trace = std::fs::read_to_string(&path).expect("the trace was written");
        let _ = std::fs::remove_file(&path);
        let fields: Vec<&str> = trace.trim_end().split('\t').collect();
        assert_eq!(fields[0], "unit");
        assert!(fields[2].ends_with("a_line_names_the_test_and_the_decided_unit"), "{fields:?}");
        assert_eq!(fields[3..], ["capability", "no.such.unit", "not_supported"]);
    }

    #[test]
    fn a_route_line_names_the_units_its_test_decided_on_any_thread() {
        let path = std::env::temp_dir()
            .join(format!("antecedent-evidence-route-{}.tsv", std::process::id()));
        let refused = UnitRef::Capability(UnitId::new("refused.unit"));
        let worker = path.clone();
        caller_thread()
            .spawn(move || {
                write_unit(&worker, "capability", "refused.unit", &CellStatus::NotSupported);
            })
            .unwrap()
            .join()
            .unwrap();
        std::thread::Builder::new()
            .name("another test".into())
            .spawn(|| decided().entry(test_name()).or_default().push("capability\tother".into()))
            .unwrap()
            .join()
            .unwrap();
        write_route(&path, "SomeRoute", None);
        write_route(&path, "SomeRoute", Some(&refused));
        let trace = std::fs::read_to_string(&path).expect("the trace was written");
        let _ = std::fs::remove_file(&path);
        let lines: Vec<Vec<&str>> = trace.lines().map(|line| line.split('\t').collect()).collect();
        assert!(
            lines[0][2].ends_with("a_route_line_names_the_units_its_test_decided_on_any_thread")
        );
        assert_eq!(lines[1][3..], ["SomeRoute", "-", "-", "capability", "refused.unit"]);
        assert_eq!(lines[2][3..], ["SomeRoute", "capability", "refused.unit"]);
    }

    #[test]
    fn an_output_line_is_a_digest_of_the_exact_values() {
        let path = std::env::temp_dir()
            .join(format!("antecedent-evidence-output-{}.tsv", std::process::id()));
        write_output(&path, Some(b"0.5"));
        write_output(&path, Some(b"0.5"));
        write_output(&path, Some(b"0.50000000000000011"));
        write_output(&path, None);
        let trace = std::fs::read_to_string(&path).expect("the trace was written");
        let _ = std::fs::remove_file(&path);
        let lines: Vec<Vec<&str>> = trace.lines().map(|line| line.split('\t').collect()).collect();
        assert_eq!(lines[0][0], "output-v2");
        assert!(lines[0][2].ends_with("an_output_line_is_a_digest_of_the_exact_values"));
        assert_eq!(lines[0][3].len(), 64);
        assert_eq!(lines[0][3], lines[1][3]);
        assert_ne!(lines[0][3], lines[2][3]);
        assert_eq!(lines[3][3], "unsupported");
    }

    #[test]
    fn a_binary_name_drops_only_a_cargo_hash() {
        let name = binary_name();
        assert!(!name.is_empty());
        assert!(name.rsplit_once('-').is_none_or(|(_, hash)| hash.len() != 16));
    }
}
