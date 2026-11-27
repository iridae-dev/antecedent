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

/// Record that the unit `unit` was decided with `status` on this thread.
#[cfg(feature = "evidence-trace")]
pub(crate) fn executed(unit: &UnitRef, status: &CellStatus) {
    if let Some(path) = std::env::var_os(TRACE_ENV) {
        append(std::path::Path::new(&path), unit, status);
        DECIDED.with(|decided| {
            decided.borrow_mut().push(format!("{}\t{}", unit.kind(), unit.record_id()))
        });
    }
}

#[cfg(feature = "evidence-trace")]
thread_local! {
    /// Units decided on this thread since its last route line.
    static DECIDED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record that the unit `unit` was decided with `status` on this thread.
#[cfg(not(feature = "evidence-trace"))]
#[inline]
pub(crate) const fn executed(_unit: &UnitRef, _status: &CellStatus) {}

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
    DECIDED.with(|decided| {
        let mut decided = decided.borrow_mut();
        decided.sort_unstable();
        decided.dedup();
        for unit in decided.drain(..) {
            line.push('\t');
            line.push_str(&unit);
        }
    });
    write_line(path, &line);
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

#[cfg(feature = "evidence-trace")]
fn append(path: &std::path::Path, unit: &UnitRef, status: &CellStatus) {
    write_line(
        path,
        &format!("unit\t{}\t{}\t{}\t{}", origin(), unit.kind(), unit.record_id(), status.as_str()),
    );
}

/// `<binary>\t<thread>`: the test binary's target name and the running test.
#[cfg(feature = "evidence-trace")]
fn origin() -> String {
    let binary = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.file_stem().map(|stem| stem.to_string_lossy().into_owned()))
        .map(|stem| stem.rsplit_once('-').map_or(stem.clone(), |(name, _)| name.to_owned()))
        .unwrap_or_default();
    format!("{binary}\t{}", std::thread::current().name().unwrap_or(""))
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
        append(&path, &UnitRef::Capability(UnitId::new("no.such.unit")), &CellStatus::NotSupported);
        let trace = std::fs::read_to_string(&path).expect("the trace was written");
        let _ = std::fs::remove_file(&path);
        let fields: Vec<&str> = trace.trim_end().split('\t').collect();
        assert_eq!(fields[0], "unit");
        assert!(fields[2].ends_with("a_line_names_the_test_and_the_decided_unit"), "{fields:?}");
        assert_eq!(fields[3..], ["capability", "no.such.unit", "not_supported"]);
    }

    #[test]
    fn a_route_line_names_the_units_decided_since_the_last_one() {
        let path = std::env::temp_dir()
            .join(format!("antecedent-evidence-route-{}.tsv", std::process::id()));
        let refused = UnitRef::Capability(UnitId::new("refused.unit"));
        DECIDED
            .with(|decided| decided.borrow_mut().push(format!("capability\t{}", "refused.unit")));
        write_route(&path, "SomeRoute", None);
        write_route(&path, "SomeRoute", Some(&refused));
        let trace = std::fs::read_to_string(&path).expect("the trace was written");
        let _ = std::fs::remove_file(&path);
        let lines: Vec<Vec<&str>> = trace.lines().map(|line| line.split('\t').collect()).collect();
        assert_eq!(lines[0][3..], ["SomeRoute", "-", "-", "capability", "refused.unit"]);
        assert_eq!(lines[1][3..], ["SomeRoute", "capability", "refused.unit"]);
    }
}
