# Python callback interruption and exception ownership

Source review and corrections cover callback error retention, actual Python API
callers, observer sinks and native worker ownership. [Validation](validation.md)
records the executed native and rebuilt-extension results separately from source
review.

## Defects and implemented lifetime contract

`callback_failure` retained original `PyErr` values in a process-global stash, but
only error attribution removed them. Successful native fallback, an untagged
failure or a superseding signal could leave the original exception/traceback
retained indefinitely. After 256 suppressed failures, later ordinary callbacks
could lose their original cause. Formatting a Python exception while holding the
stash mutex also allowed user `__str__` code to reenter the callback bridge and
deadlock.

Each native run's `Scope` now owns its callback ids from worker launch through
join and attribution. Scope destruction removes its remaining entries after any
outcome, including fallback, signal override and caught panic. Attribution removes
the chosen original first, preserving the returned exception/cause independently.
`callback_failure` captures the active `Arc<Scope>`, formats ordinary exception
text before locking the stash, inserts under a short lock, then registers the id
with its owner. No two locks are held together. Cleanup removes entries while
locked and drops the removed Python objects afterward. Python formatting,
normalization and finalizers cannot execute under the stash mutex.

Attribution receives the actual run's `Scope` and snapshots its owned ids after
formatting the domain error. It removes only the last live owned tag: an outer
callback appends its tag after nested `__str__` output. A foreign live tag cannot
consume another run's exception. The redundant unscoped attribution after design
argument parsing is removed; that phase only constructs utility wrappers and
preserves its raw parsing `PyErr`. Unscoped failures have no retention owner and
therefore keep readable diagnostics without stashing an exception.

If nested formatting occupies the final stash slot, the outer failure cannot be
retained. Its description neutralizes private lookup tag openers before returning,
so the nested original cannot masquerade as the unretained outer cause. The same
transformation protects unscoped descriptions. The capacity boundary and refusal
transformation are tested separately; a real run exercises transformed nested
text, absent attribution and owned-id cleanup without filling process-global
state and racing other native tests.

Unique ids still distinguish concurrent errors with identical text. The 256-entry
cap protects genuinely in-flight errors without evicting an earlier unconsumed
one. Each scope cleans only its own ids. Ordinary exceptions retain their original
cause and traceback; actual signal precedence remains unchanged.

## Actual callers and scheduling inventory

All production `callback_failure` sites were enumerated. Ownership follows the
actual worker, rather than guessing from globally active runs.

| Bridge | Actual API and scheduling disposition |
| --- | --- |
| `PyDynamicMechanism`: introspection, RNG, sampling, evaluation, noise inference and log probability | `attribution_api.rs::sample_do` constructs wrappers inside `detach_catch` and forces one thread when wrappers exist. Model sampling/replay executes synchronous node loops, with no native callback pool. The local wrapped model is not exposed for unrelated later parallel execution. Lazy `accepts_rng` introspection is therefore scoped too. |
| `PyConditionalIndependence::test_batch` | All eight discovery/PCMCI/posterior `resolve_ci_arg` paths force one thread for callable CI before `detach_catch`. The execution map's serial branch runs on that interrupt worker. |
| `PyUtility::evaluate_batch` | Design ranking prepares its utility decision table serially before parallel candidate scoring; workers read prepared gaps/kernels. Direct `evaluate_decision_py` now also runs through `detach_catch`, retaining its `CausalDesignError` mapping. |
| `PyCustomValidator` | Prepared execution enters `detach_catch`; custom validators run serially. Ordinary errors remain failed-validation evidence, while the independent interruption slot prevents report conversion from swallowing `BaseException`. |
| `PyBatchCallback::call` | Fitted-GCM sample, noise and replay APIs use `detach_catch` and synchronous model loops. Policy-program encode/decode only constructs metadata and does not invoke the callback. |
| `PythonLocalMechanism::evaluate` | Local cyclic solve/residual loops are synchronous. `execute_local_cyclic` now uses `detach_catch` instead of raw `py.detach`, preserving the existing `CausalValueError` mapping and original cause/interrupt. |
| `PyStageResultSink::on_stage` | Per-click controls construct the sink; estimate/click enters `detach_catch`; stage emission is synchronous. |
| `PyProgressSink::report` | Serial coordinators have run ownership. Parallel bootstrap workers call an explicitly scope-bound adapter, described below. |

New parallel callback APIs must explicitly carry ownership or preserve these
guarded serial calling paths. Ambient ownership is not implicitly inherited by a
native pool thread.

## Interruptions cannot become successful reports

Native custom validation intentionally turns ordinary callback errors into failed
report entries. That previously swallowed `KeyboardInterrupt` and `SystemExit`
as well. `Scope` now has an independent bounded first-`BaseException` slot.
`callback_failure` detects non-`Exception` outside all mutexes, retains the first
original, cancels the run's own and linked context tokens, and returns diagnostic
text without competing for ordinary stash capacity. After worker join, the run
returns that original even if native code recovered or returned a report. An
actual watcher-observed signal keeps priority.

Attribution returns a direct `BaseException` before invoking its string conversion
or interpreting callback tags. A later interruption cannot replace the first;
superseded Python objects are dropped outside the lock. Ordinary validator errors
keep their failed-report semantics and unused stash entries are scope-cleaned.
This correction changes interruption handling, not scientific output or ordinary
validation evidence.

Both observer sinks keep print-and-continue behavior for ordinary `Exception`.
Their non-`Exception` branches instead call `callback_failure`, so a progress or
stage interruption uses the same retention/cancellation contract.

## Parallel progress has an explicit run owner

`interrupt::bind_progress` wraps `ctx.progress` when
`lib.rs::py_execution_context_ext` installs caller progress. The adapter owns the
raw sink and a `Weak<Scope>`. For each callback it upgrades that precise owner,
installs it in worker-local `ACTIVE` with a restoring RAII guard, invokes the sink
without ownership mutexes held, and restores prior state on return or unwind.
An expired adapter skips stale callbacks; retaining a context/adapter after a run
cannot retain its scope or Python errors. No global active-run inference or native
core scheduling change is involved.

The constructor audit found exactly two progress-bearing producers:
`ClickControls::ctx` and `PreparedOptions::ctx`. Click controls build their context
inside `click`/`click_bound` detach closures. Preparation builds it inside
`PrepareRequest::off_gil`, which delegates to `detach_catch`, including posterior
preparation and compile; observation preparation also uses `detach_catch`.
Other direct context-constructor sites for inspect/export/retarget/refute pass no
progress sink. Production progress contexts therefore belong to the current run.
Reusing a raw Python observer on another click creates a new adapter for that run.
The existing cancellation pending queue still handles pre-run contexts without
progress; no mutable pending-progress registry is required.

## Implemented regressions and fixture isolation

Native regression scope includes:

- `successful_fallbacks_release_callback_exceptions_after_every_run`: more than
  the stash capacity of suppressed failures, checking each scope's id is removed
  immediately after return.
- `callback_base_exceptions_survive_and_superseded_failures_are_released`: exact
  original `KeyboardInterrupt`/`SystemExit` identity and cleanup of a superseded
  ordinary error.
- `concurrent_identical_callback_text_keeps_each_runs_original_cause`: simultaneous
  interrupt workers retain their own original causes despite identical text.
- `exception_formatting_can_reenter_callback_failure_without_holding_the_stash_lock`:
  user exception formatting returns a nested tagged Rust callback error without
  deadlock; attribution preserves the exact outer cause and cleans both ids.
- `a_domain_message_cannot_consume_another_live_runs_callback_tag`: two real
  overlapping runs preserve foreign-scope isolation and the owner's original.
  Rendezvous release precedes assertions, so a regression fails without hanging.
- `an_unretained_outer_failure_cannot_attribute_its_nested_callbacks_original`
  and `unscoped_callback_failures_have_no_unowned_retention_tag`: diagnostic
  text cannot expose a nested lookup tag when no outer original is retained.
- `native_success_cannot_suppress_a_callback_interrupt`: native success cannot
  swallow either interruption; own/linked tokens cancel and the first original
  wins.
- `native_progress_workers_preserve_each_runs_interrupt_and_restore_their_scope`:
  concurrent runs use actual execution-pool workers, assert no ambient owner
  before/after the adapter, retain each original object and reuse raw observers
  in a second execution with fresh run binding.
- `retained_progress_adapter_does_not_retain_a_completed_run`: the weak owner
  expires and later reporting does not call the stale observer.

Existing parsing-id, capacity non-eviction, pending-context and signal-priority
coverage remains. Tests inspect their own ids and object identities without
clearing the global stash or asserting unrelated process-wide counts. Native
concurrency initializes CPython before spawning, attaches each caller separately,
and releases the GIL while waiting; rendezvous do not hold a callback/GIL.

The concurrent progress fixture initially used `PyModule::from_code` with the
same module name in both callers. `PyImport_ExecCodeModuleEx` reused its
`sys.modules` dictionary, letting the second definition overwrite the first
callback's exception global. A standalone C-API probe confirmed module reuse and
that the first callback then raised the second original. The corrected fixture
executes each definition in a fresh private `PyDict`, without module registration,
and clears those globals after joined executions. It preserves actual pool/thread,
concurrent identity, rendezvous and observer-reuse assertions. This fixes fixture
isolation rather than relaxing attribution requirements.

Python integration regression scope includes:

- `test_decision_utility_preserves_original_callback_exception`: domain-error cause
  and original interruption identity/traceback through the decision API.
- `test_custom_validator_report_cannot_swallow_interrupts`: actual prepared
  native-ATE execution preserves ordinary failure evidence but reraises original
  interruptions, with proof the callback ran.
- `test_local_cyclic_preserves_original_callback_exception`: actual cyclic
  execution retains the ordinary cause and original interruptions.
- `test_execution_observers_print_ordinary_errors_but_preserve_interrupts`:
  prepared estimate with bootstrap 40 and four threads exercises progress/stage
  observers, ordinary stderr plus finite results, and original interruption
  identity/traceback. The native worker test independently guarantees parallel
  adapter dispatch.

## Verification and compatibility boundary

Native `interrupt::tests`, Python callback/cyclic workflows and actual SIGINT
integration are verified separately in [the execution record](validation.md).
No process-wide signal is injected into concurrent native unit tests. Extension
freshness is checked against the modified Rust sources before Python execution;
previous-binary results do not establish current-source success.

The changes preserve scientific identity, encoding, artifact layout and numerical
behavior. Error/resource ownership is explicit: run scopes own pending Python
errors, weak adapters do not prolong completed runs, and linked cancellation
reaches the native context. This record makes no claim of byte-admitting arbitrary
Python exception graphs or all callback payload allocations. No goldens were
repinned and no blanket lint allowance was introduced for these corrections.
