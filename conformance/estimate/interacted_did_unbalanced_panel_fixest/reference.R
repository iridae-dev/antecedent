# Independent public black-box reference for the interacted (Sun-Abraham) difference in
# differences on the retained unbalanced panel: fixest::feols(y ~ sunab(cohort, period) |
# unit + period).  No product code and no fixest source or test file is read; fixest is
# isolated tooling that is never used by the product or ordinary CI.  Run prepare.R first.
#
#   Rscript --vanilla conformance/estimate/interacted_did_unbalanced_panel_fixest/reference.R --check [--self-test]
#   Rscript --vanilla ... reference.R --write     (regenerate the frozen outputs; keeps the review block)
#
# The panel is panel.csv (written by export_panel.py from the committed numpy fixture's
# simulator).  The comparison target is the committed numpy full-dummy fixture
# ../interacted_did_unbalanced_panel/expected.json, whose cell estimates, aggregates and
# panel the Rust test of the product asserts.  Nothing here imports the product.
# SPDX-License-Identifier: MIT OR Apache-2.0
file_arg <- grep("^--file=", commandArgs(), value=TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
root <- normalizePath(file.path(here, "../../.."))
library_path <- Sys.getenv("ANTECEDENT_R_ORACLE_LIBRARY", file.path(root, "target/reference-environments/r-oracles/fixest/library"))
if (!dir.exists(library_path)) stop("Missing isolated library; run this fixture's prepare.R first")
.libPaths(c(normalizePath(library_path), .Library), include.site=FALSE)
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("Missing locked jsonlite; run prepare.R")
lock <- jsonlite::read_json(file.path(here, "renv.lock"), simplifyVector=FALSE)
if (!identical(as.character(getRversion()), lock$R$Version)) stop("wrong R version")
for (name in names(lock$Packages)) {
    actual <- tryCatch(utils::packageDescription(name, fields="Version"), error=function(e) NA_character_)
    if (!identical(actual, lock$Packages[[name]]$Version)) stop(paste("wrong package version:", name))
}
args <- commandArgs(trailingOnly=TRUE)
if (!length(args) || any(!args %in% c("--check", "--self-test", "--write"))) stop("Usage: (--check [--self-test] | --write)")
suppressPackageStartupMessages(library(fixest))
setFixest_nthreads(1)

# Pinned specification, independent of the frozen fixture so a changed convention is reported.
ESTIMATE_TOLERANCE <- 1e-8   # fixest cell estimate vs the committed numpy reference (the product test's own bound)
FROZEN_TOLERANCE <- 1e-9     # a regenerated value vs the frozen record
STANDARD_ERROR_TOLERANCE <- 1e-8
FIXEF_TOL <- 1e-11
SPEC <- list(
    formula="outcome ~ sunab(cohort_inf, period) | unit + period",
    never_treated_coding="cohort 0 is passed to sunab as Inf",
    reference_period="sunab default ref.p = c(-1, -Inf): the period before adoption of each cohort",
    vcov="~unit (cluster-robust by unit, CRV1)",
    ssc="fixest default: adj=TRUE, fixef.K=nonnested, cluster.adj=TRUE, cluster.df=min, t.df=min",
    fixef_tol=FIXEF_TOL,
    preprocessing="rows at periods in which no never-treated unit is observed are removed before fitting (a treated cell at such a period is collinear with the period effect)"
)

panel_path <- file.path(here, "panel.csv")
numpy_path <- file.path(root, "conformance/estimate/interacted_did_unbalanced_panel/expected.json")
expected_path <- file.path(here, "expected.json")

fit_fixest <- function() {
    d <- read.csv(panel_path, stringsAsFactors=FALSE, colClasses=c("character", "integer", "integer", "numeric"))
    units_per_cohort <- tapply(d$unit, d$cohort, function(u) length(unique(u)))
    never_periods <- sort(unique(d$period[d$cohort == 0]))
    supported <- d$period %in% never_periods
    dropped <- tapply(!supported, d$cohort, sum)
    used <- d[supported, ]
    used$cohort_inf <- ifelse(used$cohort == 0, Inf, used$cohort)
    model <- feols(outcome ~ sunab(cohort_inf, period) | unit + period, used,
                   vcov=~unit, fixef.tol=FIXEF_TOL, nthreads=1)
    stopifnot(length(model$collin.var) == 0)
    cell <- summary(model, agg=FALSE, vcov=~unit)
    cell_iid <- summary(model, agg=FALSE, vcov="iid")
    covariance <- vcov(model, agg=FALSE, vcov=~unit)
    parts <- regmatches(rownames(cell$coeftable), regexec("^period::(-?[0-9]+):cohort::([0-9]+)$", rownames(cell$coeftable)))
    event <- as.integer(vapply(parts, function(p) p[[2]], ""))
    cohort <- as.integer(vapply(parts, function(p) p[[3]], ""))
    order_index <- order(cohort, event)
    estimate <- cell$coeftable[, 1]
    cells <- lapply(order_index, function(i) list(
        cohort=cohort[i], period=cohort[i] + event[i], event_time=event[i],
        estimate=unname(estimate[i]),
        std_error_cluster_unit=unname(cell$coeftable[i, 2]),
        std_error_iid=unname(cell_iid$coeftable[i, 2])))
    # The product's aggregation: cohort-size weights (units per cohort), simple over the
    # post-adoption cells and event-time means over the pre-adoption cells.
    size <- as.numeric(units_per_cohort[as.character(cohort)])
    aggregate <- function(members) {
        w <- numeric(length(estimate)); w[members] <- size[members] / sum(size[members])
        list(estimate=sum(w * estimate), std_error_cluster_unit=sqrt(drop(t(w) %*% covariance %*% w)))
    }
    post <- which(event >= 0)
    leads <- list()
    for (e in sort(unique(event[event < 0]))) leads[[as.character(e)]] <- aggregate(which(event == e))
    # fixest's own aggregation, weighted by the observed cell counts.
    att <- summary(model, agg="att", vcov=~unit)$coeftable
    by_event <- summary(model, vcov=~unit)$coeftable
    fixest_event <- list()
    for (i in seq_len(nrow(by_event))) {
        e <- sub("^period::", "", rownames(by_event)[i])
        fixest_event[[e]] <- list(estimate=unname(by_event[i, 1]), std_error_cluster_unit=unname(by_event[i, 2]))
    }
    list(
        input=list(rows=nrow(d), rows_used=nrow(used), dropped_rows_by_cohort=as.list(dropped[names(dropped) != "0"]),
                   periods_without_comparison=as.list(setdiff(sort(unique(d$period)), never_periods)),
                   units_per_cohort=as.list(units_per_cohort)),
        cells=cells,
        product_convention=list(simple=aggregate(post), leads_by_event=leads),
        fixest_convention=list(att=list(estimate=unname(att[1, 1]), std_error_cluster_unit=unname(att[1, 2])),
                               event_time=fixest_event))
}

close_to <- function(a, b, tolerance) is.numeric(a) && is.numeric(b) && length(a) == 1 && length(b) == 1 && abs(a - b) <= tolerance

# Compare a (possibly altered) frozen document with the regenerated result and the committed
# numpy reference; return the list of problems.
check <- function(document, actual, numpy) {
    problems <- character()
    oracle <- document$oracle
    if (!identical(oracle$kind, "external_package")) problems <- c(problems, "incorrect oracle origin")
    versions <- vapply(oracle$reference, function(r) paste(r$project, r$version), "")
    if (!"fixest 0.14.2" %in% versions) problems <- c(problems, "missing pinned fixest comparator")
    if (!identical(document$specification, SPEC)) problems <- c(problems, "specification or vcov convention changed")
    for (key in c("rows", "rows_used")) {
        if (!identical(as.integer(document$input[[key]]), as.integer(actual$input[[key]]))) problems <- c(problems, paste("input", key, "differs"))
    }
    if (!identical(unlist(document$input$periods_without_comparison), as.integer(actual$input$periods_without_comparison))) problems <- c(problems, "periods without comparison differ")
    if (!identical(lapply(document$input$dropped_rows_by_cohort, as.integer), lapply(actual$input$dropped_rows_by_cohort, as.integer))) problems <- c(problems, "dropped rows differ")
    if (!isTRUE(all.equal(as.numeric(unlist(document$tolerances)), c(ESTIMATE_TOLERANCE, FROZEN_TOLERANCE, STANDARD_ERROR_TOLERANCE)))) problems <- c(problems, "tolerances changed")
    frozen_cells <- document$reference$cells
    if (length(frozen_cells) != length(actual$cells)) {
        return(c(problems, "cell count differs"))
    }
    for (i in seq_along(actual$cells)) {
        a <- actual$cells[[i]]; f <- frozen_cells[[i]]
        tag <- sprintf("cell (%d, %d)", a$cohort, a$period)
        if (!identical(as.integer(f$cohort), a$cohort) || !identical(as.integer(f$period), a$period)) { problems <- c(problems, paste(tag, "identity differs")); next }
        if (!close_to(f$estimate, a$estimate, FROZEN_TOLERANCE)) problems <- c(problems, paste(tag, "estimate differs from frozen"))
        if (!close_to(f$std_error_cluster_unit, a$std_error_cluster_unit, STANDARD_ERROR_TOLERANCE)) problems <- c(problems, paste(tag, "clustered standard error differs from frozen"))
        if (!close_to(f$std_error_iid, a$std_error_iid, STANDARD_ERROR_TOLERANCE)) problems <- c(problems, paste(tag, "iid standard error differs from frozen"))
    }
    for (name in c("simple")) {
        if (!close_to(document$reference$product_convention[[name]]$estimate, actual$product_convention[[name]]$estimate, FROZEN_TOLERANCE) ||
            !close_to(document$reference$product_convention[[name]]$std_error_cluster_unit, actual$product_convention[[name]]$std_error_cluster_unit, STANDARD_ERROR_TOLERANCE)) problems <- c(problems, "simple aggregate differs from frozen")
    }
    if (!identical(names(document$reference$product_convention$leads_by_event), names(actual$product_convention$leads_by_event))) problems <- c(problems, "lead set differs")
    for (e in names(actual$product_convention$leads_by_event)) {
        f <- document$reference$product_convention$leads_by_event[[e]]; a <- actual$product_convention$leads_by_event[[e]]
        if (!close_to(f$estimate, a$estimate, FROZEN_TOLERANCE) || !close_to(f$std_error_cluster_unit, a$std_error_cluster_unit, STANDARD_ERROR_TOLERANCE)) problems <- c(problems, paste("lead", e, "differs from frozen"))
    }
    ff <- document$reference$fixest_convention
    if (!close_to(ff$att$estimate, actual$fixest_convention$att$estimate, FROZEN_TOLERANCE) ||
        !close_to(ff$att$std_error_cluster_unit, actual$fixest_convention$att$std_error_cluster_unit, STANDARD_ERROR_TOLERANCE)) problems <- c(problems, "fixest ATT differs from frozen")
    for (e in names(actual$fixest_convention$event_time)) {
        f <- ff$event_time[[e]]; a <- actual$fixest_convention$event_time[[e]]
        if (is.null(f) || !close_to(f$estimate, a$estimate, FROZEN_TOLERANCE) || !close_to(f$std_error_cluster_unit, a$std_error_cluster_unit, STANDARD_ERROR_TOLERANCE)) problems <- c(problems, paste("fixest event time", e, "differs from frozen"))
    }
    # The committed numpy full-dummy reference, which the product test asserts, on the same panel.
    reference <- numpy$reference
    rows <- numpy$rows
    d <- read.csv(panel_path, stringsAsFactors=FALSE, colClasses=c("character", "integer", "integer", "numeric"))
    if (!identical(d$unit, unlist(rows$unit)) || !identical(d$period, as.integer(unlist(rows$period))) ||
        !identical(d$cohort, as.integer(unlist(rows$cohort))) ||
        !isTRUE(max(abs(d$outcome - as.numeric(unlist(rows$outcome)))) <= 1e-14)) {  # R's decimal parsers may differ in the last bit
        problems <- c(problems, "panel.csv differs from the committed numpy fixture's rows")
    }
    if (length(reference$cells) != length(actual$cells)) problems <- c(problems, "cell sets differ from the committed numpy reference")
    for (cell in reference$cells) {
        match <- Filter(function(a) a$cohort == cell$cohort && a$period == cell$period, actual$cells)
        if (length(match) != 1) { problems <- c(problems, sprintf("numpy cell (%d, %d) not estimated by fixest", cell$cohort, cell$period)); next }
        if (!close_to(cell$estimate, match[[1]]$estimate, ESTIMATE_TOLERANCE)) problems <- c(problems, sprintf("cell (%d, %d) fixest estimate differs from numpy reference", cell$cohort, cell$period))
    }
    if (!close_to(reference$simple, actual$product_convention$simple$estimate, ESTIMATE_TOLERANCE)) problems <- c(problems, "simple aggregate differs from numpy reference")
    if (!identical(sort(names(reference$leads_by_event)), sort(names(actual$product_convention$leads_by_event)))) problems <- c(problems, "lead set differs from numpy reference")
    for (e in names(reference$leads_by_event)) {
        if (!close_to(reference$leads_by_event[[e]], actual$product_convention$leads_by_event[[e]]$estimate, ESTIMATE_TOLERANCE)) problems <- c(problems, paste("lead", e, "differs from numpy reference"))
    }
    if (!identical(as.integer(unlist(reference$periods_without_comparison)), as.integer(actual$input$periods_without_comparison))) problems <- c(problems, "periods without comparison differ from numpy reference")
    if (sum(unlist(reference$dropped_rows_by_cohort)) != sum(unlist(actual$input$dropped_rows_by_cohort))) problems <- c(problems, "dropped rows differ from numpy reference")
    problems
}

actual <- fit_fixest()
numpy <- jsonlite::read_json(numpy_path, simplifyVector=FALSE)

if ("--write" %in% args) {
    previous <- if (file.exists(expected_path)) jsonlite::read_json(expected_path, simplifyVector=FALSE) else list()
    oracle <- if (is.null(previous$oracle)) list() else previous$oracle
    gap <- function(f) max(abs(f))
    estimates <- vapply(actual$cells, function(c) c$estimate, 0)
    numpy_estimates <- vapply(actual$cells, function(c) {
        m <- Filter(function(n) n$cohort == c$cohort && n$period == c$period, numpy$reference$cells)
        m[[1]]$estimate
    }, 0)
    document <- list(
        oracle=oracle,
        schema_version=1,
        case="interacted_did_unbalanced_panel_fixest",
        specification=SPEC,
        tolerances=list(estimate_vs_numpy_reference=ESTIMATE_TOLERANCE, regenerated_vs_frozen=FROZEN_TOLERANCE, standard_error=STANDARD_ERROR_TOLERANCE),
        input=actual$input,
        reference=list(cells=actual$cells, product_convention=actual$product_convention, fixest_convention=actual$fixest_convention),
        comparison=list(
            max_abs_cell_estimate_difference_to_numpy_reference=gap(estimates - numpy_estimates),
            max_abs_simple_difference_to_numpy_reference=abs(actual$product_convention$simple$estimate - numpy$reference$simple)))
    writeLines(jsonlite::toJSON(document, auto_unbox=TRUE, digits=I(17), pretty=TRUE, null="null"), expected_path)
    cat("wrote", expected_path, "\n")
    quit(status=0)
}

document <- jsonlite::read_json(expected_path, simplifyVector=FALSE)
problems <- check(document, actual, numpy)
if ("--self-test" %in% args) {
    require_failure <- function(wrong, name, ...) if (!length(check(wrong, actual, numpy))) stop(paste("fault escaped:", name))
    shift <- function(value, by) value + by
    for (index in seq_along(document$reference$cells)) {
        tag <- sprintf("cell %d", index)
        wrong <- document; wrong$reference$cells[[index]]$estimate <- shift(wrong$reference$cells[[index]]$estimate, 1e-6)
        require_failure(wrong, paste(tag, "shifted estimate"))
        wrong <- document; wrong$reference$cells[[index]]$std_error_cluster_unit <- shift(wrong$reference$cells[[index]]$std_error_cluster_unit, 1e-6)
        require_failure(wrong, paste(tag, "shifted clustered standard error"))
        wrong <- document; wrong$reference$cells[[index]]$std_error_iid <- shift(wrong$reference$cells[[index]]$std_error_iid, 1e-6)
        require_failure(wrong, paste(tag, "shifted iid standard error"))
    }
    wrong <- document; wrong$reference$cells <- wrong$reference$cells[-1]
    require_failure(wrong, "removed cell")
    wrong <- document; wrong$reference$product_convention$simple$estimate <- shift(wrong$reference$product_convention$simple$estimate, 1e-6)
    require_failure(wrong, "shifted simple aggregate")
    wrong <- document; wrong$reference$product_convention$leads_by_event[[1]]$estimate <- shift(wrong$reference$product_convention$leads_by_event[[1]]$estimate, 1e-6)
    require_failure(wrong, "shifted lead")
    wrong <- document; wrong$reference$fixest_convention$att$estimate <- shift(wrong$reference$fixest_convention$att$estimate, 1e-6)
    require_failure(wrong, "shifted fixest ATT")
    wrong <- document; wrong$specification$vcov <- "iid"
    require_failure(wrong, "changed vcov declaration")
    wrong <- document; wrong$tolerances$estimate_vs_numpy_reference <- 1e-3
    require_failure(wrong, "loosened tolerance")
    wrong <- document; wrong$oracle$kind <- "closed_form"
    require_failure(wrong, "false origin")
    wrong <- document; wrong$oracle$reference[[1]]$version <- "0.0.0"
    require_failure(wrong, "wrong comparator pin")
    # The comparison must discriminate: a numpy reference shifted by 1e-6 is rejected.
    wrong_numpy <- numpy; wrong_numpy$reference$cells[[1]]$estimate <- shift(wrong_numpy$reference$cells[[1]]$estimate, 1e-6)
    if (!length(check(document, actual, wrong_numpy))) stop("fault escaped: shifted numpy reference")
}
if (length(problems)) { cat(paste(problems, collapse="\n"), "\n"); quit(status=1) }
cat("fixest interacted-DiD reference", if ("--self-test" %in% args) "and fault controls", "passed\n")
