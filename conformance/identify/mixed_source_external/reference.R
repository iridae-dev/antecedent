# Independent public black-box reference for the four dosearch cases of this fixture.
# No product code or upstream source/test file is read; the package is isolated tooling that
# is never used by the product or ordinary CI.  Run prepare.R first.
#   Rscript --vanilla conformance/identify/mixed_source_external/reference.R --check [--self-test]
# SPDX-License-Identifier: MIT OR Apache-2.0
file_arg <- grep("^--file=", commandArgs(), value=TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
root <- normalizePath(file.path(here, "../../.."))
library_path <- Sys.getenv("ANTECEDENT_R_DOSEARCH_LIBRARY", file.path(root, "target/reference-environments/r-dosearch/library"))
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
if (!"--check" %in% args || any(!args %in% c("--check", "--self-test"))) stop("Usage: --check [--self-test]")
library(dosearch)

# Pinned scenario coordinates, independent of the fixture so a changed scenario is reported.
specs <- list(
    dosearch_backdoor_single_distribution = list(
        data="p(x,y,z)", graph="z -> x\nz -> y\nx -> y", opts=list(),
        nodes=c("x","y","z"), directed=list(c("x","y"), c("z","x"), c("z","y")), bidirected=list(),
        inputs=list(list(study="obs", do=list(), measured=list("x","y","z")))),
    dosearch_frontdoor_single_distribution = list(
        data="p(x,z,y)", graph="x -> z\nz -> y\nx <-> y", opts=list(),
        nodes=c("x","z","y"), directed=list(c("x","z"), c("z","y")), bidirected=list(c("x","y")),
        inputs=list(list(study="obs", do=list(), measured=list("x","z","y")))),
    dosearch_mixed_distributions_two_studies = list(
        data="p(x,z)\np(z,y)", graph="x -> z\nz -> y", opts=list(),
        nodes=c("x","z","y"), directed=list(c("x","z"), c("z","y")), bidirected=list(),
        inputs=list(list(study="study-1", do=list(), measured=list("x","z")), list(study="study-2", do=list(), measured=list("z","y")))),
    dosearch_transportability_and_selection_bias = list(
        data="p(x,z,y|s)\np(y,z|t,do(x))", graph="x -> z\nz -> y\nx -> s\nt -> z\nx <-> y",
        opts=list(transportability="t", selection_bias="s"),
        nodes=c("x","z","y","s","t"), directed=list(c("x","z"), c("z","y"), c("x","s"), c("t","z")), bidirected=list(c("x","y")), inputs=NULL)
)
run_case <- function(spec) {
    out <- do.call(dosearch::dosearch, c(list(data=spec$data, query="p(y|do(x))", graph=spec$graph), spec$opts))
    list(identifiable=isTRUE(out$identifiable), formula=if (isTRUE(out$identifiable)) out$formula else NA_character_)
}
actual <- lapply(specs, run_case)

check <- function(document) {
    problems <- character()
    if (!identical(document$oracle$kind, "external_package")) problems <- c(problems, "incorrect oracle origin")
    versions <- vapply(document$oracle$reference, function(r) paste(r$project, r$version), "")
    if (!"dosearch 1.0.12" %in% versions) problems <- c(problems, "missing pinned dosearch comparator")
    ids <- vapply(document$cases, function(c) c$id, "")
    for (name in names(specs)) {
        spec <- specs[[name]]
        if (is.na(match(name, ids))) { problems <- c(problems, paste(name, "case missing")); next }
        case <- document$cases[[match(name, ids)]]
        if (!identical(unlist(case$nodes), spec$nodes) ||
            !identical(lapply(case$directed, unlist), spec$directed) ||
            !identical(lapply(case$bidirected, unlist), spec$bidirected) ||
            !identical(case$oracle, "dosearch") ||
            !identical(unlist(case$query$treatments), "x") || !identical(unlist(case$query$outcomes), "y")) {
            problems <- c(problems, paste(name, "graph or query changed"))
        }
        got <- actual[[name]]
        verdict <- if (got$identifiable) "identified" else "not_identified"
        if (!identical(case$oracle_result, verdict)) problems <- c(problems, paste(name, "oracle verdict differs"))
        if (!identical(case$oracle_formula, got$formula)) problems <- c(problems, paste(name, "oracle formula differs"))
        if (!identical(case$oracle_output_verbatim, got$formula)) problems <- c(problems, paste(name, "verbatim output differs"))
        if (!is.null(spec$inputs)) {
            if (!identical(lapply(case$inputs, function(i) list(study=i$study, do=lapply(i$do, identity), measured=lapply(i$measured, identity))), spec$inputs)) {
                problems <- c(problems, paste(name, "study inputs changed"))
            }
        } else if (!identical(unlist(case$oracle_input_syntax), strsplit(spec$data, "\n")[[1]])) {
            problems <- c(problems, paste(name, "input syntax changed"))
        }
    }
    problems
}
document <- jsonlite::read_json(file.path(here, "expected.json"), simplifyVector=FALSE)
problems <- check(document)
if ("--self-test" %in% args) {
    require_failure <- function(wrong, name) if (!length(check(wrong))) stop(paste("fault escaped:", name))
    for (index in seq_along(document$cases)) {
        if (!identical(document$cases[[index]]$oracle, "dosearch")) next
        id <- document$cases[[index]]$id
        wrong <- document; wrong$cases[[index]]$oracle_output_verbatim <- "p(y)"
        require_failure(wrong, paste(id, "altered output"))
        wrong <- document; wrong$cases[[index]]$oracle_result <- "not_identified"
        require_failure(wrong, paste(id, "flipped verdict"))
        wrong <- document; wrong$cases[[index]]$directed <- list()
        require_failure(wrong, paste(id, "changed graph"))
        wrong <- document; wrong$cases <- wrong$cases[-index]
        require_failure(wrong, paste(id, "removed case"))
    }
    wrong <- document; wrong$oracle$kind <- "closed_form"
    require_failure(wrong, "false origin")
    wrong <- document; wrong$oracle$reference[[2]]$version <- "0.0.0"
    require_failure(wrong, "wrong comparator pin")
    # The comparator must discriminate: an unidentifiable specification is reported as such.
    unidentified <- dosearch::dosearch(data="p(x,y)", query="p(y|do(x))", graph="x -> y\nx <-> y")
    if (isTRUE(unidentified$identifiable)) stop("fault escaped: comparator failed to refuse the confounded observational specification")
}
if (length(problems)) { cat(paste(problems, collapse="\n"), "\n"); quit(status=1) }
cat("dosearch mixed-source references", if ("--self-test" %in% args) "and fault controls", "passed\n")
