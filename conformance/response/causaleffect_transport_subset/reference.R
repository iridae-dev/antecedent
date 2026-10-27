# Independent public black-box references; no product or upstream source is read.
# SPDX-License-Identifier: MIT OR Apache-2.0
file_arg <- grep("^--file=", commandArgs(), value=TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
root <- normalizePath(file.path(here, "../../.."))
library_path <- Sys.getenv("ANTECEDENT_R_ORACLE_LIBRARY", file.path(root, "target/reference-environments/r-oracles/library"))
if (!dir.exists(library_path)) stop("Missing isolated library; run this fixture's prepare.R first")
.libPaths(c(normalizePath(library_path), .Library), include.site=FALSE)
if (!requireNamespace("jsonlite", quietly=TRUE)) stop("Missing locked jsonlite; run prepare.R")
lock <- jsonlite::read_json(file.path(here, "renv.lock"), simplifyVector=FALSE)
environment_problems <- function(record) {
    problems <- character()
    if (!identical(as.character(getRversion()), record$R$Version)) problems <- c(problems, "wrong R version")
    for (name in names(record$Packages)) {
        actual <- tryCatch(utils::packageDescription(name, fields="Version"), error=function(e) NA_character_)
        if (!identical(actual, record$Packages[[name]]$Version)) problems <- c(problems, paste("wrong package version:", name))
    }
    problems
}
problems <- environment_problems(lock)
if (length(problems)) stop(paste(problems, collapse="\n"))
args <- commandArgs(trailingOnly=TRUE)
if (any(!args %in% c("--check", "--self-test"))) stop("Unknown or empty fixture selection")

causal <- jsonlite::read_json(file.path(here, "expected.json"), simplifyVector=FALSE)
bp <- jsonlite::read_json(file.path(root, "conformance/response/binary_iv_balke_pearl_table1/expected.json"), simplifyVector=FALSE)
causal_ids <- c("direct_irrelevant_selection", "standardize_exogenous_selection")
causal_coordinates <- list(
    list(nodes=list("X","Z","Y"), directed=list(list("X","Y")), bidirected=list(), selection_targets=list("Z"), source_experiments=list("X"), treatment="X", outcome="Y"),
    list(nodes=list("X","Z","Y"), directed=list(list("Z","Y"),list("X","Y")), bidirected=list(), selection_targets=list("Z"), source_experiments=list("X"), treatment="X", outcome="Y")
)
check_causal <- function(document) {
    if (length(document$cases) != 2L) return("missing required transport cases")
    problems <- character()
    for (i in seq_along(causal_ids)) {
        case <- document$cases[[i]]
        if (!identical(case$id, causal_ids[[i]]) || !identical(case$antecedent, causal_coordinates[[i]])) return("transport graph or regime coordinates changed")
        ante <- case$antecedent
        edges <- do.call(rbind, lapply(ante$directed, unlist))
        selected <- unlist(ante$selection_targets, use.names=FALSE)
        # Preserve declaration/edge order of the historical public-API calls.
        graph <- igraph::graph_from_edgelist(rbind(edges, cbind("S", selected)), directed=TRUE)
        graph <- igraph::set_vertex_attr(graph, "description", index=which(igraph::V(graph)$name == "S"), value="S")
        actual <- as.character(causaleffect::transport(y=ante$outcome, x=ante$treatment, D=graph, expr=TRUE, simp=TRUE))
        if (!identical(actual, document$reference$outputs[[case$id]]) || !identical(actual, case$oracle$expression)) problems <- c(problems, paste(case$id, "transport expression differs"))
    }
    if (!identical(document$oracle$kind, "external_package")) problems <- c(problems, "incorrect transport oracle origin")
    problems
}
check_bp <- function(document) {
    expected_cells <- matrix(c(.0064,.9936,0,0,.0028,.1972,.001,.799), nrow=2, byrow=TRUE)
    cells <- do.call(rbind, lapply(document$cells, unlist))
    if (!identical(unname(cells), expected_cells)) return("binary-IV input law changed")
    if (!identical(document$tolerance$atol, 1e-12) || !identical(document$tolerance$rtol, 1e-12)) return("binary-IV tolerance changed")
    # Fixture order is (Y,D); R array order is (D,Y,Z), first dimension fastest.
    cp <- as.numeric(t(cells[, c(1,3,2,4)]))
    probabilities <- as.table(array(cp, dim=c(2,2,2), dimnames=list(x=c(0,1), y=c(0,1), z=c(0,1))))
    bounds <- bpbounds::bpbounds(probabilities)
    actual <- list(lower=unname(bounds$bplb), upper=unname(bounds$bpub), iv_inequality_satisfied=unname(bounds$inequality))
    problems <- character()
    for (field in c("lower", "upper")) {
        for (expected in list(document$expected[[field]], document$reference$outputs[[field]])) {
            if (!is.numeric(expected) || length(expected) != 1L || !is.finite(expected) || abs(actual[[field]] - expected) > 1e-12) problems <- c(problems, paste("binary-IV", field, "differs"))
        }
    }
    for (expected in list(document$expected$iv_inequality_satisfied, document$reference$outputs$iv_inequality_satisfied)) {
        if (!identical(actual$iv_inequality_satisfied, expected)) problems <- c(problems, "binary-IV inequality verdict differs")
    }
    if (!identical(document$oracle$kind, "external_package")) problems <- c(problems, "incorrect binary-IV oracle origin")
    problems
}
problems <- c(check_causal(causal), check_bp(bp))
if ("--self-test" %in% args) {
    require_failure <- function(found, name) if (!length(found)) stop(paste("fault escaped:", name))
    wrong <- causal
    wrong$cases <- list()
    require_failure(check_causal(wrong), "empty transport selection")
    wrong <- causal
    wrong$cases[[1]]$antecedent$directed <- list()
    require_failure(check_causal(wrong), "wrong transport graph")
    wrong <- causal
    wrong$cases[[1]]$oracle$expression <- "P^*(X|Y)"
    wrong$reference$outputs[[causal_ids[[1]]]] <- "P^*(X|Y)"
    require_failure(check_causal(wrong), "both frozen expression copies corrupted")
    wrong <- bp
    wrong$cells[[1]][[1]] <- .1
    require_failure(check_bp(wrong), "wrong IV law")
    wrong <- bp
    wrong$tolerance$atol <- 1
    require_failure(check_bp(wrong), "widened IV tolerance")
    wrong <- bp
    wrong$expected$lower <- 1
    require_failure(check_bp(wrong), "wrong IV bound")
    wrong <- bp
    wrong$expected$iv_inequality_satisfied <- FALSE
    require_failure(check_bp(wrong), "wrong IV inequality")
    wrong <- bp
    wrong$oracle$kind <- "closed_form"
    require_failure(check_bp(wrong), "false independent origin")
    wrong <- lock
    wrong$Packages$causaleffect$Version <- "0"
    require_failure(environment_problems(wrong), "incorrect comparator version")
}
if (length(problems)) stop(paste(problems, collapse="\n"))
cat("Two causaleffect formulas, binary-IV bounds and R reference fault controls passed\n")
