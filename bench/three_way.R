# The lme4 half of bench/three_way.py. Not meant to be run on its own.
#
# Fits every fixture in a manifest with lme4 and writes one CSV row per
# repetition, so the Python recorder sees every timing rather than a summary.
#
# Timing is taken inside R and covers exactly what the Python side times: a
# data frame already in memory, through formula parsing and model construction,
# to a fitted model. Reading the CSV and converting the grouping column to a
# factor happen before the clock starts, as `pd.read_csv` does on the Python
# side.
#
# Sys.time() rather than proc.time(): proc.time()'s elapsed clock ticks in
# 10 ms steps on Windows, which is the whole fit at the small sizes and would
# make the small-case ratios meaningless.
#
# The manifest names, per fixture, the random-effects structure ("slope" for
# (1 + x1 | g), "intercept" for (1 | g)), REML or ML, and a repetition count.
# A fit that errors is recorded as a row with `error` set, never dropped: a
# fixture lme4 cannot fit is a result, and silently losing it would bias the
# comparison toward the cases everyone gets right.
#
# Usage:  Rscript bench/three_way.R <manifest.csv> <out.csv>

args <- commandArgs(trailingOnly = TRUE)
manifest_path <- args[1]
out_path <- args[2]

# Same library resolution as bench/vs_lme4.R: a per-user library is where
# install.packages() puts lme4 on a machine without write access to R's own.
default_lib <- if (.Platform$OS.type == "windows") {
  file.path(path.expand("~"), "R", "win-library")
} else {
  file.path(path.expand("~"), "R", "library")
}
libp <- Sys.getenv("R_USER_LIB", unset = default_lib)
if (dir.exists(libp)) .libPaths(c(libp, .libPaths()))
suppressPackageStartupMessages(library(lme4))

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
rows <- list()

for (i in seq_len(nrow(manifest))) {
  m <- manifest[i, ]
  df <- read.csv(m$file)
  df$g <- factor(df$g)

  form <- if (m$re == "slope") y ~ x1 + x2 + (1 + x1 | g) else y ~ x1 + x2 + (1 | g)
  reml <- as.logical(m$reml)

  for (rep in seq_len(m$reps)) {
    t0 <- Sys.time()
    fit <- tryCatch(suppressWarnings(lmer(form, data = df, REML = reml)),
                    error = function(e) e)
    seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    if (inherits(fit, "error")) {
      rows[[length(rows) + 1]] <- data.frame(
        case = m$case, rep = rep, seconds = seconds, loglik = NA,
        beta0 = NA, beta1 = NA, beta2 = NA, se0 = NA, se1 = NA, se2 = NA,
        re00 = NA, re01 = NA, re11 = NA, scale = NA, warnings = "",
        singular = NA, error = conditionMessage(fit))
      next
    }

    fe <- fixef(fit)
    se <- sqrt(diag(as.matrix(vcov(fit))))
    vc <- as.matrix(VarCorr(fit)$g)
    if (nrow(vc) == 1) vc <- matrix(c(vc[1, 1], NA, NA, NA), 2, 2)
    msgs <- fit@optinfo$conv$lme4$messages
    rows[[length(rows) + 1]] <- data.frame(
      case = m$case, rep = rep, seconds = seconds,
      loglik = as.numeric(logLik(fit)),
      beta0 = fe[[1]], beta1 = fe[[2]], beta2 = fe[[3]],
      se0 = se[[1]], se1 = se[[2]], se2 = se[[3]],
      re00 = vc[1, 1], re01 = vc[1, 2], re11 = vc[2, 2],
      scale = sigma(fit)^2,
      warnings = if (is.null(msgs)) "" else paste(msgs, collapse = " | "),
      singular = isSingular(fit), error = ""
    )
    cat(sprintf("  lme4 %-10s rep %d  %.4f s\n", m$case, rep, seconds))
  }
}

write.csv(do.call(rbind, rows), out_path, row.names = FALSE)
cat("lme4", as.character(packageVersion("lme4")), "R", R.version$major,
    R.version$minor, "\n", file = paste0(out_path, ".versions"))
