# Benchmark lme4 on exactly the fixtures bench/vs_lme4.py writes out.
#
# lme4 is the reference this package was built against, and it is also the floor
# for pymer4 -- pymer4 calls lme4 through rpy2, so it cannot be faster than this.
#
# Usage:  Rscript bench/vs_lme4.R <fixture-dir>

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[1] else "bench/fixtures"
libp <- Sys.getenv("R_USER_LIB", unset = "C:/Users/Fatin/R/win-library")
if (dir.exists(libp)) .libPaths(c(libp, .libPaths()))

suppressPackageStartupMessages(library(lme4))

manifest <- read.csv(file.path(dir, "manifest.csv"), stringsAsFactors = FALSE)
out <- data.frame()

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  df <- read.csv(file.path(dir, row$file))
  df$g <- factor(df$g)

  form <- if (row$re == "slope") {
    y ~ x1 + x2 + (1 + x1 | g)
  } else {
    y ~ x1 + x2 + (1 | g)
  }
  reml <- as.logical(row$reml)

  # warm up once (JIT / first-call costs), then take the best of three
  invisible(tryCatch(lmer(form, data = df, REML = reml), error = function(e) NULL))
  best <- Inf
  fit <- NULL
  for (rep in 1:3) {
    t0 <- proc.time()[["elapsed"]]
    f <- tryCatch(lmer(form, data = df, REML = reml), error = function(e) NULL)
    dt <- proc.time()[["elapsed"]] - t0
    if (dt < best) { best <- dt; fit <- f }
  }

  if (is.null(fit)) {
    out <- rbind(out, data.frame(name = row$name, seconds = NA, logLik = NA,
                                 beta0 = NA, beta1 = NA, beta2 = NA, sigma = NA))
  } else {
    fe <- fixef(fit)
    out <- rbind(out, data.frame(
      name = row$name,
      seconds = best,
      logLik = as.numeric(logLik(fit)),
      beta0 = fe[[1]], beta1 = fe[[2]], beta2 = fe[[3]],
      sigma = sigma(fit)
    ))
  }
  cat(sprintf("%-22s %8.3f s\n", row$name, best))
}

write.csv(out, file.path(dir, "lme4_results.csv"), row.names = FALSE)
cat("\nwrote", file.path(dir, "lme4_results.csv"), "\n")
