# scSHC testClusters on each partition column; called by pick_scshc.py.
#   Rscript pick_scshc.R TMPDIR ALPHA MIN_CLUSTER CORES   (reads counts.mtx, genes.txt, partitions.csv; writes scores.csv)
suppressPackageStartupMessages({ library(scSHC); library(Matrix) })
a <- commandArgs(trailingOnly = TRUE); d <- a[1]
alpha <- as.numeric(a[2]); min_cl <- as.integer(a[3]); cores <- as.integer(a[4])
X <- as(readMM(file.path(d, "counts.mtx")), "CsparseMatrix")
rownames(X) <- readLines(file.path(d, "genes.txt"))
parts <- read.csv(file.path(d, "partitions.csv"), check.names = FALSE, colClasses = "character")
colnames(X) <- parts[[1]]
rows <- list()
for (res in names(parts)[-1]) {
  cl <- parts[[res]]; small <- names(which(table(cl) < min_cl)); keep <- !(cl %in% small)
  t0 <- Sys.time()
  out <- testClusters(X[, keep], as.character(cl[keep]), alpha = alpha, num_features = 2500, num_PCs = 30,
                      parallel = cores > 1, cores = cores)
  rows[[res]] <- data.frame(resolution = as.numeric(res), k_in = length(unique(cl[keep])),
                            k_dropped_small = length(small), k_supported = length(unique(out[[1]])),
                            seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  cat(sprintf("  res %s: %d clusters in, %d supported (%d s)\n", res, rows[[res]]$k_in,
              rows[[res]]$k_supported, rows[[res]]$seconds))
}
write.csv(do.call(rbind, rows), file.path(d, "scores.csv"), row.names = FALSE)
