#!/usr/bin/env python3
"""NDIMR module: scanpy PCA on the normalized matrix over ALL genes.

Same code path as pca.py (load, sc.pp.pca, writers); only the input differs:
--normalized_h5 (NORM output, no feature selection) instead of the FEAT subset.
--pca_type standardized z-scores each gene first (sc.pp.scale, no clipping),
the standardisation nd-randomly applies; --pca_type centered (default)
mean-centres only, as the PCA arms do. On nr-scrapper output this arm is also
rd-bipca's counterfactual: same FILT genes, log-normalization instead of
biwhitening. Outputs are pca.py's: {name}_embedding.tsv, {name}_loadings.tsv.
"""
from pca import main

if __name__ == "__main__":
    main("NDIMR")
