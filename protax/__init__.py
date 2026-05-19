"""
PROTAX-GPU: A GPU-accelerated probabilistic taxonomic classification system for DNA barcodes.

This package provides a JAX-based implementation of PROTAX with custom CUDA kernels
for accelerated k-nearest neighbor search. It enables fast and accurate taxonomic
classification of query sequences against large reference databases.

Main modules:
    - classify: Functions for classifying query sequences
    - model: Core probabilistic model and computation functions
    - taxonomy: Data structures for taxonomic trees and model parameters
    - protax_utils: Utilities for reading/writing model files and sequences
    - baseline: Simple nearest-neighbor baseline classifier
    - ops: Custom JAX operations including GPU-accelerated KNN

Example:
    >>> from protax.classify import classify_file
    >>> classify_file("query.aln", "model.npz", "taxonomy.npz")
"""
