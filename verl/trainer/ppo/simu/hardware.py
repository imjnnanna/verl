from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    peak_compute_flops: float           # FP16 TFLOPS, stored as raw FLOPS not T
    peak_memory_bandwidth: float        # HBM bytes/sec
    ridge_flops_per_byte: float         # roofline ridge: AI above this is compute-bound
    compute_efficiency: float = 0.7     # achieved fraction of peak for large GEMMs
    memory_efficiency: float = 0.8      # achieved fraction of peak HBM BW
    small_gemm_efficiency: float = 0.4  # used when any GEMM dim < 1024
