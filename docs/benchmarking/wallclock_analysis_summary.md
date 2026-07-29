# Wallclock Distribution Analysis — Summary

*Generated from `wallclock_analysis.ipynb`. Run the notebook for full tables and plots.*

## Key Findings

### 1. Wallclock values cluster at partition limits

Every dataset shows discrete spikes at round-number hours. These correspond to the
configured wallclock ceilings of each system's partitions:

| Hours | Datasets observed | Interpretation |
|-------|-------------------|----------------|
| 24h | Most datasets | Universal — common "standard" partition limit |
| 48h | Kestrel, Eagle, Fugaku, CC-IN2P3 | Common for long-running systems |
| 4h | Kestrel, Eagle, PM100 | "Short" queue limit |
| 2h | Kestrel, Eagle | Short partition (dominant on Kestrel: 54% of all jobs) |
| 72h | Fugaku | Fugaku-specific (3-day limit) |
| 0.5h | Lassen, PM100 | Debug/quick-turnaround partition |
| 12h | Fugaku, CC-IN2P3, Lassen | Common intermediate limit |

### 2. Users massively over-request wallclock time

Median utilization (actual_runtime / requested_wallclock) is typically 1–5%:
- Kestrel: jobs requesting 2h have median runtime of 115s (1.6% utilization)
- Eagle: jobs requesting 4h have median runtime of ~120s
- Fugaku: jobs requesting 72h have median runtime of ~2,153s

Of jobs requesting ≥24h, typically 70–90% finish in under half their request.

### 3. Clusters represent distinct populations

The median runtime differs dramatically between clusters on the same system:
- Kestrel: 2h cluster median=115s vs 48h cluster median=5,434s (**47× difference**)
- Eagle: 4h cluster median=~120s vs 48h cluster median=~5,400s
- PM100: 0.5h cluster median=~60s vs 24h cluster median=~1,200s

### 4. Variance reduction from clustering is moderate

| Dataset | Global CV | Per-cluster CV | Reduction |
|---------|-----------|----------------|-----------|
| nlr_kestrel | ~4.3 | ~2.5 | ~40% |
| nlr_eagle | ~4.5 | ~2.8 | ~38% |
| lassen | ~2.5 | ~1.5 | ~40% |
| pm100 | ~3.8 | ~2.3 | ~39% |

Per-cluster CV is lower but still high — the within-cluster p90/p10 ratio can still
be 50–6000×. Clustering helps but doesn't solve the heavy-tail problem alone.

## Implications for Mixture of Experts

1. **Natural boundaries exist** — partition limits are not arbitrary; they're system
   configuration that users select based on job type.

2. **Different populations with different runtimes** — the 47× difference between
   Kestrel's 2h and 48h clusters confirms these are genuinely different job types.

3. **But within-cluster variance is still large** — a per-cluster model still faces
   a challenging prediction problem. Combining clustering with log_target (issue #123)
   may be more effective than either alone.

4. **XGBoost can already split on wallclock** — the 4-way comparison experiment is
   needed to determine whether explicit routing adds value beyond internal tree splits.

## Recommended Next Steps

1. Run the 4-way comparison on Kestrel as proof-of-concept
2. Use the detected spikes (2h, 4h, 24h, 48h) as bin boundaries
3. Evaluate log_target within each bin (combining issues #123 and #124)
4. If mixture beats single-model, generalize to other datasets using their
   detected spikes as boundaries
