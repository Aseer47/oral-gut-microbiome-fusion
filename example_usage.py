"""
example_usage.py
================
Minimal example showing how to use microbiome_site_attribution.py
on your own paired multi-site microbiome data.

This example uses synthetic data to demonstrate the API.
Replace the data-loading section with your own real data.
"""

import numpy as np
import pandas as pd
from microbiome_site_attribution import (
    build_fused_matrix,
    site_attribution_score,
    compare_cohorts
)

# --------------------------------------------------------------------------
# Step 1: Load your data (replace this with your real data)
# --------------------------------------------------------------------------
# Your data should be two DataFrames: one per body site.
# Rows = subjects (same index across both DataFrames).
# Columns = microbial taxa (relative abundances, before CLR).

np.random.seed(42)
n_subjects = 100
n_taxa = 200

# Synthetic stool abundance table
stool_df = pd.DataFrame(
    np.abs(np.random.dirichlet(np.ones(n_taxa), size=n_subjects)),
    columns=[f"taxon_{i}" for i in range(n_taxa)],
    index=[f"subject_{i}" for i in range(n_subjects)]
)

# Synthetic oral cavity abundance table
oral_df = pd.DataFrame(
    np.abs(np.random.dirichlet(np.ones(n_taxa), size=n_subjects)),
    columns=[f"taxon_{i}" for i in range(n_taxa)],
    index=[f"subject_{i}" for i in range(n_subjects)]
)

# Synthetic binary labels (e.g. 0 = control, 1 = case)
y = pd.Series(np.random.randint(0, 2, size=n_subjects),
              index=[f"subject_{i}" for i in range(n_subjects)])

# --------------------------------------------------------------------------
# Step 2: Build fused matrix (CLR transform + variance filtering applied)
# --------------------------------------------------------------------------
X_fused = build_fused_matrix(
    site_tables={"stool": stool_df, "oralcavity": oral_df},
    apply_clr=True,
    apply_variance_filter=True
)
print(f"Fused matrix shape: {X_fused.shape}")
print(f"Sample feature names: {list(X_fused.columns[:4])}\n")

# --------------------------------------------------------------------------
# Step 3: Run site-attribution analysis
# --------------------------------------------------------------------------
result = site_attribution_score(
    X_fused=X_fused,
    y=y,
    site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"},
    n_permutations=200,       # increase to 1000+ for publication
    random_state=42
)

# Print text summary
print(result.summary())

# Plot bar chart + permutation distribution
result.plot(show_permutation=True)

# --------------------------------------------------------------------------
# Step 4 (optional): Compare across two cohorts
# --------------------------------------------------------------------------
# If you have results from two datasets, compare them side by side

# Simulating a second cohort result for illustration
result2 = site_attribution_score(
    X_fused=X_fused,
    y=y,
    site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"},
    n_permutations=100,
    random_state=99,
    verbose=False
)

compare_cohorts({
    "Cohort 1 (n=100)": result,
    "Cohort 2 (n=100)": result2
})
