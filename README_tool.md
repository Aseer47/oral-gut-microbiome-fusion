# microbiome-site-attribution

A standalone Python tool for SHAP-based site-attribution analysis of paired multi-site microbiome data.

## What it does

When a machine learning model is trained on features from multiple body sites (e.g., stool and oral cavity), accuracy alone cannot reveal whether a second site carries complementary information or merely repeats the first. This tool uses SHAP (SHapley Additive exPlanations) to attribute predictive importance to each site of origin within a single fused model, revealing how much each site actually contributes, regardless of whether that contribution shows up as an accuracy improvement.

## Installation

```bash
pip install git+https://github.com/DaemonTargaryen47/oral-gut-microbiome-fusion
```

Or clone and install locally:

```bash
git clone https://github.com/DaemonTargaryen47/oral-gut-microbiome-fusion
cd oral-gut-microbiome-fusion
pip install -e .
```

## Quick start

```python
from microbiome_site_attribution import build_fused_matrix, site_attribution_score

# Build fused matrix from per-site abundance DataFrames
# (subjects x taxa, with subjects as the index)
X_fused = build_fused_matrix(
    site_tables={"stool": stool_df, "oralcavity": oral_df},
    apply_clr=True,
    apply_variance_filter=True
)

# Run site-attribution analysis
result = site_attribution_score(
    X_fused=X_fused,
    y=y_binary,
    site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"},
    n_permutations=200
)

print(result.summary())
result.plot()
```

## Output

```
=======================================================
  SHAP Site-Attribution Result
=======================================================
  Subjects:    140
  Model:       RandomForestClassifier

  Site contribution (% of total SHAP importance):
    stool               : 39.3%
    oralcavity          : 60.7% <-- dominant

  Permutation p-value: 0.0000
  Interpretation: site-contribution asymmetry is
  statistically significant (p < 0.05).
=======================================================
```

## Key functions

| Function | Description |
|---|---|
| `site_attribution_score()` | Main function: fits a model, computes SHAP values, attributes importance by site, runs permutation test |
| `build_fused_matrix()` | Builds a fused feature matrix from per-site DataFrames with CLR transform and variance filtering |
| `compare_cohorts()` | Plots side-by-side site-contribution bars across multiple datasets |
| `clr_transform()` | Centered log-ratio transformation for compositional data |
| `filter_low_variance()` | Removes low-variance taxa |

## Requirements

- Python >= 3.8
- numpy, pandas, scikit-learn, shap, matplotlib

## Citation

If you use this tool, please cite:

> Ruthbah, C.A., Sadi, T.H., Jahan, N.E.S., & Adib, A.N.M.T. (2026). Interpretable Machine Learning Reveals Complementary Age-Related Signatures in the Oral and Gut Microbiome. *bioRxiv*. [https://doi.org/10.1101/BIORXIV/2026/750358](https://doi.org/10.64898/2026.09.09.750358)

## License

MIT License
