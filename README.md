# Oral-Gut Microbiome Fusion

Interpretable ML framework for evaluating multi-site microbiome fusion using SHAP-based site-attribution analysis. Tests whether combining oral cavity and gut (stool) microbiome data from the same subjects reveals complementary biological signal beyond what accuracy scores can show.

## Key Findings

- Gut (stool) data alone achieves near-perfect classification (AUC = 1.00) of adult vs. newborn status, yet the fused model still draws 58.1% of its SHAP importance from oral cavity features, revealing real complementary signal that accuracy alone would miss.
- Replicated in an independent cohort (Brito et al., 2016; n=140, sex classification): oral cavity again contributed more SHAP importance (60.7% vs. 39.3%), with the asymmetry statistically significant under permutation testing (p < 0.005).
- Dominant site flips between cohorts by accuracy (stool leads in Ferretti, oral leads in Brito), but SHAP attribution consistently favors oral cavity in both — demonstrating that accuracy and interpretability can tell different stories about which site matters.
- Top taxa (*Malassezia restricta*, *Staphylococcus epidermidis*, *Prevotella melaninogenica*) behave consistently with established early-life microbiome colonization biology, verified directly via SHAP dependence plots.

## Contents

```
├── oral_gut_microbiome_fusion.ipynb      # Primary analysis pipeline (Ferretti cohort, Colab-ready)
├── microbiome_fusion_replication.ipynb   # Replication pipeline (Brito cohort + permutation tests)
├── microbiome_site_attribution.py        # Standalone reusable Python tool
├── example_usage.py                      # Minimal usage example for the tool
├── setup.py                              # pip-installable package setup
├── README_tool.md                        # Tool-specific documentation
├── results/                              # Figures, tables, robustness outputs (primary cohort)
├── results_replication/                  # Figures and outputs (replication cohort)
└── README.md
```

## Data

Two public shotgun metagenomic datasets, both accessed via the [`curatedMetagenomicData`](https://bioconductor.org/packages/release/data/experiment/html/curatedMetagenomicData.html) Bioconductor package. No new data was collected.

| Cohort | Source | n | Task |
|---|---|---|---|
| Primary | Ferretti et al. (2018) | 44 matched stool + oral subjects | Adult vs. newborn classification |
| Replication | Brito et al. (2016) | 140 matched stool + oral subjects | Sex classification |

## Reusable Tool

The site-attribution method is available as a standalone Python tool, installable directly from this repo:

```bash
pip install git+https://github.com/Aseer47/oral-gut-microbiome-fusion
```

```python
from microbiome_site_attribution import build_fused_matrix, site_attribution_score

X_fused = build_fused_matrix({"stool": stool_df, "oralcavity": oral_df})
result = site_attribution_score(X_fused, y, site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"})
print(result.summary())
result.plot()
```

See [`README_tool.md`](README_tool.md) for full documentation.

## Methods Summary

- CLR-transformed abundance features, with low-variance filtering applied per site
- Logistic Regression, Random Forest, and XGBoost evaluated via 5-fold stratified cross-validation
- SHAP site-attribution via `TreeExplainer` / `LinearExplainer`, with a permutation significance test on the site-contribution split (200 permutations per cohort)
- Robustness checks: label-shuffling null baseline, bootstrap 95% CIs, preprocessing sensitivity checks

## Running

Open either notebook in Google Colab and run all cells top to bottom. The R/Bioconductor setup step (data retrieval) takes 10–15 minutes on first run. All outputs save to `results/` or `results_replication/`.

## Citation

If you use this code or the site-attribution tool, please cite:

> Ruthbah, C.A., Sadi, T.H., Jahan, N.E.S., & Adib, A.N.M.T. (2026). Interpretable Machine Learning Reveals Complementary Age-Related Signatures in the Oral and Gut Microbiome. *bioRxiv*. https://doi.org/10.64898/2026.09.09.750358

## Authors

Chowdhury Aseer Ruthbah, Talim Hossain Sadi, Nur E Shiratun Jahan, Abu Nayem Md. Tanzim Adib — BRAC University

## License

[MIT](LICENSE). The underlying datasets retain their original licensing terms.

## Contact

Chowdhury Aseer Ruthbah — chowdhury.aseer.ruthbah@g.bracu.ac.bd
