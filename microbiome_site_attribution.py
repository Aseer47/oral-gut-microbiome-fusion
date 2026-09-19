"""
microbiome_site_attribution.py
==============================
A standalone Python module for SHAP-based site-attribution analysis
of paired multi-site microbiome data.

This implements the site-contribution framework described in:

    Ruthbah, C.A., Sadi, T.H., Jahan, N.E.S., & Adib, A.N.M.T. (2026).
    Interpretable Machine Learning Reveals Complementary Age-Related
    Signatures in the Oral and Gut Microbiome. bioRxiv.
    https://doi.org/10.1101/BIORXIV/2026/750358

The core idea: when a machine learning model is trained on features from
multiple body sites (e.g., stool and oral cavity), accuracy alone cannot
determine whether a second site carries complementary information or merely
repeats the first. This module uses SHAP (SHapley Additive exPlanations) to
attribute predictive importance to each site of origin within a single fused
model, revealing how much each site actually contributes regardless of
whether that contribution shows up as an accuracy improvement.

Quick start
-----------
    import pandas as pd
    from microbiome_site_attribution import site_attribution_score

    # X_fused: a DataFrame where features from each site are prefixed
    # e.g. columns named "stool__taxon_A", "oralcavity__taxon_B", etc.
    result = site_attribution_score(
        X_fused=X_fused,
        y=y_binary,
        site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"},
        n_permutations=200
    )

    print(result.summary())
    result.plot()

Requirements
------------
    numpy, pandas, scikit-learn, shap, matplotlib
    Install: pip install numpy pandas scikit-learn shap matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------

@dataclass
class SiteAttributionResult:
    """
    Holds the output of a site_attribution_score() call.

    Attributes
    ----------
    site_contribution_pct : dict
        Percentage of total mean absolute SHAP importance attributed to
        each site. Keys are site names, values are percentages (sum to 100).
        Example: {"stool": 41.9, "oralcavity": 58.1}

    site_contribution_abs : dict
        Raw total mean absolute SHAP importance per site, before converting
        to percentages.

    p_value : float or None
        Permutation p-value for the site-contribution asymmetry. Computed
        by shuffling which features belong to which site (preserving site
        sizes) and checking how often random reassignment produces an equal
        or greater asymmetry than observed. None if n_permutations=0.

    null_distribution : np.ndarray or None
        The distribution of max-site-share values under the null (random
        feature-to-site reassignment). None if n_permutations=0.

    n_subjects : int
        Number of subjects used to fit the model and compute SHAP values.

    n_features_per_site : dict
        Number of features per site after any preprocessing.

    dominant_site : str
        The site with the highest SHAP contribution percentage.

    model_used : str
        The model class name used for SHAP computation.
    """
    site_contribution_pct: Dict[str, float]
    site_contribution_abs: Dict[str, float]
    p_value: Optional[float]
    null_distribution: Optional[np.ndarray]
    n_subjects: int
    n_features_per_site: Dict[str, int]
    dominant_site: str
    model_used: str
    _site_names: List[str] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Return a human-readable text summary of the result."""
        lines = [
            "=" * 55,
            "  SHAP Site-Attribution Result",
            "=" * 55,
            f"  Subjects:    {self.n_subjects}",
            f"  Model:       {self.model_used}",
            "",
            "  Site contribution (% of total SHAP importance):",
        ]
        for site, pct in self.site_contribution_pct.items():
            marker = " <-- dominant" if site == self.dominant_site else ""
            lines.append(f"    {site:<20s}: {pct:.1f}%{marker}")
        lines.append("")
        if self.p_value is not None:
            lines.append(f"  Permutation p-value: {self.p_value:.4f}")
            if self.p_value < 0.05:
                lines.append("  Interpretation: site-contribution asymmetry is")
                lines.append("  statistically significant (p < 0.05).")
            else:
                lines.append("  Interpretation: site-contribution asymmetry is")
                lines.append("  not statistically significant at p < 0.05.")
                lines.append("  This may reflect limited statistical power")
                lines.append("  rather than absence of a real effect.")
        else:
            lines.append("  Permutation test: not run (n_permutations=0)")
        lines.append("=" * 55)
        return "\n".join(lines)

    def plot(self,
             figsize: Tuple[int, int] = None,
             show_permutation: bool = True,
             save_path: Optional[str] = None):
        """
        Plot the site-contribution bar chart, and optionally the permutation
        test null distribution alongside it.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Defaults to (6, 4) if show_permutation=False,
            or (12, 4.5) if show_permutation=True.
        show_permutation : bool
            Whether to show the permutation null distribution panel.
            Only available if n_permutations > 0. Default True.
        save_path : str, optional
            If provided, saves the figure to this path instead of displaying.
        """
        show_perm = show_permutation and self.null_distribution is not None

        if figsize is None:
            figsize = (12, 4.5) if show_perm else (6, 4)

        if show_perm:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax_bar = axes[0]
            ax_perm = axes[1]
        else:
            fig, ax_bar = plt.subplots(figsize=figsize)

        # --- bar chart ---
        sites = list(self.site_contribution_pct.keys())
        pcts = [self.site_contribution_pct[s] for s in sites]
        colors = ["steelblue" if s != self.dominant_site else "darkorange" for s in sites]
        ax_bar.bar(sites, pcts, color=colors)
        ax_bar.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% (equal contribution)")
        ax_bar.set_ylabel("% of total mean |SHAP value|")
        ax_bar.set_title(f"Site-attribution (n={self.n_subjects})")
        ax_bar.set_ylim(0, 100)
        ax_bar.legend(fontsize=8)

        # --- permutation panel ---
        if show_perm:
            max_pct = max(self.site_contribution_pct.values())
            ax_perm.hist(self.null_distribution, bins=20, color="gray",
                         edgecolor="black", alpha=0.7)
            ax_perm.axvline(max_pct, color="red", linestyle="--", linewidth=2,
                            label=f"Observed ({max_pct:.1f}%)")
            ax_perm.axvline(50, color="blue", linestyle=":",
                            label="50% (no asymmetry)")
            p_str = f"p = {self.p_value:.4f}" if self.p_value > 0 else "p < 0.005"
            ax_perm.set_title(f"Permutation null distribution\n{p_str}")
            ax_perm.set_xlabel("Max site share of SHAP importance (%)")
            ax_perm.set_ylabel("Count")
            ax_perm.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


# --------------------------------------------------------------------------
# Preprocessing utilities
# --------------------------------------------------------------------------

def clr_transform(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Apply centered log-ratio (CLR) transformation to a compositional
    microbiome abundance table.

    Parameters
    ----------
    df : pd.DataFrame
        Relative-abundance feature matrix (subjects x taxa).
        Values should be non-negative and sum to approximately 1 per row.
    pseudocount : float
        Small value added before log to handle zero abundances.
        Default 1e-6.

    Returns
    -------
    pd.DataFrame
        CLR-transformed feature matrix, same shape as input.
    """
    df = df.copy() + pseudocount
    log_df = np.log(df)
    geometric_mean = log_df.mean(axis=1)
    return log_df.sub(geometric_mean, axis=0)


def filter_low_variance(df: pd.DataFrame,
                        min_variance: float = 1e-4) -> pd.DataFrame:
    """
    Remove taxa (columns) with variance below a threshold across subjects.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (subjects x taxa).
    min_variance : float
        Minimum variance threshold. Default 1e-4.

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix with low-variance columns removed.
    """
    variances = df.var(axis=0)
    return df[variances[variances > min_variance].index]


def build_fused_matrix(
    site_tables: Dict[str, pd.DataFrame],
    apply_clr: bool = True,
    apply_variance_filter: bool = True,
    pseudocount: float = 1e-6,
    min_variance: float = 1e-4
) -> pd.DataFrame:
    """
    Build a fused feature matrix from per-site abundance tables, with
    optional CLR transformation and low-variance filtering.

    Parameters
    ----------
    site_tables : dict
        Dictionary mapping site names to DataFrames (subjects x taxa).
        Subjects must be in the index. Example:
        {"stool": stool_df, "oralcavity": oral_df}
    apply_clr : bool
        Whether to apply CLR transformation per site. Default True.
    apply_variance_filter : bool
        Whether to remove low-variance taxa per site. Default True.
    pseudocount : float
        Pseudocount for CLR transformation. Default 1e-6.
    min_variance : float
        Minimum variance threshold for filtering. Default 1e-4.

    Returns
    -------
    pd.DataFrame
        Fused feature matrix (subjects x all-site-features), with feature
        names prefixed by site name (e.g. "stool__taxon_A").
        Only subjects present in ALL sites are included.
    """
    processed = {}
    for site, df in site_tables.items():
        if apply_variance_filter:
            df = filter_low_variance(df, min_variance)
        if apply_clr:
            df = clr_transform(df, pseudocount)
        processed[site] = df

    common_subjects = set(processed[list(processed.keys())[0]].index)
    for df in processed.values():
        common_subjects &= set(df.index)
    common_subjects = sorted(common_subjects)

    aligned = {site: df.loc[common_subjects].add_prefix(f"{site}__")
               for site, df in processed.items()}

    return pd.concat(aligned.values(), axis=1)


# --------------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------------

def site_attribution_score(
    X_fused: pd.DataFrame,
    y: pd.Series,
    site_prefixes: Dict[str, str],
    model=None,
    n_permutations: int = 200,
    random_state: int = 42,
    verbose: bool = True
) -> SiteAttributionResult:
    """
    Compute SHAP-based site-attribution scores for a fused multi-site
    microbiome feature matrix, with an optional permutation significance test.

    This function trains a Random Forest classifier on the fused feature
    matrix, computes SHAP values for each subject, and attributes the total
    mean absolute SHAP importance to each body site based on feature name
    prefixes. It then optionally tests whether the observed site-contribution
    asymmetry is statistically distinguishable from random feature-to-site
    reassignment via a permutation test.

    Parameters
    ----------
    X_fused : pd.DataFrame
        Fused feature matrix (subjects x features) where each column name
        begins with a site-identifying prefix (e.g. "stool__", "oralcavity__").
        Can be built using build_fused_matrix() or constructed manually.
        Should already be preprocessed (CLR-transformed, variance-filtered).

    y : pd.Series or array-like
        Binary classification labels for each subject. Must align with
        X_fused's index. Values should be 0 or 1.

    site_prefixes : dict
        Mapping from site name to the column prefix used in X_fused.
        Example: {"stool": "stool__", "oralcavity": "oralcavity__"}
        Every feature in X_fused must match exactly one prefix.

    model : sklearn estimator, optional
        A fitted or unfitted sklearn-compatible classifier to use for SHAP
        computation. Must be compatible with shap.TreeExplainer (Random
        Forest, XGBoost, LightGBM, ExtraTrees, etc.). If None, uses a
        Random Forest with 300 trees. The model will be fit inside this
        function on the full X_fused and y.

    n_permutations : int
        Number of permutations for the significance test. Set to 0 to skip
        the permutation test. Default 200. For a final publication-ready
        result, 1000+ is recommended for stable p-values.

    random_state : int
        Random seed for the model and permutation test. Default 42.

    verbose : bool
        Whether to print progress messages. Default True.

    Returns
    -------
    SiteAttributionResult
        A result object with site_contribution_pct, p_value,
        null_distribution, and plot/summary methods.

    Examples
    --------
    Basic usage with a pre-built fused matrix:

        from microbiome_site_attribution import site_attribution_score

        result = site_attribution_score(
            X_fused=X_fused,
            y=y_binary,
            site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"},
            n_permutations=200
        )
        print(result.summary())
        result.plot()

    Using build_fused_matrix to construct the input:

        from microbiome_site_attribution import build_fused_matrix, site_attribution_score

        X_fused = build_fused_matrix(
            site_tables={"stool": stool_df, "oralcavity": oral_df},
            apply_clr=True,
            apply_variance_filter=True
        )
        result = site_attribution_score(
            X_fused=X_fused,
            y=y_binary,
            site_prefixes={"stool": "stool__", "oralcavity": "oralcavity__"}
        )

    Raises
    ------
    ValueError
        If any feature in X_fused doesn't match any prefix in site_prefixes,
        or if a prefix matches no features in X_fused.
    ImportError
        If shap is not installed.
    """
    try:
        import shap as shap_lib
    except ImportError:
        raise ImportError(
            "shap is required. Install it with: pip install shap"
        )

    from sklearn.ensemble import RandomForestClassifier

    X = X_fused.fillna(0)
    y_arr = np.array(y)
    feature_names = list(X.columns)
    n_subjects = len(X)

    # Validate prefixes
    site_names = list(site_prefixes.keys())
    n_features_per_site = {}
    for site, prefix in site_prefixes.items():
        idx = [i for i, f in enumerate(feature_names) if f.startswith(prefix)]
        if len(idx) == 0:
            raise ValueError(
                f"Prefix '{prefix}' for site '{site}' matches no features in X_fused. "
                f"Check that column names begin with the correct prefix."
            )
        n_features_per_site[site] = len(idx)

    unmatched = [f for f in feature_names
                 if not any(f.startswith(p) for p in site_prefixes.values())]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} features in X_fused don't match any prefix in "
            f"site_prefixes. Examples: {unmatched[:5]}. "
            f"Every column must begin with one of: {list(site_prefixes.values())}"
        )

    # Fit model
    if model is None:
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    model_name = type(model).__name__

    if verbose:
        print(f"Fitting {model_name} on {n_subjects} subjects, "
              f"{len(feature_names)} features...")
    model.fit(X, y_arr)

    # Compute SHAP values
    if verbose:
        print("Computing SHAP values...")
    explainer = shap_lib.TreeExplainer(model)
    raw_shap = explainer.shap_values(X)

    if isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_matrix = raw_shap[:, :, 1]
    elif isinstance(raw_shap, list):
        shap_matrix = np.array(raw_shap[1])
    elif hasattr(raw_shap, "values"):
        vals = raw_shap.values
        shap_matrix = vals[:, :, 1] if vals.ndim == 3 else vals
    else:
        shap_matrix = np.array(raw_shap)

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

    # Compute site contributions
    def compute_site_importance(mean_abs_shap_arr, feature_names_list, site_prefixes_dict):
        importance = {}
        for site, prefix in site_prefixes_dict.items():
            idx = [i for i, f in enumerate(feature_names_list) if f.startswith(prefix)]
            importance[site] = mean_abs_shap_arr[idx].sum()
        return importance

    site_importance_abs = compute_site_importance(
        mean_abs_shap, feature_names, site_prefixes
    )
    total = sum(site_importance_abs.values())
    site_importance_pct = {s: v / total * 100 for s, v in site_importance_abs.items()}
    dominant_site = max(site_importance_pct, key=site_importance_pct.get)

    # Permutation test
    p_value = None
    null_distribution = None

    if n_permutations > 0:
        if verbose:
            print(f"Running permutation test ({n_permutations} permutations)...")

        rng = np.random.RandomState(random_state)
        real_max_pct = max(site_importance_pct.values())
        all_idx = np.arange(len(feature_names))
        site_sizes = {s: n_features_per_site[s] for s in site_names}

        null_max_pcts = []
        for i in range(n_permutations):
            shuffled_idx = rng.permutation(all_idx)
            pos = 0
            perm_importance = {}
            for site in site_names:
                size = site_sizes[site]
                idx_for_site = shuffled_idx[pos:pos + size]
                perm_importance[site] = mean_abs_shap[idx_for_site].sum()
                pos += size
            perm_total = sum(perm_importance.values())
            perm_pct = {s: perm_importance[s] / perm_total * 100 for s in site_names}
            null_max_pcts.append(max(perm_pct.values()))

        null_distribution = np.array(null_max_pcts)
        p_value = (null_distribution >= real_max_pct).mean()

        if verbose:
            print(f"Permutation test complete. p-value: {p_value:.4f}")

    if verbose:
        print("Done.")

    return SiteAttributionResult(
        site_contribution_pct=site_importance_pct,
        site_contribution_abs=site_importance_abs,
        p_value=p_value,
        null_distribution=null_distribution,
        n_subjects=n_subjects,
        n_features_per_site=n_features_per_site,
        dominant_site=dominant_site,
        model_used=model_name,
        _site_names=site_names
    )


# --------------------------------------------------------------------------
# Cross-cohort comparison utility
# --------------------------------------------------------------------------

def compare_cohorts(
    results: Dict[str, SiteAttributionResult],
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
):
    """
    Plot a side-by-side bar chart comparing site-contribution percentages
    across multiple cohorts or datasets.

    Parameters
    ----------
    results : dict
        Mapping from cohort/dataset name to SiteAttributionResult.
        Example: {"Ferretti (n=44)": result1, "Brito (n=140)": result2}
    figsize : tuple, optional
        Figure size. Defaults to (5 * number of cohorts, 4).
    save_path : str, optional
        If provided, saves the figure to this path.

    Examples
    --------
        compare_cohorts({
            "Ferretti et al. (n=44)": ferretti_result,
            "Brito et al. (n=140)": brito_result
        })
    """
    n_cohorts = len(results)
    if figsize is None:
        figsize = (5 * n_cohorts, 4)

    fig, axes = plt.subplots(1, n_cohorts, figsize=figsize, sharey=True)
    if n_cohorts == 1:
        axes = [axes]

    for ax, (cohort_name, result) in zip(axes, results.items()):
        sites = list(result.site_contribution_pct.keys())
        pcts = [result.site_contribution_pct[s] for s in sites]
        colors = ["steelblue" if s != result.dominant_site else "darkorange"
                  for s in sites]
        ax.bar(sites, pcts, color=colors)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        p_str = f"p = {result.p_value:.4f}" if result.p_value is not None and result.p_value > 0 \
            else ("p < 0.005" if result.p_value == 0.0 else "")
        ax.set_title(f"{cohort_name}\n{p_str}", fontsize=10)
        ax.set_ylabel("% of total SHAP importance" if ax == axes[0] else "")
        for s, pct in zip(sites, pcts):
            ax.text(sites.index(s), pct + 1.5, f"{pct:.1f}%", ha="center", fontsize=9)

    plt.suptitle("SHAP Site-Attribution Across Cohorts", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
