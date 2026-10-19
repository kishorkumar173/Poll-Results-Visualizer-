"""
Module: poll_analyzer.py
Description: Mathematical and statistical analytics engine for poll & survey data.
Calculates vote shares, margins of error (MOE) at 95% confidence interval,
demographic cross-tabulations, Chi-Square tests of independence,
Net Promoter Score (NPS), and longitudinal trends.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple

class PollAnalyzer:
    """
    Advanced statistical analysis suite for survey research.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.total_responses = len(df)

    def calculate_option_summary(self, option_col: str = "selected_option") -> pd.DataFrame:
        """
        Calculates vote counts, percentage shares, and 95% Confidence Interval
        Margin of Error (MOE) for each option.

        MOE Formula: 1.96 * sqrt( (p * (1 - p)) / N )
        """
        if option_col not in self.df.columns:
            raise KeyError(f"Column '{option_col}' not found in dataset.")

        summary = self.df[option_col].value_counts().reset_index()
        summary.columns = [option_col, "vote_count"]
        summary["percentage_share"] = (summary["vote_count"] / self.total_responses) * 100

        # Margin of Error (MOE) at 95% Confidence level (Z = 1.96)
        p = summary["percentage_share"] / 100.0
        n = self.total_responses
        moe = 1.96 * np.sqrt((p * (1 - p)) / n) * 100
        summary["moe_pct"] = moe.round(2)
        summary["ci_lower_pct"] = (summary["percentage_share"] - moe).clip(lower=0).round(2)
        summary["ci_upper_pct"] = (summary["percentage_share"] + moe).clip(upper=100).round(2)

        # Compute average rating per option if available
        if "satisfaction_rating" in self.df.columns:
            avg_rating = self.df.groupby(option_col)["satisfaction_rating"].mean().reset_index()
            avg_rating.columns = [option_col, "mean_satisfaction"]
            summary = summary.merge(avg_rating, on=option_col, how="left")
            summary["mean_satisfaction"] = summary["mean_satisfaction"].round(2)

        return summary.sort_values(by="vote_count", ascending=False).reset_index(drop=True)

    def calculate_cross_tabulation(
        self,
        row_dim: str = "region",
        col_dim: str = "selected_option",
        normalize: str = "index"
    ) -> pd.DataFrame:
        """
        Generates cross-tabulation table between two categorical variables.
        normalize='index' calculates row percentages (e.g., % of each region choosing option X).
        """
        if row_dim not in self.df.columns or col_dim not in self.df.columns:
            raise KeyError(f"Both {row_dim} and {col_dim} must exist in the dataset.")

        if normalize == "index":
            ct = pd.crosstab(self.df[row_dim], self.df[col_dim], normalize="index") * 100
        elif normalize == "columns":
            ct = pd.crosstab(self.df[row_dim], self.df[col_dim], normalize="columns") * 100
        else:
            ct = pd.crosstab(self.df[row_dim], self.df[col_dim])

        return ct.round(2)

    def test_chi_square_independence(
        self,
        var1: str = "age_group",
        var2: str = "selected_option"
    ) -> Dict[str, Any]:
        """
        Conducts Pearson's Chi-Square Test of Independence to determine if
        demographic segment has a statistically significant relationship with option choice.
        """
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

        is_significant = p_val < 0.05
        interpretation = (
            f"Statistically Significant (p = {p_val:.4e} < 0.05): "
            f"Evidence indicates a strong association between {var1} and {var2}."
            if is_significant else
            f"Not Statistically Significant (p = {p_val:.4f} >= 0.05): "
            f"No compelling evidence of dependency between {var1} and {var2}."
        )

        return {
            "chi2_statistic": round(chi2, 4),
            "p_value": p_val,
            "degrees_of_freedom": dof,
            "is_significant": is_significant,
            "interpretation": interpretation
        }

    def calculate_nps(self, nps_col: str = "nps_score") -> Dict[str, Any]:
        """
        Computes Net Promoter Score (NPS):
        % Promoters (9-10) - % Detractors (0-6)
        """
        if nps_col not in self.df.columns:
            return {"nps": None, "promoters_pct": 0, "passives_pct": 0, "detractors_pct": 0}

        scores = self.df[nps_col].dropna()
        n = len(scores)
        if n == 0:
            return {"nps": 0, "promoters_pct": 0, "passives_pct": 0, "detractors_pct": 0}

        promoters = (scores >= 9).sum()
        passives = ((scores >= 7) & (scores <= 8)).sum()
        detractors = (scores <= 6).sum()

        p_pct = (promoters / n) * 100
        pas_pct = (passives / n) * 100
        d_pct = (detractors / n) * 100
        nps = p_pct - d_pct

        return {
            "nps": round(nps, 1),
            "promoters_pct": round(p_pct, 1),
            "passives_pct": round(pas_pct, 1),
            "detractors_pct": round(d_pct, 1),
            "benchmark_rating": (
                "World Class (Excellent)" if nps >= 50 else
                "Great (Healthy)" if nps >= 30 else
                "Good (Average)" if nps >= 0 else
                "Needs Attention (Critical)"
            )
        }

    def calculate_time_trends(
        self,
        date_col: str = "response_date",
        option_col: str = "selected_option"
    ) -> pd.DataFrame:
        """
        Aggregates daily response volume and leading option over time.
        """
        if date_col not in self.df.columns or option_col not in self.df.columns:
            return pd.DataFrame()

        trend = self.df.groupby([date_col, option_col]).size().unstack(fill_value=0)
        trend["daily_total"] = trend.sum(axis=1)
        return trend.reset_index()
