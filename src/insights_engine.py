"""
Module: insights_engine.py
Description: Automated insight extraction engine that synthesizes statistical data,
margin of error, demographic biases, and sentiment metrics into executive summaries
and strategic decision-making recommendations.
"""

import pandas as pd
from typing import Dict, Any, List

class InsightsEngine:
    """
    Transforms raw statistical outputs into actionable, business-ready narrative briefs.
    """

    def __init__(
        self,
        summary_df: pd.DataFrame,
        nps_data: Dict[str, Any],
        chi_square_results: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        option_col: str = "selected_option"
    ):
        self.summary_df = summary_df.sort_values(by="percentage_share", ascending=False).reset_index(drop=True)
        self.nps_data = nps_data
        self.chi_square = chi_square_results
        self.sentiment_data = sentiment_data
        self.option_col = option_col

    def generate_executive_insights(self) -> Dict[str, Any]:
        """
        Synthesizes all analytics modules into structured insights.
        """
        top_option = self.summary_df.iloc[0]
        runner_up = self.summary_df.iloc[1] if len(self.summary_df) > 1 else None

        lead_margin = round(top_option["percentage_share"] - runner_up["percentage_share"], 2) if runner_up is not None else 0
        combined_moe = round(top_option.get("moe_pct", 0) + runner_up.get("moe_pct", 0), 2) if runner_up is not None else 0

        # Assess statistical decisiveness
        is_statistically_decisive = lead_margin > combined_moe

        if is_statistically_decisive:
            lead_verdict = (
                f"Statistically Decisive Lead: '{top_option[self.option_col]}' leads by {lead_margin}% "
                f"(exceeding combined Margin of Error of ±{combined_moe}%)."
            )
        else:
            lead_verdict = (
                f"Statistical Dead Heat / Contested: '{top_option[self.option_col]}' leads by {lead_margin}%, "
                f"which falls within the combined Margin of Error (±{combined_moe}%)."
            )

        # NPS diagnosis
        nps_val = self.nps_data.get("nps", 0)
        nps_assessment = (
            f"Strong Advocate Base (+{nps_val}) - Organization enjoys high loyalty and positive word-of-mouth."
            if nps_val >= 35 else
            f"Moderate Alignment (+{nps_val}) - Acceptable baseline, but requires proactive mitigation of detractors."
            if nps_val >= 10 else
            f"Vulnerability Alert ({nps_val}) - Detractors outnumber or closely match promoters. Critical intervention needed."
        )

        # Sentiment breakdown
        sent_pct = self.sentiment_data.get("sentiment_percentages", {"Positive": 0, "Neutral": 0, "Negative": 0})
        sentiment_summary = (
            f"Qualitative feedback leans {sent_pct['Positive']}% Positive, {sent_pct['Neutral']}% Neutral, "
            f"and {sent_pct['Negative']}% Negative across open-ended responses."
        )

        # Demographic significance
        demo_significance = self.chi_square.get("interpretation", "No demographic test available.")

        # Prescriptive Strategic Recommendations
        recommendations = [
            f"Adopt the consensus choice: Allocate strategic resources primarily toward '{top_option[self.option_col]}' ({top_option['percentage_share']:.1f}% share).",
            f"Address cohort polarization: Tailor regional or demographic rollouts as indicated by Chi-Square test ({demo_significance.split(':')[0]}).",
            f"Mitigate friction points: Focus on negative feedback drivers ({sent_pct['Negative']}% of comments) to elevate NPS beyond +{nps_val}.",
            "Continuous pulse measurement: Re-survey after 90 days to assess velocity shifts and policy adoption rates."
        ]

        return {
            "top_choice": top_option[self.option_col],
            "top_choice_share": top_option["percentage_share"],
            "lead_margin": lead_margin,
            "lead_verdict": lead_verdict,
            "nps_score": nps_val,
            "nps_assessment": nps_assessment,
            "sentiment_summary": sentiment_summary,
            "demographic_finding": demo_significance,
            "strategic_recommendations": recommendations
        }

    def export_text_brief(self, filepath: str) -> str:
        """
        Exports a formatted executive briefing document to disk.
        """
        insights = self.generate_executive_insights()

        brief = f"""========================================================================
POLL RESULTS VISUALIZER - EXECUTIVE INSIGHTS BRIEFING REPORT
========================================================================

1. EXECUTIVE SUMMARY & WINNING SELECTION:
   - Primary Consensus: {insights['top_choice']} ({insights['top_choice_share']:.1f}% Vote Share)
   - Victory Margin: {insights['lead_margin']}% over runner-up
   - Reliability: {insights['lead_verdict']}

2. CUSTOMER & STAKEHOLDER SENTIMENT (NPS & NLP):
   - Net Promoter Score (NPS): {insights['nps_score']} ({self.nps_data.get('benchmark_rating', 'N/A')})
   - NPS Diagnostic: {insights['nps_assessment']}
   - Feedback Sentiment: {insights['sentiment_summary']}

3. DEMOGRAPHIC DEPENDENCY & BIAS AUDIT:
   - Statistical Test: Pearson's Chi-Square Test of Independence
   - Finding: {insights['demographic_finding']}

4. STRATEGIC & ACTIONABLE RECOMMENDATIONS:
"""
        for i, rec in enumerate(insights["strategic_recommendations"], 1):
            brief += f"   [{i}] {rec}\n"

        brief += "\n========================================================================\n"
        brief += "Report Generated Automatically by Poll Results Visualizer v1.0\n"
        brief += "========================================================================\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(brief)

        return brief
