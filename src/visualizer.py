"""
Module: visualizer.py
Description: Comprehensive visualization suite for poll and survey datasets.
Generates publication-grade static figures (Matplotlib / Seaborn) saved as high-res PNGs,
and rich interactive figures (Plotly) for web apps and dashboards.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from typing import Optional

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})

class PollVisualizer:
    """
    Visualization engine rendering bar, pie, stacked demographic, trend charts, and word clouds.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    # ==========================================
    # STATIC CHARTS (Matplotlib / Seaborn)
    # ==========================================

    def plot_bar_vote_distribution(
        self,
        summary_df: pd.DataFrame,
        option_col: str = "selected_option",
        save_filename: str = "01_vote_distribution_bar.png"
    ) -> str:
        """
        Generates horizontal bar chart with percentage labels and Margin of Error (MOE) intervals.
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Plot horizontal bars
        bars = ax.barh(
            summary_df[option_col],
            summary_df["percentage_share"],
            xerr=summary_df["moe_pct"] if "moe_pct" in summary_df.columns else None,
            capsize=4,
            color=sns.color_palette("mako", len(summary_df)),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9
        )

        ax.invert_yaxis()  # Highest vote on top
        ax.set_xlabel("Vote Share (%) [with 95% Confidence Interval]", fontweight="bold", labelpad=10)
        ax.set_title("Overall Poll Results: Vote Distribution & Margin of Error", pad=15)
        ax.set_xlim(0, max(summary_df["percentage_share"]) + 15)

        # Annotate exact percentages and counts
        for bar, (_, row) in zip(bars, summary_df.iterrows()):
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height() / 2
            label_text = f" {row['percentage_share']:.1f}% ({int(row['vote_count']):,} votes)"
            if "moe_pct" in row:
                label_text += f" ±{row['moe_pct']:.1f}%"
            ax.text(width + 1.2, y_pos, label_text, va="center", ha="left", fontweight="bold", fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename) if self.output_dir else save_filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_pie_distribution(
        self,
        summary_df: pd.DataFrame,
        option_col: str = "selected_option",
        save_filename: str = "02_vote_share_donut.png"
    ) -> str:
        """
        Generates a modern Donut Chart showing proportion of vote share.
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        colors = sns.color_palette("Set2", len(summary_df))
        wedges, texts, autotexts = ax.pie(
            summary_df["percentage_share"],
            labels=summary_df[option_col],
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
        )

        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.setp(texts, size=10, weight="medium")
        ax.set_title("Proportional Vote Share (Donut Breakdown)", pad=20)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename) if self.output_dir else save_filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_stacked_demographics(
        self,
        crosstab_df: pd.DataFrame,
        dim_label: str = "Region",
        save_filename: str = "03_demographic_stacked_bar.png"
    ) -> str:
        """
        Generates a 100% Stacked Bar Chart for demographic cross-tabulation.
        """
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        # Normalized cross-tabulation plotted
        crosstab_df.plot(
            kind="barh",
            stacked=True,
            ax=ax,
            colormap="Spectral",
            edgecolor="white",
            linewidth=0.8
        )

        ax.set_xlabel("Percentage Distribution (%)", fontweight="bold")
        ax.set_ylabel(dim_label, fontweight="bold")
        ax.set_title(f"Poll Preference Breakdown Across {dim_label} (100% Stacked)", pad=15)
        ax.legend(title="Selected Option", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.set_xlim(0, 100)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename) if self.output_dir else save_filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_time_trend(
        self,
        trend_df: pd.DataFrame,
        save_filename: str = "04_response_timeline_trend.png"
    ) -> str:
        """
        Generates longitudinal line chart tracking daily/weekly survey participation.
        """
        if trend_df.empty or "response_date" not in trend_df.columns:
            return ""

        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

        trend_df["response_date"] = pd.to_datetime(trend_df["response_date"])
        ax.plot(
            trend_df["response_date"],
            trend_df["daily_total"],
            marker="o",
            markersize=5,
            linewidth=2.5,
            color="#2b5c8f",
            label="Daily Total Responses"
        )

        ax.fill_between(
            trend_df["response_date"],
            trend_df["daily_total"],
            color="#2b5c8f",
            alpha=0.15
        )

        ax.set_xlabel("Date of Submission", fontweight="bold", labelpad=10)
        ax.set_ylabel("Number of Responses", fontweight="bold", labelpad=10)
        ax.set_title("Longitudinal Survey Participation Velocity", pad=15)
        plt.xticks(rotation=30)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename) if self.output_dir else save_filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def generate_wordcloud(
        self,
        text_corpus: str,
        save_filename: str = "05_feedback_wordcloud.png"
    ) -> str:
        """
        Renders a high-resolution Word Cloud from open-ended comments.
        """
        if not text_corpus.strip():
            text_corpus = "Satisfied Product Experience Reliable Fast Responsive"

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=80,
            contour_width=1,
            contour_color="steelblue"
        ).generate(text_corpus)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Survey Qualitative Feedback: Key Sentiment Themes", fontsize=14, weight="bold", pad=12)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename) if self.output_dir else save_filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # ==========================================
    # INTERACTIVE CHARTS (Plotly)
    # ==========================================

    @staticmethod
    def interactive_bar_chart(summary_df: pd.DataFrame, option_col: str = "selected_option"):
        """
        Returns Plotly Interactive Bar Chart with tooltips.
        """
        fig = px.bar(
            summary_df,
            x="percentage_share",
            y=option_col,
            orientation="h",
            color="percentage_share",
            color_continuous_scale="Blues",
            error_x="moe_pct" if "moe_pct" in summary_df.columns else None,
            labels={"percentage_share": "Vote Share (%)", option_col: "Option"},
            title="Interactive Poll Distribution (Hover for Margin of Error & Vote Counts)",
            text=summary_df["percentage_share"].apply(lambda v: f"{v:.1f}%")
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
        return fig

    @staticmethod
    def interactive_sunburst_demographics(df: pd.DataFrame):
        """
        Returns Plotly Interactive Sunburst hierarchy (Region -> Age Group -> Selected Option).
        """
        cols = ["region", "age_group", "selected_option"]
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) < 2:
            return None

        fig = px.sunburst(
            df,
            path=valid_cols,
            title="Hierarchical Demographic Breakdown (Region → Age → Selected Option)",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        return fig
