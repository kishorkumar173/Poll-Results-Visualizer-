"""
Script: main.py
Description: Master CLI Orchestrator for the Poll Results Visualizer.
Runs complete end-to-end data pipeline:
Data Generation/Ingestion -> Cleaning -> Statistical Analysis -> Chart Generation -> Executive Insights.
"""

import os
import sys
import pandas as pd

# Support UTF-8 encoding in Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure package modules can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_synthetic_poll, generate_and_save_all_datasets
from src.data_processor import PollDataProcessor
from src.poll_analyzer import PollAnalyzer
from src.text_analyzer import TextAnalyzer
from src.visualizer import PollVisualizer
from src.insights_engine import InsightsEngine

def run_pipeline():
    print("=" * 80)
    print("      [*] POLL RESULTS VISUALIZER - END-TO-END ANALYTICS PIPELINE")
    print("=" * 80)

    # Setup directories
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    charts_dir = os.path.join(PROJECT_ROOT, "outputs", "charts")
    reports_dir = os.path.join(PROJECT_ROOT, "outputs", "reports")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Data Generation & Ingestion
    # ---------------------------------------------------------
    print("\n[Phase 1 & 2] Generating & Loading Survey Datasets...")
    generate_and_save_all_datasets(raw_dir)

    primary_raw_file = os.path.join(raw_dir, "workplace_pulse_raw.csv")
    print(f"Loading primary survey dataset: {primary_raw_file}")

    processor = PollDataProcessor(primary_raw_file)
    processor.load_data(primary_raw_file)

    # ---------------------------------------------------------
    # STEP 2: Data Audit & Cleaning
    # ---------------------------------------------------------
    print("\n[Phase 3] Auditing and Preprocessing Raw Poll Data...")
    audit = processor.audit_raw_data()
    print(f"  - Initial Records: {audit['initial_rows']}")
    print(f"  - Duplicate Records Found: {audit['duplicate_records']}")
    print(f"  - Columns: {audit['initial_columns']}")

    cleaned_df, clean_audit = processor.clean_pipeline()
    cleaned_file = os.path.join(proc_dir, "cleaned_poll_data.csv")
    processor.save_cleaned_data(cleaned_file)

    print(f"  - Cleaned Records Retained: {clean_audit['final_rows']} ({clean_audit['retention_rate_pct']}%)")
    print(f"  - Successfully sanitized demographic labels, parsed dates, and classified NPS segments.")

    # ---------------------------------------------------------
    # STEP 3: Statistical Analytics Engine
    # ---------------------------------------------------------
    print("\n[Phase 4 & 5] Performing Statistical Poll Analysis & Hypothesis Testing...")
    analyzer = PollAnalyzer(cleaned_df)

    # Option Vote Summary with Margin of Error
    summary_df = analyzer.calculate_option_summary("selected_option")
    summary_path = os.path.join(proc_dir, "option_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n--- OVERALL VOTE SHARE & 95% CONFIDENCE INTERVAL ---")
    print(summary_df[["selected_option", "vote_count", "percentage_share", "moe_pct", "mean_satisfaction"]].to_string(index=False))

    # Regional Cross-Tabulation
    regional_crosstab = analyzer.calculate_cross_tabulation(row_dim="region", col_dim="selected_option")
    crosstab_path = os.path.join(proc_dir, "regional_cross_tab.csv")
    regional_crosstab.to_csv(crosstab_path)
    print("\n--- REGIONAL CROSS-TABULATION (% SHARE BY REGION) ---")
    print(regional_crosstab.to_string())

    # Pearson's Chi-Square Test of Independence
    chi_res = analyzer.test_chi_square_independence("age_group", "selected_option")
    print(f"\n--- HYPOTHESIS TEST: CHI-SQUARE INDEPENDENCE (Age vs Option) ---")
    print(f"  - Chi2 Statistic: {chi_res['chi2_statistic']}")
    print(f"  - p-value: {chi_res['p_value']:.4e}")
    print(f"  - Degrees of Freedom: {chi_res['degrees_of_freedom']}")
    print(f"  - Verdict: {chi_res['interpretation']}")

    # Net Promoter Score (NPS)
    nps_res = analyzer.calculate_nps("nps_score")
    print(f"\n--- NET PROMOTER SCORE (NPS) METRICS ---")
    print(f"  - NPS Score: {nps_res['nps']} ({nps_res['benchmark_rating']})")
    print(f"  - Promoters: {nps_res['promoters_pct']}% | Passives: {nps_res['passives_pct']}% | Detractors: {nps_res['detractors_pct']}%")

    # Time trends
    time_trends = analyzer.calculate_time_trends("response_date", "selected_option")

    # ---------------------------------------------------------
    # STEP 4: Natural Language Processing (Open-ended Feedback)
    # ---------------------------------------------------------
    print("\n[Phase 5.1] Processing Qualitative Comments & Sentiment Mining...")
    text_engine = TextAnalyzer(cleaned_df["feedback_comment"].dropna().tolist())
    sentiments = text_engine.analyze_sentiments()
    top_words = text_engine.get_top_keywords(12)
    print(f"  - Analyzed {sentiments['total_comments']} subjective comments.")
    print(f"  - Sentiment Distribution: {sentiments['sentiment_percentages']}")
    print(f"  - Top Feedback Keywords: {[w for w, _ in top_words[:6]]}")

    # ---------------------------------------------------------
    # STEP 5: Data Visualizations (High-Res Static Charts)
    # ---------------------------------------------------------
    print("\n[Phase 6] Generating Publication-Quality Data Visualizations...")
    visualizer = PollVisualizer(charts_dir)

    c1 = visualizer.plot_bar_vote_distribution(summary_df, save_filename="01_vote_distribution_bar.png")
    c2 = visualizer.plot_pie_distribution(summary_df, save_filename="02_vote_share_donut.png")
    c3 = visualizer.plot_stacked_demographics(regional_crosstab, dim_label="Region", save_filename="03_demographic_stacked_bar.png")
    c4 = visualizer.plot_time_trend(time_trends, save_filename="04_response_timeline_trend.png")
    all_text = " ".join(cleaned_df["feedback_comment"].dropna().tolist())
    c5 = visualizer.generate_wordcloud(all_text, save_filename="05_feedback_wordcloud.png")

    print(f"  [SAVED]: {os.path.basename(c1)}")
    print(f"  [SAVED]: {os.path.basename(c2)}")
    print(f"  [SAVED]: {os.path.basename(c3)}")
    print(f"  [SAVED]: {os.path.basename(c4)}")
    print(f"  [SAVED]: {os.path.basename(c5)}")

    # ---------------------------------------------------------
    # STEP 6: Automated Executive Insights Generation
    # ---------------------------------------------------------
    print("\n[Phase 7] Generating Strategic Executive Insights & Decision Brief...")
    insights_engine = InsightsEngine(
        summary_df=summary_df,
        nps_data=nps_res,
        chi_square_results=chi_res,
        sentiment_data=sentiments,
        option_col="selected_option"
    )
    brief_file = os.path.join(reports_dir, "Executive_Poll_Summary.txt")
    brief_text = insights_engine.export_text_brief(brief_file)

    print(f"\n{brief_text}")
    print(f"[SUCCESS] Executive summary exported to: {brief_file}")

    print("\n" + "=" * 80)
    print("  [SUCCESS] PIPELINE EXECUTION COMPLETED!")
    print(f"  - Cleaned Data: {proc_dir}")
    print(f"  - Generated Charts: {charts_dir}")
    print(f"  - Executive Report: {brief_file}")
    print("  - To launch interactive Web App run: streamlit run app.py")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
