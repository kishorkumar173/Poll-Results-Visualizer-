"""
Interactive Web Application: app.py
Description: Full-featured, modern Streamlit interactive dashboard for the
Poll Results Visualizer. Features multi-survey loading, custom CSV/Google Form upload,
dynamic cohort filtering, statistical hypothesis testing, NLP text mining, and automated briefings.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_synthetic_poll
from src.data_processor import PollDataProcessor
from src.poll_analyzer import PollAnalyzer
from src.text_analyzer import TextAnalyzer
from src.visualizer import PollVisualizer
from src.insights_engine import InsightsEngine

# Page configuration
st.set_page_config(
    page_title="Poll Results Visualizer | Survey Insights Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS: DATA SOURCE & COHORT FILTERS
# -------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=64)
st.sidebar.title("Survey Configuration")

data_source = st.sidebar.radio(
    "Select Survey Dataset:",
    [
        "💼 Workplace Pulse (Sample)",
        "📱 Consumer Tech Preferences",
        "🏛️ Public Policy & Transit",
        "📂 Upload Google Form / CSV"
    ]
)

# Load data based on selection
@st.cache_data
def get_dataset(source_type: str):
    if "Workplace" in source_type:
        raw_df = generate_synthetic_poll(num_responses=1400, survey_type="workplace", random_seed=42)
    elif "Consumer" in source_type:
        raw_df = generate_synthetic_poll(num_responses=1200, survey_type="consumer_tech", random_seed=42)
    elif "Policy" in source_type:
        raw_df = generate_synthetic_poll(num_responses=1000, survey_type="public_policy", random_seed=42)
    else:
        return None
    
    proc = PollDataProcessor()
    proc.load_data(raw_df)
    proc.audit_raw_data()
    clean_df, _ = proc.clean_pipeline()
    return clean_df

if "Upload" in data_source:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel Survey File", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            proc = PollDataProcessor()
            proc.load_data(raw_df)
            proc.audit_raw_data()
            df, audit = proc.clean_pipeline()
            st.sidebar.success(f"Loaded {len(df)} cleaned rows!")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")
            df = get_dataset("💼 Workplace Pulse (Sample)")
    else:
        st.info("👆 Upload your survey data or explore using the sample surveys above.")
        df = get_dataset("💼 Workplace Pulse (Sample)")
else:
    df = get_dataset(data_source)

# -------------------------------------------------------------
# DYNAMIC COHORT FILTERS
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Dynamic Cohort Filtering")

filtered_df = df.copy()

# Region filter
if "region" in df.columns:
    all_regions = sorted(df["region"].unique().tolist())
    selected_regions = st.sidebar.multiselect("Region:", all_regions, default=all_regions)
    if selected_regions:
        filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]

# Age filter
if "age_group" in df.columns:
    all_ages = sorted(df["age_group"].unique().tolist())
    selected_ages = st.sidebar.multiselect("Age Group:", all_ages, default=all_ages)
    if selected_ages:
        filtered_df = filtered_df[filtered_df["age_group"].isin(selected_ages)]

# Gender filter
if "gender" in df.columns:
    all_genders = sorted(df["gender"].unique().tolist())
    selected_genders = st.sidebar.multiselect("Gender:", all_genders, default=all_genders)
    if selected_genders:
        filtered_df = filtered_df[filtered_df["gender"].isin(selected_genders)]

st.sidebar.markdown("---")
st.sidebar.caption("Poll Results Visualizer v1.0 • Built for Industry Analytics")

# -------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------
st.markdown('<div class="main-header">📊 Poll Results Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive Decision-Support Analytics Platform for Public Opinion, '
    'Market Research, and Voice-of-Customer Polls</div>',
    unsafe_allow_html=True
)

if len(filtered_df) == 0:
    st.warning("⚠️ No survey responses match the selected filters. Please adjust the sidebar filter criteria.")
    st.stop()

# -------------------------------------------------------------
# ANALYTICS ENGINE COMPUTATIONS
# -------------------------------------------------------------
analyzer = PollAnalyzer(filtered_df)
summary_df = analyzer.calculate_option_summary("selected_option")
nps_results = analyzer.calculate_nps("nps_score")
chi_results = (
    analyzer.test_chi_square_independence("age_group", "selected_option")
    if "age_group" in filtered_df.columns and len(filtered_df["age_group"].unique()) > 1
    else {"interpretation": "Insufficient categories for Chi-Square test", "is_significant": False, "p_value": 1.0}
)

text_engine = TextAnalyzer(filtered_df["feedback_comment"].dropna().tolist() if "feedback_comment" in filtered_df.columns else [])
sentiments = text_engine.analyze_sentiments()

# -------------------------------------------------------------
# TOP KPI METRICS SCORECARD
# -------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="Total Responses",
        value=f"{len(filtered_df):,}",
        delta=f"{round(len(filtered_df)/len(df)*100, 1)}% of total" if len(filtered_df) != len(df) else "100% active"
    )

with kpi2:
    top_choice = summary_df.iloc[0]["selected_option"]
    top_share = summary_df.iloc[0]["percentage_share"]
    st.metric(label="Leading Preference", value=f"{top_share:.1f}%", delta=top_choice[:22])

with kpi3:
    avg_moe = summary_df.iloc[0]["moe_pct"] if "moe_pct" in summary_df.columns else 2.5
    st.metric(label="Margin of Error (95% CI)", value=f"±{avg_moe:.1f}%", delta="Sample Rigor", delta_color="off")

with kpi4:
    nps_val = nps_results["nps"]
    st.metric(label="Net Promoter Score", value=f"{nps_val:+0.1f}", delta=nps_results["benchmark_rating"][:15])

with kpi5:
    pos_pct = sentiments["sentiment_percentages"]["Positive"]
    st.metric(label="Positive Sentiment", value=f"{pos_pct}%", delta=f"{sentiments['sentiment_counts']['Positive']} comments")

st.markdown("---")

# -------------------------------------------------------------
# TABBED ANALYTICS EXPLORATION
# -------------------------------------------------------------
tab_overview, tab_demographics, tab_trends, tab_nlp, tab_executive = st.tabs([
    "📈 Overall Results",
    "👥 Demographic Cross-Tabs",
    "📅 Longitudinal Trends",
    "💬 Voice of Customer (NLP)",
    "📑 Executive Decision Brief"
])

# -------------------------------------------------------------
# TAB 1: OVERALL RESULTS
# -------------------------------------------------------------
with tab_overview:
    col_chart, col_donut = st.columns([1.3, 1])

    with col_chart:
        st.subheader("Vote Distribution with 95% Confidence Intervals")
        fig_bar = px.bar(
            summary_df,
            x="percentage_share",
            y="selected_option",
            orientation="h",
            color="percentage_share",
            color_continuous_scale="Blues",
            error_x="moe_pct",
            text=summary_df["percentage_share"].apply(lambda x: f"{x:.1f}%"),
            labels={"percentage_share": "Vote Share (%)", "selected_option": "Option"},
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=420,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_donut:
        st.subheader("Proportional Share")
        fig_pie = px.pie(
            summary_df,
            names="selected_option",
            values="vote_count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template="plotly_white", height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Detailed Poll Vote Breakdown & Confidence Intervals")
    st.dataframe(
        summary_df.style.format({
            "percentage_share": "{:.2f}%",
            "moe_pct": "±{:.2f}%",
            "ci_lower_pct": "{:.2f}%",
            "ci_upper_pct": "{:.2f}%",
            "mean_satisfaction": "{:.2f} / 5.0"
        }),
        use_container_width=True
    )

# -------------------------------------------------------------
# TAB 2: DEMOGRAPHIC CROSS-TABS
# -------------------------------------------------------------
with tab_demographics:
    st.subheader("Demographic Segmentation & Cross-Tabulations")
    
    col_demo1, col_demo2 = st.columns(2)

    with col_demo1:
        demo_feature = st.selectbox("Select Demographic Variable to Cross-Tabulate:", ["region", "age_group", "gender", "education"])
        if demo_feature in filtered_df.columns:
            ct = analyzer.calculate_cross_tabulation(row_dim=demo_feature, col_dim="selected_option")
            st.write(f"**Cross-Tabulation (% Share within each {demo_feature.title()}):**")
            st.dataframe(ct.style.background_gradient(cmap="Blues", axis=1).format("{:.1f}%"), use_container_width=True)

    with col_demo2:
        st.write(f"**100% Stacked Preference Distribution ({demo_feature.title()}):**")
        if demo_feature in filtered_df.columns:
            ct_reset = ct.reset_index().melt(id_vars=demo_feature, var_name="Option", value_name="Percentage")
            fig_stack = px.bar(
                ct_reset,
                x="Percentage",
                y=demo_feature,
                color="Option",
                orientation="h",
                barmode="stack",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_stack.update_layout(template="plotly_white", height=340, legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Statistical Hypothesis Testing (Chi-Square Independence)")
    st.info(f"**Hypothesis Test Analysis:** {chi_results['interpretation']}")
    if "chi2_statistic" in chi_results:
        st.caption(f"Chi-Square Statistic: {chi_results['chi2_statistic']} | p-value: {chi_results['p_value']:.4e} | Degrees of Freedom: {chi_results['degrees_of_freedom']}")

# -------------------------------------------------------------
# TAB 3: LONGITUDINAL TRENDS
# -------------------------------------------------------------
with tab_trends:
    st.subheader("Survey Velocity & Longitudinal Response Timeline")
    if "response_date" in filtered_df.columns:
        daily_counts = filtered_df.groupby("response_date").size().reset_index(name="responses")
        daily_counts["response_date"] = pd.to_datetime(daily_counts["response_date"])
        daily_counts["cumulative"] = daily_counts["responses"].cumsum()

        fig_trend = px.area(
            daily_counts,
            x="response_date",
            y="responses",
            title="Daily Response Submissions",
            labels={"response_date": "Date", "responses": "Submissions"},
            color_discrete_sequence=["#2563EB"]
        )
        fig_trend.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.write("No date/timestamp column available in the current dataset.")

# -------------------------------------------------------------
# TAB 4: VOICE OF CUSTOMER (NLP)
# -------------------------------------------------------------
with tab_nlp:
    st.subheader("Qualitative Feedback & Sentiment Mining")
    
    col_sent, col_wc = st.columns([1, 1.3])

    with col_sent:
        st.write("**Sentiment Polarity Breakdown:**")
        sent_df = pd.DataFrame(list(sentiments["sentiment_percentages"].items()), columns=["Sentiment", "Percentage"])
        fig_sent = px.pie(
            sent_df,
            names="Sentiment",
            values="Percentage",
            color="Sentiment",
            color_discrete_map={"Positive": "#10B981", "Neutral": "#6B7280", "Negative": "#EF4444"},
            hole=0.4
        )
        fig_sent.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_wc:
        st.write("**Key Thematic Keywords:**")
        top_kws = text_engine.get_top_keywords(12)
        kw_df = pd.DataFrame(top_kws, columns=["Keyword", "Frequency"])
        fig_kw = px.bar(kw_df, x="Frequency", y="Keyword", orientation="h", color="Frequency", color_continuous_scale="Viridis")
        fig_kw.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=320)
        st.plotly_chart(fig_kw, use_container_width=True)

    st.subheader("Sample Qualitative Feedback Comments")
    if "feedback_comment" in filtered_df.columns:
        st.dataframe(filtered_df[["selected_option", "satisfaction_rating", "feedback_comment"]].dropna().head(15), use_container_width=True)

# -------------------------------------------------------------
# TAB 5: EXECUTIVE BRIEF
# -------------------------------------------------------------
with tab_executive:
    st.subheader("Automated Executive Briefing & Decision Matrix")
    
    insights_engine = InsightsEngine(
        summary_df=summary_df,
        nps_data=nps_results,
        chi_square_results=chi_results,
        sentiment_data=sentiments,
        option_col="selected_option"
    )
    brief = insights_engine.generate_executive_insights()

    st.markdown(f"### 🏆 Primary Recommendation: **{brief['top_choice']}**")
    st.write(f"- **Consensus Share:** {brief['top_choice_share']:.1f}%")
    st.write(f"- **Margin over Runner-up:** {brief['lead_margin']}%")
    st.info(f"**Lead Reliability:** {brief['lead_verdict']}")

    st.markdown("### 📋 Prescriptive Strategic Action Plan:")
    for idx, rec in enumerate(brief["strategic_recommendations"], 1):
        st.markdown(f"**{idx}.** {rec}")

    st.markdown("---")
    # Export full report button
    report_text = f"""POLL RESULTS VISUALIZER - EXECUTIVE SUMMARY
Consensus Choice: {brief['top_choice']} ({brief['top_choice_share']:.1f}%)
Victory Margin: {brief['lead_margin']}%
NPS Score: {brief['nps_score']}
Demographic Finding: {brief['demographic_finding']}
Recommendations:
""" + "\n".join([f"- {r}" for r in brief['strategic_recommendations']])

    st.download_button(
        label="📥 Download Executive Briefing (TXT)",
        data=report_text,
        file_name="executive_poll_summary.txt",
        mime="text/plain"
    )
