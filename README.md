# 📊 Poll Results Visualizer & Survey Analytics Suite

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visuals-3F4F75.svg)](https://plotly.com/)
[![SciPy](https://img.shields.io/badge/SciPy-Hypothesis%20Testing-8CAAE6.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An enterprise-grade, interactive survey analytics platform and decision-support engine.** Translates raw poll data (Google Forms, CSVs, Excel) into statistically rigorous vote distributions, demographic cross-tabulations, Chi-Square independence tests, Net Promoter Score (NPS) diagnostics, and NLP qualitative sentiment insights.

---

## 📌 Executive Summary & Industry Context

Organizations worldwide—including **Google (Forms & Consumer Surveys)**, **Qualtrics**, **SurveyMonkey**, **YouGov**, **Pew Research**, and **McKinsey**—rely on survey telemetry to make high-stakes product, workforce, and strategic decisions. 

However, raw survey exports suffer from common pitfalls:
1. **Misleading Percentages:** Reporting raw counts without a **Margin of Error (MOE)** at statistical confidence intervals.
2. **Hidden Demographic Skew:** Failing to detect whether age or region heavily biases opinion via hypothesis testing.
3. **Ignored Qualitative Voice:** Discarding open-ended text comments due to lack of NLP pipelines.
4. **Static, Non-Interactive Reports:** Inability for non-technical executives to filter responses dynamically by cohort.

**Poll Results Visualizer** solves these challenges by combining a production-grade data cleansing pipeline, inferential statistical mechanics, automated natural language sentiment extraction, and an interactive Streamlit decision dashboard.

---

## 🚀 Key Features

- **Automated Data Sanitization Pipeline:** Deduplicates respondents, repairs inconsistent casing, parses timestamps, imputes optional comments, and audits data retention.
- **Statistical Rigor & Margin of Error (MOE):** Computes 95% Confidence Intervals for every poll option using standard inferential survey sampling theory:
  $$\text{MOE} = 1.96 \times \sqrt{\frac{p \times (1 - p)}{N}}$$
- **Demographic Cross-Tabulations & Heatmaps:** Interactively pivots responses across Region, Age Bracket, Gender, and Education Level.
- **Hypothesis Testing ($\chi^2$ Test of Independence):** Calculates Pearson's Chi-Square statistic ($p$-value and degrees of freedom) to scientifically prove whether demographic characteristics drive voter preference.
- **Customer Loyalty Diagnostics (Net Promoter Score - NPS):** Segments respondents into Promoters (9-10), Passives (7-8), and Detractors (0-6) with benchmark evaluations.
- **Voice-of-Customer NLP Mining:** Tokenizes feedback, computes sentiment polarity density (Positive, Neutral, Negative), extracts dominant n-gram keywords, and renders word clouds.
- **Interactive Multi-Survey Streamlit Dashboard:** Load pre-configured industry scenarios (Workplace Culture, Consumer Tech, Public Policy) or drag-and-drop your custom Google Forms CSV.
- **Automated Executive Briefing Generator:** Rule-based decision engine exports plain-language business recommendations and lead-verdict briefings.

---

## 🏗️ Project Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             1. DATA INGESTION                               │
│  - Google Forms Export (CSV/XLSX)                                           │
│  - Public Survey Repositories                                               │
│  - High-Fidelity Synthetic Generator (Workplace, Consumer, Policy)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. AUDITING & CLEANSING                            │
│  - Fingerprint Deduplication (Respondent ID / Multi-Field Match)             │
│  - Case Normalization & String Sanitization (Title Casing)                  │
│  - Temporal Parsing (Date, Week, Submission Velocity)                       │
│  - NPS Categorization (Promoters vs. Detractors)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     3. ADVANCED ANALYTICAL ENGINE                           │
│  ┌───────────────────────┬───────────────────────┬───────────────────────┐  │
│  │ Descriptive Share (%) │ 95% CI Margin of Error│ Cross-Tabulation      │  │
│  ├───────────────────────┼───────────────────────┼───────────────────────┤  │
│  │ Chi-Square Test (χ²)  │ Net Promoter Score    │ Sentiment Mining      │  │
│  └───────────────────────┴───────────────────────┴───────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│   4. PRESENTATION & VISUALIZATION   │ │      5. DECISION BRIEF ENGINE       │
│  - Matplotlib / Seaborn 300-DPI PNGs│ │  - Statistical Lead Assessment      │
│  - Interactive Plotly HTML Visuals  │ │  - Decisive vs Dead-Heat Detection  │
│  - Dynamic Streamlit App (app.py)   │ │  - Executive Report (.txt / .pdf)   │
└─────────────────────────────────────┘ └─────────────────────────────────────┘
```

---

## 📂 Folder Structure

```
Poll-Results-Visualizer/
│
├── data/
│   ├── raw/                           # Raw poll CSVs (Google Forms exports / synthetic)
│   │   ├── workplace_pulse_raw.csv
│   │   ├── consumer_tech_survey_raw.csv
│   │   └── public_policy_poll_raw.csv
│   └── processed/                     # Sanitized, deduplicated, and feature-engineered datasets
│       ├── cleaned_poll_data.csv
│       ├── option_summary.csv
│       └── regional_cross_tab.csv
│
├── notebooks/                         # Exploratory data analysis & prototyping
│   └── 01_poll_exploratory_data_analysis.py
│
├── src/                               # Modular, production-grade source code
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic survey generator with demographic skew
│   ├── data_processor.py              # Cleansing, deduplication, timestamp formatting
│   ├── poll_analyzer.py               # Statistics, MOE, Chi-Square, NPS, Cross-tabs
│   ├── text_analyzer.py               # NLP tokenization, stopwords, sentiment mining
│   ├── visualizer.py                  # Static (Matplotlib/Seaborn) & Interactive (Plotly)
│   └── insights_engine.py             # Automated strategic executive briefing generator
│
├── outputs/
│   ├── charts/                        # High-resolution (300 DPI) publication charts
│   │   ├── 01_vote_distribution_bar.png
│   │   ├── 02_vote_share_donut.png
│   │   ├── 03_demographic_stacked_bar.png
│   │   ├── 04_response_timeline_trend.png
│   │   └── 05_feedback_wordcloud.png
│   └── reports/                       # Auto-generated strategic executive briefings
│       └── Executive_Poll_Summary.txt
│
├── images/                            # Portfolio screenshots and documentation assets
├── requirements.txt                   # Pinned project dependencies
├── main.py                            # Master CLI pipeline orchestrator
├── app.py                             # Full-featured Streamlit interactive web dashboard
└── README.md                          # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/Poll-Results-Visualizer.git
cd Poll-Results-Visualizer
```

### 3. Set Up Virtual Environment

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run the Project

### Option A: Run the End-to-End CLI Pipeline
Executes data generation, cleaning, statistical analysis, high-res chart generation, and executive brief export:
```bash
python main.py
```

### Option B: Launch the Interactive Streamlit Web Application
Launches the browser-based dashboard with dynamic cohort filtering and custom file upload:
```bash
streamlit run app.py
```
*App will automatically open in your default browser at `http://localhost:8501`.*

---

## 📈 Visual Gallery & Output Artifacts

| Visualization | Description | Industry Value |
| :--- | :--- | :--- |
| **01. Vote Distribution with MOE** | Horizontal bar chart with 95% Confidence Interval error bars | Prevents false conclusions by highlighting whether a lead is statistically significant. |
| **02. Vote Share Donut Chart** | Proportional share of total responses | Ideal for executive slide decks and high-level stakeholder summaries. |
| **03. Demographic Stacked Bar** | 100% normalized preference across regions/ages | Reveals geographic or generational polarization across options. |
| **04. Longitudinal Velocity Trend** | Daily response velocity with shaded volume | Tracks survey response momentum, fatigue, and campaign efficacy. |
| **05. Voice of Customer Word Cloud** | Thematic keyword density from open-ended text | Uncovers unprompted customer sentiment, complaints, and praise. |

*(Sample charts are generated directly into the `outputs/charts/` and `images/` directories).*

---

## 🔬 Mathematical Formulations

### 1. Margin of Error at 95% Confidence Interval
$$\text{MOE} = Z \times \sqrt{\frac{\hat{p}(1 - \hat{p})}{N}}$$
Where:
- $Z = 1.96$ for a 95% confidence level
- $\hat{p} = \text{Sample proportion choosing option } k$
- $N = \text{Total valid sample size}$

### 2. Pearson's Chi-Square Test of Independence
$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
Where:
- $O_{ij} = \text{Observed count for demographic row } i \text{ and option column } j$
- $E_{ij} = \frac{(\text{Row Total}_i \times \text{Column Total}_j)}{N} = \text{Expected count under null hypothesis of independence}$

### 3. Net Promoter Score (NPS)
$$\text{NPS} = \% \text{Promoters (Score 9-10)} - \% \text{Detractors (Score 0-6)}$$

---

## 💼 Placement & Interview Readiness

### Sample Interview Talking Points:
1. **Business Impact:** *"Rather than simply plotting raw counts, I engineered a pipeline that calculates 95% Confidence Intervals to tell business leaders whether an option's lead is statistically decisive or a dead heat."*
2. **Data Cleansing:** *"I designed a modular pipeline that handles duplicate survey submissions, normalizes casing variations, and standardizes Likert ratings into actionable NPS segments."*
3. **Statistical Literacy:** *"I incorporated a Chi-Square test of independence to validate whether demographic segments (e.g., Gen-Z vs. Boomers) exhibited statistically significant differences in preference."*
4. **Interactive BI:** *"I built an interactive Streamlit application enabling cross-functional stakeholders to slice and dice survey data in real time without writing SQL or Python."*

---

## 📜 License
This project is licensed under the MIT License - feel free to use it for educational and portfolio demonstration purposes.
