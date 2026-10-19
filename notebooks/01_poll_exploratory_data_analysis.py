"""
Jupyter Notebook Companion Script: 01_poll_exploratory_data_analysis.py
Run this interactively or convert to .ipynb to inspect step-by-step EDA,
statistical assumptions, and visualization prototypes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_synthetic_poll
from src.data_processor import PollDataProcessor
from src.poll_analyzer import PollAnalyzer
from src.visualizer import PollVisualizer

def main():
    print("Executing Step-by-Step Exploratory Data Analysis...")
    
    # 1. Generate Raw Data
    df_raw = generate_synthetic_poll(num_responses=1000, survey_type="workplace", random_seed=42)
    print(f"Raw Data Shape: {df_raw.shape}")
    print(df_raw.head())

    # 2. Process & Clean
    processor = PollDataProcessor()
    processor.load_data(df_raw)
    df_clean, audit = processor.clean_pipeline()
    print("Cleaning Audit Summary:", audit)

    # 3. Analyze
    analyzer = PollAnalyzer(df_clean)
    summary = analyzer.calculate_option_summary()
    print("\nOption Summary:\n", summary)

    # 4. Regional Cross-Tabulation
    ct = analyzer.calculate_cross_tabulation("region", "selected_option")
    print("\nRegional Distribution (%):\n", ct)

    # 5. Chi-Square Test
    chi_test = analyzer.test_chi_square_independence("age_group", "selected_option")
    print("\nChi-Square Test (Age vs Choice):\n", chi_test)

    print("\nEDA Completed successfully.")

if __name__ == "__main__":
    main()
