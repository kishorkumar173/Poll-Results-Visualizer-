"""
Module: data_processor.py
Description: Robust preprocessing and cleaning pipeline for raw survey/poll datasets.
Cleans raw Google Form responses, handles missing values, removes duplicates, standardizes
demographic variables, and adds derived metrics (NPS segments, date parts).
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class PollDataProcessor:
    """
    Automated data cleansing and feature engineering pipeline for survey analytics.
    """

    def __init__(self, raw_filepath: str = None):
        self.raw_filepath = raw_filepath
        self.raw_df = None
        self.cleaned_df = None
        self.cleaning_audit = {}

    def load_data(self, source) -> pd.DataFrame:
        """
        Loads data from file path or existing DataFrame.
        Supports CSV and Excel files.
        """
        if isinstance(source, pd.DataFrame):
            self.raw_df = source.copy()
        elif isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Source file not found: {source}")
            if source.endswith(('.xls', '.xlsx')):
                self.raw_df = pd.read_excel(source)
            else:
                self.raw_df = pd.read_csv(source)
        else:
            raise ValueError("Input source must be a valid file path or pandas DataFrame.")

        return self.raw_df

    def audit_raw_data(self) -> Dict[str, Any]:
        """
        Generates comprehensive profiling of raw data prior to cleansing.
        """
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_rows = len(self.raw_df)
        initial_cols = len(self.raw_df.columns)
        duplicate_count = int(self.raw_df.duplicated().sum())
        null_counts = self.raw_df.isnull().sum().to_dict()

        self.cleaning_audit = {
            "initial_rows": initial_rows,
            "initial_columns": initial_cols,
            "duplicate_records": duplicate_count,
            "missing_values_per_col": null_counts,
            "columns": list(self.raw_df.columns)
        }
        return self.cleaning_audit

    def clean_pipeline(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes end-to-end data cleansing:
        1. Deduplication
        2. Column name standardization
        3. Whitespace trimming & casing alignment
        4. Timestamp parsing and date extraction
        5. NPS category calculation (Promoter, Passive, Detractor)
        6. Missing value handling
        """
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Ensure raw data is audited before cleaning
        if not self.cleaning_audit or "initial_rows" not in self.cleaning_audit:
            self.audit_raw_data()

        df = self.raw_df.copy()

        # Step 1: Deduplication
        before_dup = len(df)
        if "respondent_id" in df.columns:
            df = df.drop_duplicates(subset=["respondent_id"], keep="first")
        else:
            df = df.drop_duplicates(keep="first")
        removed_duplicates = before_dup - len(df)

        # Step 2: Standardize column headers (lowercase, stripped, underscores)
        df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

        # Step 3: String trimming and casing for categorical demographics
        cat_cols = ["region", "gender", "age_group", "education", "selected_option"]
        for col in cat_cols:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip().str.title()
                # Fix edge cases like "Non-Binary" title-cased properly
                df[col] = df[col].replace({"Non-Binary": "Non-Binary", "Nan": "Unknown"})

        # Step 4: Parse Timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # Extract date components for longitudinal analysis
            df["response_date"] = df["timestamp"].dt.date
            df["response_week"] = df["timestamp"].dt.to_period("W").astype(str)
            df["response_hour"] = df["timestamp"].dt.hour
        else:
            df["timestamp"] = pd.Timestamp.now()
            df["response_date"] = df["timestamp"].dt.date
            df["response_week"] = "Current Week"
            df["response_hour"] = 12

        # Step 5: Clean Numerical Ratings & NPS
        if "satisfaction_rating" in df.columns:
            df["satisfaction_rating"] = pd.to_numeric(df["satisfaction_rating"], errors="coerce").fillna(3).astype(int)
            df["satisfaction_rating"] = df["satisfaction_rating"].clip(1, 5)

        if "nps_score" in df.columns:
            df["nps_score"] = pd.to_numeric(df["nps_score"], errors="coerce").fillna(7).astype(int)
            df["nps_score"] = df["nps_score"].clip(0, 10)
            
            # Feature engineering: NPS category
            # Promoters: 9-10, Passives: 7-8, Detractors: 0-6
            conditions = [
                (df["nps_score"] >= 9),
                (df["nps_score"] >= 7) & (df["nps_score"] <= 8),
                (df["nps_score"] <= 6)
            ]
            choices = ["Promoter", "Passive", "Detractor"]
            df["nps_category"] = np.select(conditions, choices, default="Passive")

        # Step 6: Text imputation for open-ended comments
        comment_col = None
        for cand in ["feedback_comment", "feedback", "comments", "open_ended_feedback"]:
            if cand in df.columns:
                comment_col = cand
                break

        if comment_col:
            df[comment_col] = df[comment_col].fillna("No specific comment provided").astype(str).str.strip()

        # Step 7: Final reset
        df = df.reset_index(drop=True)
        self.cleaned_df = df

        initial_r = self.cleaning_audit.get("initial_rows", len(self.raw_df) if self.raw_df is not None else len(df))
        self.cleaning_audit["final_rows"] = len(df)
        self.cleaning_audit["removed_duplicates"] = removed_duplicates
        self.cleaning_audit["retention_rate_pct"] = round((len(df) / initial_r) * 100, 2) if initial_r > 0 else 100.0

        return self.cleaned_df, self.cleaning_audit

    def save_cleaned_data(self, output_filepath: str):
        """
        Saves the processed DataFrame to disk.
        """
        if self.cleaned_df is None:
            raise ValueError("Cleaned DataFrame is not available. Run clean_pipeline() first.")
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        self.cleaned_df.to_csv(output_filepath, index=False)
        print(f"Cleaned dataset saved successfully to: {output_filepath}")
