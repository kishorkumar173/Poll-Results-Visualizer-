"""
Module: data_generator.py
Description: Generates high-fidelity, realistic synthetic poll and survey datasets
simulating real-world Google Forms exports, market research surveys, and public opinion polls.
Includes intentional real-world artifacts (duplicates, casing variations, missing comments)
to test and validate data preprocessing pipelines.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_poll(
    num_responses: int = 1200,
    survey_type: str = "workplace",
    random_seed: int = 42,
    inject_noise: bool = True
) -> pd.DataFrame:
    """
    Generates a synthetic poll dataset with realistic demographic correlations,
    categorical options, Likert ratings, NPS scores, and open-ended text comments.

    Parameters:
    -----------
    num_responses : int
        Total number of survey responses to generate.
    survey_type : str
        Type of survey: 'workplace', 'consumer_tech', or 'public_policy'.
    random_seed : int
        Seed for reproducibility.
    inject_noise : bool
        If True, injects realistic duplicates, nulls, and formatting artifacts.

    Returns:
    --------
    pd.DataFrame: Synthetic survey dataframe.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Base Demographics
    regions = ["North", "South", "East", "West", "Central"]
    region_weights = [0.28, 0.22, 0.18, 0.22, 0.10]

    age_groups = ["18-25", "26-35", "36-45", "46-55", "56+"]
    age_weights = [0.25, 0.35, 0.22, 0.12, 0.06]

    genders = ["Female", "Male", "Non-Binary", "Prefer not to say"]
    gender_weights = [0.48, 0.46, 0.04, 0.02]

    education_levels = ["High School", "Bachelor's", "Master's", "Doctorate"]
    edu_weights = [0.15, 0.52, 0.27, 0.06]

    start_date = datetime(2026, 8, 1)
    timestamps = [
        start_date + timedelta(days=int(np.random.exponential(scale=18)), hours=random.randint(8, 22), minutes=random.randint(0, 59))
        for _ in range(num_responses)
    ]

    # Domain specific configurations
    if survey_type == "consumer_tech":
        main_question = "What is your primary factor when purchasing a new smartphone or tech device?"
        options = [
            "Battery Life & Efficiency",
            "Camera & Content Creation",
            "Performance & Gaming",
            "Brand Value & Ecosystem",
            "Pricing & Discounts"
        ]
        # Option distribution with slight age skew
        option_weights = [0.32, 0.24, 0.18, 0.16, 0.10]
        
        pos_feedbacks = [
            "Great battery longevity and super clean UI.",
            "Exceptional camera quality for the price.",
            "Smooth performance and snappy multitasking.",
            "Love the ecosystem seamless connectivity with laptop.",
            "Value for money proposition is top tier."
        ]
        neu_feedbacks = [
            "Good product overall, but charging speed could improve.",
            "Decent specs, though slightly overpriced.",
            "Standard features, nothing revolutionary but reliable.",
            "Average experience, meets daily basic expectations."
        ]
        neg_feedbacks = [
            "Battery drains way too fast under moderate load.",
            "Customer service was unresponsive and delayed.",
            "Software bloatware degrades overall responsiveness.",
            "Build quality feels plasticky and fragile."
        ]

    elif survey_type == "public_policy":
        main_question = "Which municipal infrastructure initiative should receive priority funding?"
        options = [
            "Renewable Energy & Solar Grids",
            "Public Mass Rapid Transit",
            "Healthcare & Community Clinics",
            "Affordable Housing Developments",
            "Road Repair & Highway Expansion"
        ]
        option_weights = [0.29, 0.25, 0.22, 0.14, 0.10]
        pos_feedbacks = [
            "Clean energy transition is crucial for long-term health.",
            "Metro connectivity significantly cut down daily commute.",
            "Strengthening public healthcare centers is a must.",
            "Investment in green transit promotes sustainable cities."
        ]
        neu_feedbacks = [
            "Funding is needed, but execution timeline must be transparent.",
            "Both transit and healthcare require balanced allocations.",
            "Adequate proposal, dependent on local municipal governance."
        ]
        neg_feedbacks = [
            "Highway projects get delayed with budget overruns.",
            "Lack of proper maintenance on existing infrastructure.",
            "Taxes are high yet local road quality remains unacceptable."
        ]

    else:  # workplace (default)
        main_question = "What is your preferred workplace model for sustainable productivity?"
        options = [
            "Flexible Hybrid (2-3 Days Office)",
            "Fully Remote (Work from Anywhere)",
            "Office First (4+ Days in Office)",
            "Activity-Based / Dynamic Co-Working",
            "Strictly On-Site Work"
        ]
        option_weights = [0.42, 0.31, 0.14, 0.09, 0.04]
        pos_feedbacks = [
            "Hybrid model provides ideal work-life equilibrium.",
            "Remote work saved 2 hours of exhausting daily commute.",
            "In-person collaboration on key days boosts creative synergy.",
            "High flexibility has improved my mental well-being and output.",
            "Clear team sync routines keep productivity at an all-time high."
        ]
        neu_feedbacks = [
            "Hybrid is acceptable if meeting calendars are structured.",
            "Office equipment is adequate but commute remains troublesome.",
            "Policy is fine, but cross-team alignment needs improvement.",
            "Middle managers need better remote leadership training."
        ]
        neg_feedbacks = [
            "Mandating 4 days in office damages team morale and retention.",
            "Open office floor plan causes relentless audio distraction.",
            "Lack of ergonomic home office allowance is disappointing.",
            "Micromanagement has noticeably spiked with return-to-office."
        ]

    # Generate records
    data = []
    for i in range(num_responses):
        resp_id = f"RESP_{i+1:05d}"
        t_stamp = timestamps[i]
        reg = np.random.choice(regions, p=region_weights)
        age = np.random.choice(age_groups, p=age_weights)
        gen = np.random.choice(genders, p=gender_weights)
        edu = np.random.choice(education_levels, p=edu_weights)

        # Realistic conditional probability: younger demographic slightly favors remote/hybrid or tech
        adj_weights = list(option_weights)
        if age in ["18-25", "26-35"]:
            adj_weights[0] += 0.05
            adj_weights[1] += 0.05
            adj_weights[-1] = max(0.01, adj_weights[-1] - 0.05)
            adj_weights[-2] = max(0.01, adj_weights[-2] - 0.05)
            adj_weights = [w / sum(adj_weights) for w in adj_weights]

        opt = np.random.choice(options, p=adj_weights)

        # Satisfaction Likert (1 to 5)
        if opt in [options[0], options[1]]:
            likert = int(np.clip(np.random.normal(loc=4.2, scale=0.8), 1, 5))
            nps = int(np.clip(np.random.normal(loc=8.4, scale=1.4), 0, 10))
            fb = np.random.choice(pos_feedbacks) if random.random() < 0.75 else np.random.choice(neu_feedbacks)
        else:
            likert = int(np.clip(np.random.normal(loc=3.1, scale=1.1), 1, 5))
            nps = int(np.clip(np.random.normal(loc=6.2, scale=2.1), 0, 10))
            fb = np.random.choice(neg_feedbacks) if random.random() < 0.65 else np.random.choice(neu_feedbacks)

        # Likert label text mapping
        likert_map = {
            1: "Strongly Disagree",
            2: "Disagree",
            3: "Neutral",
            4: "Agree",
            5: "Strongly Agree"
        }

        data.append({
            "respondent_id": resp_id,
            "timestamp": t_stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "region": reg,
            "age_group": age,
            "gender": gen,
            "education": edu,
            "survey_topic": survey_type,
            "poll_question": main_question,
            "selected_option": opt,
            "satisfaction_rating": likert,
            "satisfaction_label": likert_map[likert],
            "nps_score": nps,
            "feedback_comment": fb
        })

    df = pd.DataFrame(data)

    if inject_noise:
        # 1. Inject duplicate responses (simulating duplicate submissions)
        dup_rows = df.sample(n=int(num_responses * 0.03), random_state=random_seed)
        df = pd.concat([df, dup_rows], ignore_index=True)

        # 2. Inject missing values in open-ended comments (simulating optional fields)
        mask_missing = np.random.rand(len(df)) < 0.12
        df.loc[mask_missing, "feedback_comment"] = np.nan

        # 3. Inject casing discrepancies in Region
        casing_idx = df.sample(frac=0.04, random_state=random_seed).index
        df.loc[casing_idx, "region"] = df.loc[casing_idx, "region"].str.lower()

        # Shuffle
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df

def generate_and_save_all_datasets(output_dir: str):
    """
    Generates multiple realistic datasets (Workplace, Consumer Tech, Public Policy)
    and saves them as raw CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        "workplace_pulse_raw.csv": ("workplace", 1400),
        "consumer_tech_survey_raw.csv": ("consumer_tech", 1200),
        "public_policy_poll_raw.csv": ("public_policy", 1000)
    }

    generated_files = []
    for filename, (stype, n_rows) in datasets.items():
        filepath = os.path.join(output_dir, filename)
        df = generate_synthetic_poll(num_responses=n_rows, survey_type=stype, random_seed=42)
        df.to_csv(filepath, index=False)
        generated_files.append((filepath, len(df)))
        print(f"Generated: {filepath} with {len(df)} rows.")

    return generated_files

if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    generate_and_save_all_datasets(raw_dir)
