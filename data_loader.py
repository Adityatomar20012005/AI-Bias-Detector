import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import streamlit as st


@st.cache_data
def load_sample_data():

    try:
        # Fetch from UCI ML repository
        adult = fetch_ucirepo(id=2)
        
        X = adult.data
        y = adult.targets
        
        # Combine features and target
        df = X.copy()
        df['income'] = y
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure income is binary
        df['income'] = df['income'].astype(str).str.strip()
        
        return df
    
    except Exception as e:
        st.error(f"Error loading UCI Adult dataset: {str(e)}")
        st.info("Creating synthetic dataset as fallback...")
        return create_synthetic_dataset()


def create_synthetic_dataset():
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'education_num': np.random.randint(1, 16, n_samples),
        'hours_per_week': np.random.randint(1, 100, n_samples),
        'capital_gain': np.random.exponential(10000, n_samples),
        'capital_loss': np.random.exponential(1000, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create biased income target
    prob_income = (0.3 + 
                   df['age'] / 100 + 
                   df['education_num'] / 20 +
                   (df['sex'] == 'Male').astype(int) * 0.2 +  # Gender bias
                   (df['race'] == 'White').astype(int) * 0.15)  # Race bias
    
    prob_income = 1 / (1 + np.exp(-prob_income))
    df['income'] = (np.random.rand(n_samples) < prob_income).astype(int)
    df['income'] = df['income'].map({0: '<=50K', 1: '>50K'})
    
    return df


def get_columns_names(df):
    return list(df.columns)


def validate_data(df, target_column, sensitive_columns):
    errors = []
    
    # Check target column exists
    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' not found")
    
    # Check sensitive columns exist
    for col in sensitive_columns:
        if col not in df.columns:
            errors.append(f"Sensitive column '{col}' not found")
    
    # Check minimum samples
    if len(df) < 50:
        errors.append(f"Dataset too small ({len(df)} rows). Need at least 50.")
    
    # Check for sufficient class distribution
    if target_column in df.columns:
        value_counts = df[target_column].value_counts()
        if len(value_counts) < 2:
            errors.append("Target column must have at least 2 distinct values")
        for val, count in value_counts.items():
            if count < 5:
                errors.append(f"Target class '{val}' has only {count} samples. Need at least 5.")
    
    # Check sensitive columns have variation
    for col in sensitive_columns:
        if col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals < 2:
                errors.append(f"Sensitive column '{col}' has only 1 unique value")
    
    if errors:
        return False, " | ".join(errors)
    return True, "OK"
