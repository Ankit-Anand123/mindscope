#!/usr/bin/env python3
"""
MindScope: Data Exploration Script
=====================================
Initial exploration of the social media mental health dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main data exploration function"""
    
    print("MindScope Data Exploration")
    print("=" * 50)
    
    # Load the data
    data_path = "data/survey_data/raw/smmh.csv"
    
    if not Path(data_path).exists():
        print(f"Data file not found at: {data_path}")
        print("Please make sure smmh.csv is in the data/survey_data/raw/ folder")
        return
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic dataset information
    print(f"\nDATASET OVERVIEW")
    print("-" * 30)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\nCOLUMNS ({len(df.columns)} total)")
    print("-" * 30)
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].count()
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        print(f"{i:2d}. {col:<25} | {str(dtype):<10} | {non_null:>6,} non-null ({null_pct:4.1f}% missing)")
    
    # First few rows
    print(f"\nFIRST 3 ROWS")
    print("-" * 30)
    print(df.head(3).to_string())
    
    # Data types analysis
    print(f"\nDATA TYPES SUMMARY")
    print("-" * 30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Text/Object columns ({len(text_cols)}): {text_cols}")
    
    # Look for potential target variables (binary or categorical with few unique values)
    print(f"\nPOTENTIAL TARGET VARIABLES")
    print("-" * 30)
    potential_targets = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count <= 10 and unique_count >= 2:  # Good range for classification
            potential_targets.append(col)
            unique_vals = sorted(df[col].unique())
            print(f"• {col}: {unique_count} unique values → {unique_vals}")
    
    # Look for text columns (likely social media content)
    print(f"\nPOTENTIAL TEXT CONTENT")
    print("-" * 30)
    text_content_cols = []
    
    for col in text_cols:
        # Check if this column contains substantial text
        sample_texts = df[col].dropna().head(3)
        avg_length = df[col].astype(str).str.len().mean()
        
        if avg_length > 20:  # Likely contains substantial text
            text_content_cols.append(col)
            print(f"• {col}: Avg length = {avg_length:.1f} characters")
            print(f"  Sample: '{sample_texts.iloc[0][:100]}...'")
            print()
    
    # Missing data analysis
    print(f"\nMISSING DATA ANALYSIS")
    print("-" * 30)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        for col, count in missing_data.items():
            pct = (count / len(df)) * 100
            print(f"• {col}: {count:,} missing ({pct:.1f}%)")
    else:
        print("No missing data found!")
    
    # Basic statistics for numeric columns
    if numeric_cols:
        print(f"\nNUMERIC COLUMNS STATISTICS")
        print("-" * 30)
        print(df[numeric_cols].describe())
    
    # Save exploration summary
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': numeric_cols,
        'text_columns': text_cols,
        'potential_targets': potential_targets,
        'text_content_columns': text_content_cols,
        'missing_data': missing_data.to_dict() if len(missing_data) > 0 else {}
    }
    
    print(f"\nEXPLORATION SUMMARY")
    print("-" * 30)
    print(f"Dataset ready for analysis: {len(df):,} samples")
    print(f"Potential target columns: {potential_targets}")
    print(f"Potential text columns: {text_content_cols}")
    
    return df, summary

if __name__ == "__main__":
    df, summary = main()
    print(f"\nData exploration complete!")
    print(f"Next step: Review the output above and identify your text and label columns")