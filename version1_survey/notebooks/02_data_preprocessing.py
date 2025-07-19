#!/usr/bin/env python3
"""
MindScope: Data Preprocessing for Survey-Based Mental Health Classification
===========================================================================
Preprocessing survey responses for mental health risk assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SurveyPreprocessor:
    """Preprocess survey data for mental health classification"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load the survey dataset"""
        print(f"📂 Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df):,} survey responses")
        return df
    
    def create_target_variable(self, df):
        """Create binary mental health risk target from depression scores"""
        depression_col = '18. How often do you feel depressed or down?'
        
        print(f"\n🎯 Creating target variable from: {depression_col}")
        print(f"Original distribution:")
        print(df[depression_col].value_counts().sort_index())
        
        # Binary classification: 1-3 = Low Risk, 4-5 = High Risk
        df['mental_health_risk'] = (df[depression_col] >= 4).astype(int)
        
        print(f"\n📊 Target variable distribution:")
        risk_counts = df['mental_health_risk'].value_counts()
        print(f"Low Risk (0): {risk_counts[0]} ({risk_counts[0]/len(df)*100:.1f}%)")
        print(f"High Risk (1): {risk_counts[1]} ({risk_counts[1]/len(df)*100:.1f}%)")
        
        return df
    
    def create_feature_set(self, df):
        """Create feature set from survey responses"""
        print(f"\n🔧 Engineering features from survey responses...")
        
        # Define feature categories
        social_media_features = [
            '9. How often do you find yourself using Social media without a specific purpose?',
            '10. How often do you get distracted by Social media when you are busy doing something?',
            '11. Do you feel restless if you haven\'t used Social media in a while?',
            '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?',
            '17. How often do you look to seek validation from features of social media?'
        ]
        
        psychological_features = [
            '12. On a scale of 1 to 5, how easily distracted are you?',
            '13. On a scale of 1 to 5, how much are you bothered by worries?',
            '14. Do you find it difficult to concentrate on things?',
            '16. Following the previous question, how do you feel about these comparisons, generally speaking?',
            '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?',
            '20. On a scale of 1 to 5, how often do you face issues regarding sleep?'
        ]
        
        # Demographic features
        demographic_features = ['1. What is your age?']
        
        # Create feature dataframe
        features_df = df[social_media_features + psychological_features + demographic_features].copy()
        
        # Rename columns for clarity
        feature_mapping = {
            '9. How often do you find yourself using Social media without a specific purpose?': 'aimless_social_media_use',
            '10. How often do you get distracted by Social media when you are busy doing something?': 'social_media_distraction',
            '11. Do you feel restless if you haven\'t used Social media in a while?': 'social_media_restlessness',
            '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': 'social_comparison',
            '17. How often do you look to seek validation from features of social media?': 'validation_seeking',
            '12. On a scale of 1 to 5, how easily distracted are you?': 'distractibility',
            '13. On a scale of 1 to 5, how much are you bothered by worries?': 'worry_level',
            '14. Do you find it difficult to concentrate on things?': 'concentration_difficulty',
            '16. Following the previous question, how do you feel about these comparisons, generally speaking?': 'comparison_feelings',
            '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': 'interest_fluctuation',
            '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'sleep_issues',
            '1. What is your age?': 'age'
        }
        
        features_df = features_df.rename(columns=feature_mapping)
        self.feature_names = list(features_df.columns)
        
        # Encode categorical variables (if any)
        categorical_cols = ['2. Gender', '3. Relationship Status', '8. What is the average time you spend on social media every day?']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create simplified encodings
                if col == '2. Gender':
                    # Simplify gender categories
                    df[col] = df[col].str.lower().str.strip()
                    gender_mapping = {
                        'male': 0, 'female': 1, 'nb': 2, 'non binary': 2, 
                        'non-binary': 2, 'nonbinary': 2, 'trans': 2, 
                        'there are others???': 2, 'unsure': 2
                    }
                    features_df['gender'] = df[col].map(gender_mapping).fillna(2)
                    
                elif col == '3. Relationship Status':
                    relationship_mapping = {
                        'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3
                    }
                    features_df['relationship_status'] = df[col].map(relationship_mapping)
                    
                elif col == '8. What is the average time you spend on social media every day?':
                    # Convert time to numeric scale
                    time_mapping = {
                        'Less than an Hour': 1,
                        'Between 1 and 2 hours': 2,
                        'Between 2 and 3 hours': 3,
                        'Between 3 and 4 hours': 4,
                        'Between 4 and 5 hours': 5,
                        'More than 5 hours': 6
                    }
                    features_df['daily_social_media_hours'] = df[col].map(time_mapping)
        
        self.feature_names = list(features_df.columns)
        print(f"✅ Created {len(self.feature_names)} features: {self.feature_names}")
        
        return features_df
    
    def create_composite_scores(self, features_df):
        """Create composite psychological scores"""
        print(f"\n🧮 Creating composite psychological scores...")
        
        # Social Media Addiction Score
        sm_addiction_cols = ['aimless_social_media_use', 'social_media_distraction', 'social_media_restlessness']
        features_df['social_media_addiction_score'] = features_df[sm_addiction_cols].mean(axis=1)
        
        # Social Comparison Score
        comparison_cols = ['social_comparison', 'validation_seeking']
        if 'comparison_feelings' in features_df.columns:
            comparison_cols.append('comparison_feelings')
        features_df['social_comparison_score'] = features_df[comparison_cols].mean(axis=1)
        
        # General Mental Health Score
        mental_health_cols = ['distractibility', 'worry_level', 'concentration_difficulty', 'interest_fluctuation', 'sleep_issues']
        features_df['general_mental_health_score'] = features_df[mental_health_cols].mean(axis=1)
        
        self.feature_names = list(features_df.columns)
        print(f"✅ Added composite scores. Total features: {len(self.feature_names)}")
        
        return features_df
    
    def split_and_scale_data(self, features_df, target_df, test_size=0.2, random_state=42):
        """Split data and apply scaling"""
        print(f"\n📊 Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target_df, test_size=test_size, 
            random_state=random_state, stratify=target_df
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        print(f"🔧 Scaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        print(f"\n💾 Saving processed data...")
        
        # Create processed directory
        Path("data/survey_data/processed").mkdir(parents=True, exist_ok=True)
        
        # Combine features and targets
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        test_data = X_test.copy()
        test_data['target'] = y_test
        
        # Save
        train_data.to_csv("data/survey_data/processed/train.csv", index=False)
        test_data.to_csv("data/survey_data/processed/test.csv", index=False)
        
        # Save feature names
        pd.Series(self.feature_names).to_csv("data/survey_data/processed/feature_names.csv", index=False, header=['feature'])
        
        print(f"✅ Saved processed data to data/survey_data/processed/")
        print(f"   - train.csv: {len(train_data)} samples")
        print(f"   - test.csv: {len(test_data)} samples")
        print(f"   - feature_names.csv: {len(self.feature_names)} features")

def main():
    """Main preprocessing pipeline"""
    print("🔧 MindScope: Survey Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = SurveyPreprocessor()
    
    # Load data
    df = preprocessor.load_data("data/survey_data/raw/smmh.csv")
    
    # Create target variable
    df = preprocessor.create_target_variable(df)
    
    # Create features
    features_df = preprocessor.create_feature_set(df)
    
    # Add composite scores
    features_df = preprocessor.create_composite_scores(features_df)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = preprocessor.split_and_scale_data(
        features_df, df['mental_health_risk']
    )
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    
    # Display summary
    print(f"\n📋 PREPROCESSING SUMMARY")
    print("-" * 40)
    print(f"Original dataset: {len(df)} samples")
    print(f"Features created: {len(features_df.columns)}")
    print(f"Target variable: mental_health_risk (binary)")
    print(f"Class distribution:")
    print(f"  - Low Risk: {(y_train == 0).sum() + (y_test == 0).sum()} samples")
    print(f"  - High Risk: {(y_train == 1).sum() + (y_test == 1).sum()} samples")
    
    return df, features_df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df, features_df, X_train, X_test, y_train, y_test = main()
    print(f"\n✅ Preprocessing complete!")
    print(f"Next step: Train mental health risk classification model")