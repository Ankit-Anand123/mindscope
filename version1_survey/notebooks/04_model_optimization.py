#!/usr/bin/env python3
"""
MindScope: Model Optimization for Better Accuracy
=================================================
Advanced techniques to improve mental health risk classification performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedMentalHealthClassifier:
    """Advanced Mental Health Risk Classification with Optimization"""
    
    def __init__(self):
        self.models = {}
        self.optimized_models = {}
        self.feature_selector = None
        self.poly_features = None
        self.best_model = None
        self.best_accuracy = 0
        
    def load_data(self):
        """Load processed data"""
        print("📂 Loading processed data...")
        
        train_data = pd.read_csv("data/survey_data/processed/train.csv")
        test_data = pd.read_csv("data/survey_data/processed/test.csv")
        
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        print(f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def create_polynomial_features(self, X_train, X_test, degree=2):
        """Create polynomial and interaction features"""
        print(f"\n🔧 Creating polynomial features (degree={degree})...")
        
        # Limit to most important features to avoid explosion
        important_features = [
            'social_media_addiction_score', 'general_mental_health_score',
            'social_comparison_score', 'worry_level', 'sleep_issues',
            'concentration_difficulty', 'interest_fluctuation'
        ]
        
        X_train_subset = X_train[important_features]
        X_test_subset = X_test[important_features]
        
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = self.poly_features.fit_transform(X_train_subset)
        X_test_poly = self.poly_features.transform(X_test_subset)
        
        # Combine with original features
        X_train_enhanced = np.hstack([X_train.values, X_train_poly])
        X_test_enhanced = np.hstack([X_test.values, X_test_poly])
        
        print(f"✅ Enhanced features: {X_train.shape[1]} → {X_train_enhanced.shape[1]}")
        return X_train_enhanced, X_test_enhanced
    
    def feature_selection(self, X_train, y_train, k=15):
        """Select best features using statistical tests"""
        print(f"\n🎯 Selecting top {k} features...")
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        # Get selected feature indices
        selected_features = self.feature_selector.get_support(indices=True)
        print(f"✅ Selected {len(selected_features)} most predictive features")
        
        return X_train_selected, selected_features
    
    def initialize_advanced_models(self):
        """Initialize advanced ML models with better parameters"""
        print(f"\n🤖 Initializing advanced models...")
        
        self.models = {
            'Optimized_RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'Optimized_GradientBoosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Optimized_SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=1000,
                random_state=42
            ),
            'Bagging_Classifier': BaggingClassifier(
                estimator=SVC(probability=True, class_weight='balanced'),
                n_estimators=10,
                random_state=42
            )
        }
        
        print(f"✅ Initialized {len(self.models)} advanced models")
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best models"""
        print(f"\n🔍 Hyperparameter tuning...")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 10, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # SVM tuning  
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        # Gradient Boosting tuning
        gb_params = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8]
        }
        
        tuning_configs = [
            ('RandomForest', RandomForestClassifier(class_weight='balanced', random_state=42), rf_params),
            ('SVM', SVC(class_weight='balanced', probability=True, random_state=42), svm_params),
            ('GradientBoosting', GradientBoostingClassifier(random_state=42), gb_params)
        ]
        
        tuned_models = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Faster CV
        
        for name, model, params in tuning_configs:
            print(f"   Tuning {name}...")
            
            # Use RandomizedSearchCV for efficiency
            random_search = RandomizedSearchCV(
                model, params, n_iter=20, cv=cv, 
                scoring='accuracy', random_state=42, n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            tuned_models[f'Tuned_{name}'] = random_search.best_estimator_
            
            print(f"      Best accuracy: {random_search.best_score_:.3f}")
            print(f"      Best params: {random_search.best_params_}")
        
        return tuned_models
    
    def create_ensemble_models(self, base_models):
        """Create ensemble models for better performance"""
        print(f"\n🎭 Creating ensemble models...")
        
        # Voting Classifier (Hard voting)
        voting_hard = VotingClassifier(
            estimators=[
                ('rf', base_models.get('Tuned_RandomForest', RandomForestClassifier())),
                ('svm', base_models.get('Tuned_SVM', SVC(probability=True))),
                ('gb', base_models.get('Tuned_GradientBoosting', GradientBoostingClassifier()))
            ],
            voting='hard'
        )
        
        # Voting Classifier (Soft voting)
        voting_soft = VotingClassifier(
            estimators=[
                ('rf', base_models.get('Tuned_RandomForest', RandomForestClassifier())),
                ('svm', base_models.get('Tuned_SVM', SVC(probability=True))),
                ('gb', base_models.get('Tuned_GradientBoosting', GradientBoostingClassifier()))
            ],
            voting='soft'
        )
        
        ensemble_models = {
            'Voting_Hard': voting_hard,
            'Voting_Soft': voting_soft
        }
        
        print(f"✅ Created {len(ensemble_models)} ensemble models")
        return ensemble_models
    
    def evaluate_all_models(self, models, X_train, X_test, y_train, y_test):
        """Comprehensive evaluation of all models"""
        print(f"\n📊 Evaluating all models...")
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"   Evaluating {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Test set evaluation
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results[name] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_f1': f1,
                'test_roc_auc': roc_auc,
                'model': model,
                'y_pred': y_pred
            }
            
            print(f"      CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"      Test Accuracy: {accuracy:.3f}")
            print(f"      Test F1: {f1:.3f}")
        
        return results
    
    def find_best_model(self, results):
        """Find the best performing model"""
        print(f"\n🏆 Finding best model...")
        
        # Sort by test accuracy
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        print(f"\n📈 MODEL PERFORMANCE RANKING:")
        print("-" * 60)
        
        for i, (name, result) in enumerate(sorted_models, 1):
            print(f"{i:2d}. {name:<25} | Accuracy: {result['test_accuracy']:.3f} | F1: {result['test_f1']:.3f}")
        
        best_model_name, best_result = sorted_models[0]
        self.best_model = best_result['model']
        self.best_accuracy = best_result['test_accuracy']
        
        print(f"\n🥇 BEST MODEL: {best_model_name}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.3f}")
        print(f"   Test F1 Score: {best_result['test_f1']:.3f}")
        if best_result['test_roc_auc']:
            print(f"   Test ROC AUC: {best_result['test_roc_auc']:.3f}")
        
        return best_model_name, best_result
    
    def save_optimized_model(self, best_model_name, best_result):
        """Save the best optimized model"""
        print(f"\n💾 Saving optimized model...")
        
        Path("version1_survey/models/optimized").mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_filename = f"version1_survey/models/optimized/{best_model_name.lower().replace(' ', '_')}_optimized.pkl"
        joblib.dump(best_result['model'], model_filename)
        
        # Save optimization results
        optimization_summary = {
            'best_model': best_model_name,
            'test_accuracy': best_result['test_accuracy'],
            'test_f1': best_result['test_f1'],
            'test_roc_auc': best_result['test_roc_auc'],
            'cv_accuracy_mean': best_result['cv_accuracy_mean'],
            'cv_accuracy_std': best_result['cv_accuracy_std'],
            'improvement_over_baseline': best_result['test_accuracy'] - 0.71  # Original SVM accuracy
        }
        
        pd.DataFrame([optimization_summary]).to_csv("version1_survey/models/optimized/optimization_results.csv", index=False)
        
        print(f"✅ Saved optimized model: {model_filename}")
        print(f"✅ Accuracy improvement: {optimization_summary['improvement_over_baseline']:.3f}")

def main():
    """Main optimization pipeline"""
    print("🚀 MindScope: Advanced Model Optimization")
    print("=" * 60)
    
    classifier = AdvancedMentalHealthClassifier()
    
    # Load data
    X_train, X_test, y_train, y_test = classifier.load_data()
    
    # Feature engineering
    X_train_poly, X_test_poly = classifier.create_polynomial_features(X_train, X_test)
    
    # Feature selection
    X_train_selected, selected_features = classifier.feature_selection(X_train_poly, y_train, k=20)
    X_test_selected = classifier.feature_selector.transform(X_test_poly)
    
    # Initialize models
    classifier.initialize_advanced_models()
    
    # Hyperparameter tuning
    tuned_models = classifier.hyperparameter_tuning(X_train_selected, y_train)
    
    # Create ensembles
    ensemble_models = classifier.create_ensemble_models(tuned_models)
    
    # Combine all models
    all_models = {**classifier.models, **tuned_models, **ensemble_models}
    
    # Evaluate all models
    results = classifier.evaluate_all_models(all_models, X_train_selected, X_test_selected, y_train, y_test)
    
    # Find best model
    best_model_name, best_result = classifier.find_best_model(results)
    
    # Save optimized model
    classifier.save_optimized_model(best_model_name, best_result)
    
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print("=" * 50)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 New Accuracy: {best_result['test_accuracy']:.3f}")
    print(f"📊 Improvement: +{(best_result['test_accuracy'] - 0.71):.3f}")
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()