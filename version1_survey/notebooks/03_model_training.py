#!/usr/bin/env python3
"""
MindScope: Model Training for Mental Health Risk Classification
==============================================================
Train and evaluate multiple ML models for mental health risk assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MentalHealthClassifier:
    """Mental Health Risk Classification Model"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = []
        
    def load_processed_data(self):
        """Load preprocessed training and test data"""
        print("📂 Loading processed data...")
        
        train_data = pd.read_csv("data/survey_data/processed/train.csv")
        test_data = pd.read_csv("data/survey_data/processed/test.csv")
        feature_names = pd.read_csv("data/survey_data/processed/feature_names.csv")
        
        # Separate features and targets
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        self.feature_names = feature_names['feature'].tolist()
        
        print(f"✅ Data loaded:")
        print(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"   Features: {self.feature_names}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize different ML models for comparison"""
        print(f"\n🤖 Initializing ML models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'SVM': SVC(
                random_state=42, probability=True, class_weight='balanced'
            )
        }
        
        print(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def cross_validate_models(self, X_train, y_train):
        """Perform cross-validation for all models"""
        print(f"\n🔄 Performing 5-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"   Evaluating {name}...")
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            cv_results[name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }
            print(f"      F1 Score: {scores.mean():.3f} (±{scores.std():.3f})")
        
        return cv_results
    
    def train_models(self, X_train, y_train):
        """Train all models on full training set"""
        print(f"\n🚀 Training models on full training set...")
        
        trained_models = {}
        for name, model in self.models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        print(f"✅ All models trained successfully!")
        return trained_models
    
    def evaluate_models(self, trained_models, X_test, y_test):
        """Evaluate all trained models on test set"""
        print(f"\n📊 Evaluating models on test set...")
        
        results = {}
        
        for name, model in trained_models.items():
            print(f"\n--- {name} ---")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"   F1 Score: {f1:.3f}")
            print(f"   ROC AUC: {roc_auc:.3f}")
            print(f"   Classification Report:")
            print(f"   {classification_report(y_test, y_pred)}")
        
        return results
    
    def select_best_model(self, cv_results, test_results, trained_models):
        """Select the best performing model"""
        print(f"\n🏆 Selecting best model...")
        
        # Combine CV and test performance
        model_scores = {}
        for name in self.models.keys():
            cv_f1 = cv_results[name]['mean_f1']
            test_f1 = test_results[name]['f1_score']
            combined_score = (cv_f1 + test_f1) / 2  # Average of CV and test F1
            
            model_scores[name] = {
                'cv_f1': cv_f1,
                'test_f1': test_f1,
                'combined_score': combined_score
            }
        
        # Find best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['combined_score'])
        self.best_model = trained_models[best_model_name]
        
        print(f"🥇 Best Model: {best_model_name}")
        print(f"   CV F1: {model_scores[best_model_name]['cv_f1']:.3f}")
        print(f"   Test F1: {model_scores[best_model_name]['test_f1']:.3f}")
        print(f"   Combined Score: {model_scores[best_model_name]['combined_score']:.3f}")
        
        return best_model_name, model_scores
    
    def analyze_feature_importance(self, best_model_name, trained_models):
        """Analyze feature importance for interpretability"""
        print(f"\n🔍 Analyzing feature importance...")
        
        best_model = trained_models[best_model_name]
        
        # Get feature importance based on model type
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models
            importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            # Linear models
            importance = np.abs(best_model.coef_[0])
        else:
            print("   Feature importance not available for this model type")
            return None
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 10 Most Important Features:")
        for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"   {idx:2d}. {row['feature']:<30} | {row['importance']:.3f}")
        
        return feature_importance
    
    def create_visualizations(self, results, y_test, best_model_name):
        """Create performance visualizations"""
        print(f"\n📈 Creating performance visualizations...")
        
        # Create results directory
        Path("version1_survey/results/figures").mkdir(parents=True, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MindScope: Mental Health Risk Classification Results', fontsize=16, fontweight='bold')
        
        # 1. Model Comparison (F1 Scores)
        ax1 = axes[0, 0]
        model_names = list(results.keys())
        f1_scores = [results[name]['f1_score'] for name in model_names]
        
        bars = ax1.bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Model Performance Comparison (F1 Score)', fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        
        # Highlight best model
        best_idx = model_names.index(best_model_name)
        bars[best_idx].set_color('gold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1_scores[i]:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, result) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            auc_score = result['roc_auc']
            
            linewidth = 3 if name == best_model_name else 2
            ax2.plot(fpr, tpr, color=colors[i], linewidth=linewidth,
                    label=f'{name} (AUC = {auc_score:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for Best Model
        ax3 = axes[1, 0]
        cm = results[best_model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        ax3.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Performance Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create performance summary text
        best_result = results[best_model_name]
        summary_text = f"""
        Best Model: {best_model_name}
        
        Performance Metrics:
        • F1 Score: {best_result['f1_score']:.3f}
        • ROC AUC: {best_result['roc_auc']:.3f}
        
        Test Set Results:
        • Total Samples: {len(y_test)}
        • Correct Predictions: {(results[best_model_name]['y_pred'] == y_test).sum()}
        • Accuracy: {(results[best_model_name]['y_pred'] == y_test).mean():.3f}
        
        Model Interpretation:
        This model can identify individuals at high
        risk for mental health issues based on their
        social media usage patterns and psychological
        indicators from survey responses.
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('version1_survey/results/figures/model_performance.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved performance visualization to results/figures/model_performance.png")
        
        return fig
    
    def save_best_model(self, best_model_name, trained_models, feature_importance):
        """Save the best model and related artifacts"""
        print(f"\n💾 Saving best model and artifacts...")
        
        # Create models directory
        Path("version1_survey/models/saved_models").mkdir(parents=True, exist_ok=True)
        Path("version1_survey/results/metrics").mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump(trained_models[best_model_name], f"version1_survey/models/saved_models/{best_model_name.lower().replace(' ', '_')}_model.pkl")
        
        # Save feature importance if available
        if feature_importance is not None:
            feature_importance.to_csv("version1_survey/results/metrics/feature_importance.csv", index=False)
            print(f"   ✅ Saved feature importance rankings")
        else:
            print(f"   ℹ️  Feature importance not available for {best_model_name}")
        
        # Save model metadata
        metadata = {
            'model_name': best_model_name,
            'model_type': type(trained_models[best_model_name]).__name__,
            'features': self.feature_names,
            'num_features': len(self.feature_names),
            'target_variable': 'mental_health_risk'
        }
        
        pd.Series(metadata).to_csv("version1_survey/models/saved_models/model_metadata.csv")
        
        print(f"   ✅ Saved model: {best_model_name}")
        print(f"   ✅ Saved model metadata")

def main():
    """Main model training pipeline"""
    print("🤖 MindScope: Mental Health Risk Classification Training")
    print("=" * 65)
    
    # Initialize classifier
    classifier = MentalHealthClassifier()
    
    # Load data
    X_train, X_test, y_train, y_test = classifier.load_processed_data()
    
    # Initialize models
    classifier.initialize_models()
    
    # Cross-validation
    cv_results = classifier.cross_validate_models(X_train, y_train)
    
    # Train models
    trained_models = classifier.train_models(X_train, y_train)
    
    # Evaluate models
    test_results = classifier.evaluate_models(trained_models, X_test, y_test)
    
    # Select best model
    best_model_name, model_scores = classifier.select_best_model(cv_results, test_results, trained_models)
    
    # Feature importance analysis
    feature_importance = classifier.analyze_feature_importance(best_model_name, trained_models)
    
    # Create visualizations
    classifier.create_visualizations(test_results, y_test, best_model_name)
    
    # Save best model
    classifier.save_best_model(best_model_name, trained_models, feature_importance)
    
    print(f"\n🎉 MODEL TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Best Model: {best_model_name}")
    print(f"Performance: {test_results[best_model_name]['f1_score']:.3f} F1 Score")
    print(f"Model saved to: models/saved_models/")
    print(f"Visualizations saved to: results/figures/")
    
    return classifier, trained_models, test_results

if __name__ == "__main__":
    classifier, models, results = main()
    print(f"\n✅ Training complete! Ready for deployment.")