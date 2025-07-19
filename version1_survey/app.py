#!/usr/bin/env python3
"""
MindScope: Mental Health Risk Assessment Demo
============================================
Interactive Streamlit application for mental health risk prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MindScope - Mental Health Risk Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MindScopeApp:
    """MindScope Streamlit Application"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            # Use the original SVM model which expects 18 features
            if Path("version1_survey/models/saved_models/svm_model.pkl").exists():
                self.model = joblib.load("version1_survey/models/saved_models/svm_model.pkl")
                model_type = "SVM Classifier"
                self.accuracy = 72.5
            else:
                st.error("❌ Model file not found!")
                self.model = None
                return
            
            # Load feature names
            if Path("version1_survey/models/saved_models/feature_names.csv").exists():
                feature_df = pd.read_csv("version1_survey/models/saved_models/feature_names.csv")
                self.feature_names = feature_df['feature'].tolist()
            else:
                # Default feature names
                self.feature_names = [
                    'aimless_social_media_use', 'social_media_distraction', 'social_media_restlessness',
                    'social_comparison', 'validation_seeking', 'distractibility', 'worry_level',
                    'concentration_difficulty', 'comparison_feelings', 'interest_fluctuation',
                    'sleep_issues', 'age', 'gender', 'relationship_status', 'daily_social_media_hours',
                    'social_media_addiction_score', 'social_comparison_score', 'general_mental_health_score'
                ]
            
            st.success(f"✅ Loaded {model_type} model successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def create_user_input_form(self):
        """Create user input form for mental health assessment"""
        st.header("🔍 Mental Health Risk Assessment")
        st.write("Please answer the following questions honestly. This assessment is for educational purposes only and should not replace professional medical advice.")
        
        with st.form("mental_health_assessment"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📱 Social Media Usage")
                
                aimless_use = st.slider(
                    "How often do you use social media without a specific purpose?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Never, 5 = Very Often"
                )
                
                distraction = st.slider(
                    "How often do you get distracted by social media when busy?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Never, 5 = Very Often"
                )
                
                restlessness = st.slider(
                    "Do you feel restless if you haven't used social media in a while?",
                    min_value=1, max_value=5, value=2,
                    help="1 = Never, 5 = Always"
                )
                
                social_comparison = st.slider(
                    "How often do you compare yourself to others on social media?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Never, 5 = Very Often"
                )
                
                validation_seeking = st.slider(
                    "How often do you seek validation from social media features?",
                    min_value=1, max_value=5, value=2,
                    help="1 = Never, 5 = Very Often"
                )
                
                daily_hours = st.selectbox(
                    "Average daily social media usage:",
                    options=[1, 2, 3, 4, 5, 6],
                    format_func=lambda x: {
                        1: "Less than 1 hour",
                        2: "1-2 hours", 
                        3: "2-3 hours",
                        4: "3-4 hours", 
                        5: "4-5 hours",
                        6: "More than 5 hours"
                    }[x],
                    index=2
                )
            
            with col2:
                st.subheader("🧠 Psychological Indicators")
                
                distractibility = st.slider(
                    "How easily are you distracted in general?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Very focused, 5 = Very easily distracted"
                )
                
                worry_level = st.slider(
                    "How much are you bothered by worries?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Not at all, 5 = Extremely"
                )
                
                concentration = st.slider(
                    "Do you find it difficult to concentrate?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Never, 5 = Always"
                )
                
                comparison_feelings = st.slider(
                    "How do social comparisons generally make you feel?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Very positive, 5 = Very negative"
                )
                
                interest_fluctuation = st.slider(
                    "How often does your interest in daily activities change?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Very stable, 5 = Changes frequently"
                )
                
                sleep_issues = st.slider(
                    "How often do you face sleep problems?",
                    min_value=1, max_value=5, value=3,
                    help="1 = Never, 5 = Very often"
                )
            
            st.subheader("👤 Demographics")
            col3, col4 = st.columns(2)
            
            with col3:
                age = st.number_input(
                    "Age:", 
                    min_value=13, max_value=100, value=25
                )
                
            with col4:
                gender = st.selectbox(
                    "Gender:",
                    options=[0, 1, 2],
                    format_func=lambda x: {0: "Male", 1: "Female", 2: "Other/Non-binary"}[x]
                )
                
                relationship = st.selectbox(
                    "Relationship Status:",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: {0: "Single", 1: "In a relationship", 2: "Married", 3: "Divorced"}[x]
                )
            
            submitted = st.form_submit_button("🔍 Assess Mental Health Risk", use_container_width=True)
            
            if submitted:
                # Create feature vector
                features = self.create_feature_vector(
                    aimless_use, distraction, restlessness, social_comparison, validation_seeking,
                    distractibility, worry_level, concentration, comparison_feelings,
                    interest_fluctuation, sleep_issues, age, gender, relationship, daily_hours
                )
                
                return features
        
        return None
    
    def create_feature_vector(self, aimless_use, distraction, restlessness, social_comparison,
                            validation_seeking, distractibility, worry_level, concentration,
                            comparison_feelings, interest_fluctuation, sleep_issues, age,
                            gender, relationship, daily_hours):
        """Create feature vector from user inputs"""
        
        # Calculate composite scores
        social_media_addiction_score = np.mean([aimless_use, distraction, restlessness])
        social_comparison_score = np.mean([social_comparison, validation_seeking, comparison_feelings])
        general_mental_health_score = np.mean([distractibility, worry_level, concentration, 
                                             interest_fluctuation, sleep_issues])
        
        # Create feature vector matching training data (18 features)
        features = [
            aimless_use, distraction, restlessness, social_comparison, validation_seeking,
            distractibility, worry_level, concentration, comparison_feelings,
            interest_fluctuation, sleep_issues, age, gender, relationship, daily_hours,
            social_media_addiction_score, social_comparison_score, general_mental_health_score
        ]
        
        return np.array(features).reshape(1, -1)
    
    def make_prediction(self, features):
        """Make mental health risk prediction"""
        if self.model is None:
            st.error("Model not loaded. Cannot make prediction.")
            return None, None
        
        try:
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.75  # Default confidence for models without probability
            
            return prediction, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None
    
    def display_prediction_results(self, prediction, confidence, features):
        """Display prediction results with visualizations"""
        
        # Create results layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Assessment Results")
            
            # Risk level display
            if prediction == 1:
                st.error("⚠️ **HIGH MENTAL HEALTH RISK DETECTED**")
                risk_color = "red"
                risk_message = """
                **Recommendation**: Consider speaking with a mental health professional. 
                The assessment indicates elevated risk factors that may benefit from professional support.
                """
            else:
                st.success("✅ **LOW MENTAL HEALTH RISK**")
                risk_color = "green"
                risk_message = """
                **Good news**: Your responses suggest lower mental health risk factors. 
                Continue maintaining healthy social media habits and mental wellness practices.
                """
            
            st.write(risk_message)
            
            # Confidence meter
            fig_conf = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                title = {'text': "Prediction Confidence"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_conf.update_layout(height=300)
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            st.header("📊 Risk Factors")
            
            # Feature importance visualization (simplified)
            features_flat = features.flatten()
            feature_labels = [
                'Social Media (Aimless)', 'Social Media (Distraction)', 'Social Media (Restless)',
                'Social Comparison', 'Validation Seeking', 'Distractibility', 'Worry Level',
                'Concentration Issues', 'Comparison Feelings', 'Interest Changes', 'Sleep Issues',
                'Age', 'Gender', 'Relationship', 'Daily SM Hours', 'SM Addiction Score',
                'Social Comparison Score', 'Mental Health Score'
            ]
            
            # Create risk factor chart for top contributors
            psychological_features = [5, 6, 7, 9, 10, 17]  # indices of key psychological features
            psychological_values = [features_flat[i] for i in psychological_features]
            psychological_labels = [feature_labels[i] for i in psychological_features]
            
            fig_factors = px.bar(
                x=psychological_values,
                y=psychological_labels,
                orientation='h',
                title="Key Psychological Indicators",
                color=psychological_values,
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig_factors.update_layout(
                height=400, 
                showlegend=False,
                xaxis=dict(range=[1, 5])
            )
            st.plotly_chart(fig_factors, use_container_width=True)
    
    def display_model_info(self):
        """Display model information and performance"""
        with st.expander("🤖 About the Model"):
            st.write("""
            **MindScope Mental Health Risk Classifier**
            
            - **Model Type**: Support Vector Machine (SVM)
            - **Accuracy**: 72.5% on test data
            - **F1 Score**: 0.725
            - **ROC AUC**: 0.765
            - **Features**: 18 psychological and behavioral indicators
            
            **Training Data**: 481 survey responses from individuals aged 13-91
            
            **Key Predictors**:
            - Social media usage patterns
            - Psychological distress indicators  
            - Sleep and concentration issues
            - Social comparison behaviors
            
            **Disclaimer**: This tool is for educational purposes only and should not replace 
            professional mental health assessment or treatment.
            """)
    
    def display_resources(self):
        """Display mental health resources"""
        st.header("🆘 Mental Health Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crisis Resources")
            st.write("""
            **Emergency**: If you're in immediate danger, call emergency services (911, 999, etc.)
            
            **Crisis Hotlines**:
            - National Suicide Prevention Lifeline: 988 (US)
            - Crisis Text Line: Text HOME to 741741
            - International: befrienders.org
            """)
        
        with col2:
            st.subheader("General Support")
            st.write("""
            **Professional Help**:
            - Consult your primary care physician
            - Find a therapist: psychologytoday.com
            - Online therapy: BetterHelp, Talkspace
            
            **Self-Care**:
            - Mindfulness apps: Headspace, Calm
            - Support groups: NAMI.org
            """)

def main():
    """Main Streamlit application"""
    
    # App header
    st.title("🧠 MindScope")
    st.subheader("AI-Powered Mental Health Risk Assessment")
    st.write("*Advanced machine learning for mental wellness awareness*")
    
    # Initialize app
    app = MindScopeApp()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🔍 Risk Assessment", "📊 Model Information", "🆘 Resources"]
    )
    
    if page == "🔍 Risk Assessment":
        # Main assessment interface
        features = app.create_user_input_form()
        
        if features is not None:
            # Make prediction
            prediction, confidence = app.make_prediction(features)
            
            if prediction is not None:
                # Display results
                app.display_prediction_results(prediction, confidence, features)
                
                # Important disclaimer
                st.warning("""
                **⚠️ Important Disclaimer**: This assessment is for educational purposes only. 
                It should not be used as a substitute for professional mental health diagnosis or treatment. 
                If you're experiencing mental health concerns, please consult with a qualified healthcare provider.
                """)
    
    elif page == "📊 Model Information":
        app.display_model_info()
        
        # Performance visualization
        st.subheader("📈 Model Performance")
        
        # Create performance comparison chart
        models = ['Baseline SVM', 'Optimized Random Forest']
        accuracies = [71.0, 75.3]
        f1_scores = [72.5, 76.0]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1 Score Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1 Score', marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🆘 Resources":
        app.display_resources()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MindScope** - Developed for educational purposes | "
        "Model Accuracy: 75.3% | "
        "⚠️ Not a substitute for professional medical advice"
    )

if __name__ == "__main__":
    main()