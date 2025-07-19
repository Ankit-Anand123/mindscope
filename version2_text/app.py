#!/usr/bin/env python3
"""
MindScope: DistilBERT Mental Health Text Classification Demo
==========================================================
Interactive Streamlit app for mental health risk detection in text
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MindScope - AI Mental Health Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MindScopeTextApp:
    """MindScope DistilBERT Text Classification Application"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """Load the trained DistilBERT model and tokenizer"""
        model_path = "version2_text/models"
        
        try:
            if Path(model_path).exists():
                # Load tokenizer and model
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
                
                st.success("✅ DistilBERT model loaded successfully!")
                
                # Load model info
                info_path = f"{model_path}/model_info.json"
                if Path(info_path).exists():
                    with open(info_path, 'r') as f:
                        self.model_info = json.load(f)
                else:
                    self.model_info = {
                        'model_name': 'distilbert-base-uncased',
                        'accuracy': 0.90,
                        'f1_score': 0.889
                    }
            else:
                st.error("❌ Model not found! Please train the model first.")
                self.model = None
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict_text(self, text):
        """Predict mental health risk for input text"""
        if self.model is None or self.tokenizer is None:
            return None, None, None
        
        try:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = torch.max(probabilities).item()
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                probs_array = probabilities[0].cpu().numpy()
            
            return prediction, confidence, probs_array
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None, None
    
    def get_attention_visualization(self, text):
        """Get attention weights for text interpretation"""
        if self.model is None or self.tokenizer is None:
            return None, None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True,
                max_length=128
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
            
            # Extract attention weights (average across heads and layers)
            attention = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Average attention for each token (from CLS token)
            token_attention = attention[0, 1:]  # Skip CLS token attention to itself
            
            return tokens[1:], token_attention[:len(tokens)-1]  # Skip CLS and SEP
            
        except Exception as e:
            st.error(f"Error getting attention weights: {e}")
            return None, None
    
    def analyze_text_features(self, text):
        """Analyze text features that might indicate mental health risk"""
        # Mental health risk indicators
        distress_keywords = [
            'overwhelmed', 'anxious', 'depressed', 'hopeless', 'worthless',
            'empty', 'numb', 'exhausted', 'isolat', 'alone', 'sleep',
            'tired', 'worry', 'stress', 'panic', 'fear', 'sad', 'cry',
            'hurt', 'pain', 'struggle', 'difficult', 'hard', 'can\'t',
            'nothing', 'pointless', 'useless', 'helpless'
        ]
        
        positive_keywords = [
            'happy', 'grateful', 'positive', 'good', 'better', 'improve',
            'hope', 'love', 'joy', 'excited', 'amazing', 'wonderful',
            'great', 'excellent', 'fantastic', 'beautiful', 'peaceful',
            'calm', 'confident', 'strong', 'proud', 'thankful'
        ]
        
        text_lower = text.lower()
        
        # Count keywords
        distress_count = sum(1 for keyword in distress_keywords if keyword in text_lower)
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        
        # Text statistics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        return {
            'distress_keywords': distress_count,
            'positive_keywords': positive_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'questions': question_count,
            'exclamations': exclamation_count,
            'distress_ratio': distress_count / max(word_count, 1),
            'positive_ratio': positive_count / max(word_count, 1)
        }
    
    def create_text_input_interface(self):
        """Create the main text input interface"""
        st.header("🔍 Mental Health Text Analysis")
        st.write("Enter any text message and get an AI-powered mental health risk assessment using DistilBERT.")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Type your message", "📝 Use example texts"],
            horizontal=True
        )
        
        if input_method == "✍️ Type your message":
            user_text = st.text_area(
                "Enter your message:",
                placeholder="Type how you're feeling or any message you'd like analyzed...",
                height=120,
                help="The AI will analyze your text for mental health risk indicators"
            )
        else:
            # Example texts
            examples = {
                "High Risk Example": "I haven't slept in days and feel completely overwhelmed. Nothing seems to matter anymore and I just want to stay in bed forever.",
                "Low Risk Example": "Had a great therapy session today and feeling hopeful about the future. Grateful for the support system I have.",
                "Neutral Example": "Going to work today and then meeting friends for dinner. Should be a good day overall.",
                "Mixed Emotions": "Struggling with anxiety lately but trying to stay positive and focus on self-care activities."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            user_text = examples[selected_example]
            st.text_area("Selected text:", value=user_text, height=80, disabled=True)
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analyze Mental Health Risk",
                use_container_width=True,
                type="primary"
            )
        
        if analyze_button and user_text:
            return user_text.strip()
        
        return None
    
    def display_prediction_results(self, text, prediction, confidence, probabilities):
        """Display prediction results with detailed analysis"""
        
        # Main results layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 AI Analysis Results")
            
            # Risk level display
            if prediction == 1:
                st.error("⚠️ **HIGH MENTAL HEALTH RISK DETECTED**")
                risk_color = "red"
                risk_emoji = "🚨"
                risk_message = f"""
                **AI Assessment**: The text shows indicators of mental health distress.
                
                **Recommendation**: Consider reaching out to:
                - A mental health professional
                - A trusted friend or family member  
                - A crisis helpline if needed
                
                **Confidence**: {confidence:.1%} certain of this assessment
                """
            else:
                st.success("✅ **LOW MENTAL HEALTH RISK**")
                risk_color = "green"
                risk_emoji = "✅"
                risk_message = f"""
                **AI Assessment**: The text suggests positive or neutral mental health indicators.
                
                **Observation**: Continue maintaining healthy mental wellness practices.
                
                **Confidence**: {confidence:.1%} certain of this assessment
                """
            
            st.markdown(risk_message)
            
            # Probability breakdown
            st.subheader("📊 Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric(
                    "Low Risk Probability", 
                    f"{probabilities[0]:.1%}",
                    delta=None
                )
            
            with prob_col2:
                st.metric(
                    "High Risk Probability", 
                    f"{probabilities[1]:.1%}",
                    delta=None
                )
            
            # Confidence gauge
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "AI Confidence Level"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
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
            st.header("📋 Text Analysis")
            
            # Analyze text features
            features = self.analyze_text_features(text)
            
            # Key metrics
            st.metric("Word Count", features['word_count'])
            st.metric("Distress Keywords", features['distress_keywords'])
            st.metric("Positive Keywords", features['positive_keywords'])
            
            # Feature visualization
            feature_data = {
                'Metric': ['Distress Signals', 'Positive Signals', 'Questions', 'Exclamations'],
                'Count': [
                    features['distress_keywords'],
                    features['positive_keywords'], 
                    features['questions'],
                    features['exclamations']
                ]
            }
            
            fig_features = px.bar(
                x=feature_data['Count'],
                y=feature_data['Metric'],
                orientation='h',
                title="Text Feature Analysis",
                color=feature_data['Count'],
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig_features.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_features, use_container_width=True)
        
        # Attention visualization
        st.header("🎯 AI Attention Analysis")
        st.write("See which words the AI model focused on when making its decision:")
        
        tokens, attention_weights = self.get_attention_visualization(text)
        
        if tokens is not None and attention_weights is not None:
            # Create attention visualization
            attention_df = pd.DataFrame({
                'Token': tokens[:min(len(tokens), len(attention_weights))],
                'Attention': attention_weights[:min(len(tokens), len(attention_weights))]
            })
            
            # Remove special tokens and clean
            attention_df = attention_df[~attention_df['Token'].str.startswith('[')]
            attention_df = attention_df[~attention_df['Token'].str.startswith('##')]
            attention_df = attention_df.head(15)  # Top 15 tokens
            
            if not attention_df.empty:
                fig_attention = px.bar(
                    attention_df,
                    x='Attention',
                    y='Token',
                    orientation='h',
                    title="Words the AI Focused On (Attention Weights)",
                    color='Attention',
                    color_continuous_scale='Viridis'
                )
                fig_attention.update_layout(height=400)
                st.plotly_chart(fig_attention, use_container_width=True)
        
        # Important disclaimer
        st.warning("""
        **⚠️ Important Disclaimer**: This AI analysis is for educational and awareness purposes only. 
        It should NOT be used as a substitute for professional mental health diagnosis, treatment, or advice. 
        If you're experiencing mental health concerns, please consult with a qualified healthcare provider or contact a crisis helpline.
        """)
    
    def display_model_info(self):
        """Display information about the DistilBERT model"""
        st.header("🤖 About the AI Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.write("""
            **Base Model**: DistilBERT (Distilled BERT)
            - **Type**: Transformer-based language model
            - **Parameters**: ~66M parameters
            - **Speed**: 60% faster than BERT
            - **Performance**: 97% of BERT's performance
            
            **Fine-tuning Details**:
            - **Task**: Binary text classification
            - **Classes**: Low Risk / High Risk
            - **Training Data**: 100 mental health text samples
            - **Epochs**: 3 epochs with early stopping
            """)
        
        with col2:
            st.subheader("Performance Metrics")
            
            # Performance metrics
            metrics_data = {
                'Metric': ['Accuracy', 'F1 Score', 'Precision (High Risk)', 'Recall (Low Risk)'],
                'Score': [90.0, 88.9, 100.0, 100.0]
            }
            
            fig_metrics = px.bar(
                x=metrics_data['Score'],
                y=metrics_data['Metric'],
                orientation='h',
                title="Model Performance (%)",
                color=metrics_data['Score'],
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            fig_metrics.update_layout(height=300)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Technical details
        with st.expander("🔧 Technical Implementation Details"):
            st.write("""
            **Training Process**:
            - **Optimizer**: AdamW with learning rate 2e-5
            - **Batch Size**: 8 samples per batch
            - **Max Sequence Length**: 128 tokens
            - **Validation Split**: 80% train, 20% validation
            
            **Text Preprocessing**:
            - Tokenization using DistilBERT tokenizer
            - Truncation and padding to max length
            - Attention mask generation
            
            **Model Architecture**:
            - DistilBERT base encoder
            - Classification head with dropout
            - Binary cross-entropy loss
            - Softmax activation for probabilities
            """)
    
    def display_examples_and_tips(self):
        """Display example texts and usage tips"""
        st.header("💡 Examples & Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Example Texts to Try")
            
            examples = {
                "🚨 High Risk Examples": [
                    "I haven't slept in days and feel completely overwhelmed",
                    "Everything seems pointless and I just want to give up",
                    "I feel so alone and nobody understands what I'm going through"
                ],
                "✅ Low Risk Examples": [
                    "Had a great day with friends and feeling grateful",
                    "Working on self-improvement and feeling hopeful",
                    "Challenging week but I'm managing well with good support"
                ]
            }
            
            for category, texts in examples.items():
                st.write(f"**{category}:**")
                for text in texts:
                    st.write(f"• \"{text}\"")
                st.write("")
        
        with col2:
            st.subheader("🎯 How It Works")
            st.write("""
            **The AI analyzes**:
            - **Language patterns** indicating distress or wellness
            - **Emotional keywords** and sentiment
            - **Context and meaning** using transformer attention
            - **Linguistic features** like question patterns
            
            **Best practices**:
            - Use natural, conversational language
            - Include context about your feelings
            - Be honest in your expression
            - Remember this is a screening tool, not diagnosis
            
            **Limitations**:
            - Works best with English text
            - Trained on general mental health indicators
            - Cannot replace professional assessment
            - May not capture all cultural contexts
            """)

def main():
    """Main Streamlit application"""
    
    # App header
    st.title("🧠 MindScope v2.0")
    st.subheader("AI-Powered Mental Health Text Analysis")
    st.write("*Advanced DistilBERT transformer for mental wellness awareness*")
    
    # Initialize app
    app = MindScopeTextApp()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🔍 Text Analysis", "🤖 Model Information", "💡 Examples & Tips", "🆘 Resources"]
    )
    
    if page == "🔍 Text Analysis":
        # Main text analysis interface
        user_text = app.create_text_input_interface()
        
        if user_text:
            # Make prediction
            with st.spinner("🤖 AI is analyzing your text..."):
                prediction, confidence, probabilities = app.predict_text(user_text)
            
            if prediction is not None:
                # Display results
                app.display_prediction_results(user_text, prediction, confidence, probabilities)
    
    elif page == "🤖 Model Information":
        app.display_model_info()
    
    elif page == "💡 Examples & Tips":
        app.display_examples_and_tips()
    
    elif page == "🆘 Resources":
        st.header("🆘 Mental Health Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Crisis Resources")
            st.write("""
            **Emergency**: If you're in immediate danger, call emergency services (911, 999, etc.)
            
            **Crisis Hotlines**:
            - **National Suicide Prevention Lifeline**: 988 (US)
            - **Crisis Text Line**: Text HOME to 741741
            - **International**: befrienders.org
            - **Trevor Project** (LGBTQ): 1-866-488-7386
            
            **Online Crisis Support**:
            - Crisis Text Line: crisistextline.org
            - SAMHSA National Helpline: 1-800-662-4357
            """)
        
        with col2:
            st.subheader("💚 General Support")
            st.write("""
            **Professional Help**:
            - Find a therapist: psychologytoday.com
            - Online therapy: BetterHelp, Talkspace
            - Mental health apps: Headspace, Calm, Sanvello
            
            **Self-Care Resources**:
            - Mindfulness: mindful.org
            - Support groups: NAMI.org
            - Mental health education: mentalhealth.gov
            
            **Academic/Work Support**:
            - Student counseling services
            - Employee assistance programs
            - Peer support groups
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MindScope v2.0** - DistilBERT-powered mental health text analysis | "
        "90% Accuracy | 88.9% F1 Score | "
        "⚠️ Educational purposes only - Not a substitute for professional care"
    )

if __name__ == "__main__":
    main()