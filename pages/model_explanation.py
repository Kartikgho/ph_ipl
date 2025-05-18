import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

from config import VISUALIZATION_IMAGES

def show():
    """
    Page for explaining the prediction model and its components
    """
    st.title("Prediction Model Explanation")
    
    # Display a visualization image
    st.image(random.choice(VISUALIZATION_IMAGES), use_column_width=True)
    
    # Model overview
    st.header("Model Overview")
    
    st.markdown("""
    Our IPL prediction system uses an ensemble of machine learning models to forecast match outcomes, 
    team scores, and player performances. The system combines multiple algorithms to leverage the 
    strengths of different approaches and provide robust predictions.
    
    The core prediction system consists of:
    
    1. **Match Outcome Predictor**: Forecasts the winning team based on historical data and current form
    2. **Score Predictor**: Estimates the final score for both teams
    3. **Player Performance Predictor**: Projects individual player statistics
    4. **LLM Reasoning System**: Provides natural language explanations for predictions
    """)
    
    # Model architecture
    st.header("Model Architecture")
    
    st.markdown("""
    ### Ensemble Approach
    
    Our prediction system uses an ensemble approach that combines multiple algorithms:
    
    - **Gradient Boosting Machines**: XGBoost models that capture complex non-linear relationships
    - **Random Forests**: Provide robust predictions by averaging multiple decision trees
    - **Neural Networks**: Capture intricate patterns and relationships in cricket data
    - **Time Series Models**: Account for team momentum and trends over time
    
    The final prediction is a weighted average of these individual models, with weights determined 
    through cross-validation.
    """)
    
    # Create a simple visualization of the ensemble architecture
    st.subheader("Ensemble Model Architecture")
    
    # Mock data for model architecture visualization
    models = ['XGBoost', 'Random Forest', 'Neural Network', 'Time Series']
    accuracy = [0.76, 0.72, 0.74, 0.68]
    model_weight = [0.35, 0.25, 0.30, 0.10]
    
    # Create architecture dataframe
    model_data = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Weight': model_weight
    })
    
    # Display model architecture
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualization of model weights
        fig = px.pie(
            model_data, 
            values='Weight', 
            names='Model',
            title='Model Contribution in Ensemble'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(model_data)
    
    # Feature importance
    st.header("Feature Importance")
    
    st.markdown("""
    Our prediction model uses a wide range of features derived from historical cricket data. 
    The most influential features include:
    """)
    
    # Mock feature importance data
    features = [
        'Recent Team Form', 
        'Head-to-Head Record', 
        'Home Advantage', 
        'Toss Result', 
        'Player Availability',
        'Venue Statistics',
        'Batting Average',
        'Bowling Economy',
        'Previous Match Performance',
        'Weather Conditions'
    ]
    
    importance = [
        0.18, 0.15, 0.12, 0.11, 0.10,
        0.09, 0.08, 0.07, 0.06, 0.04
    ]
    
    # Create feature importance dataframe
    feature_data = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    # Sort by importance
    feature_data = feature_data.sort_values('Importance', ascending=False)
    
    # Visualization of feature importance
    st.subheader("Feature Importance in Prediction Model")
    
    fig = px.bar(
        feature_data,
        y='Feature',
        x='Importance',
        orientation='h',
        color='Importance',
        labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
        color_continuous_scale=['#ffa726', '#2e7d32']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    st.header("Model Performance")
    
    st.markdown("""
    The prediction model is evaluated using various metrics to ensure reliable forecasts:
    """)
    
    # Mock performance metrics
    metrics = [
        'Accuracy', 
        'Precision', 
        'Recall', 
        'F1 Score',
        'Log Loss'
    ]
    
    values = [0.74, 0.72, 0.76, 0.74, 0.58]
    
    # Create performance metrics dataframe
    performance_data = pd.DataFrame({
        'Metric': metrics,
        'Value': values
    })
    
    # Display performance metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualization of performance metrics
        fig = px.bar(
            performance_data,
            x='Metric',
            y='Value',
            color='Value',
            labels={'Value': 'Score', 'Metric': 'Performance Metric'},
            color_continuous_scale=['#ffa726', '#2e7d32']
        )
        
        fig.update_layout(yaxis=dict(range=[0, 1]))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(performance_data)
    
    # Historical prediction accuracy
    st.subheader("Historical Prediction Accuracy")
    
    # Generate mock historical accuracy data
    seasons = [f"IPL {year}" for year in range(2018, 2024)]
    prediction_accuracy = [round(random.uniform(0.65, 0.80), 2) for _ in range(len(seasons))]
    
    # Create historical accuracy dataframe
    history_data = pd.DataFrame({
        'Season': seasons,
        'Accuracy': prediction_accuracy
    })
    
    # Visualization of historical accuracy
    fig = px.line(
        history_data,
        x='Season',
        y='Accuracy',
        markers=True,
        labels={'Accuracy': 'Prediction Accuracy', 'Season': 'IPL Season'}
    )
    
    fig.update_layout(yaxis=dict(range=[0.6, 0.85]))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # LLM integration
    st.header("LLM Reasoning Integration")
    
    st.markdown("""
    Our system integrates Large Language Models (LLMs) to provide natural language explanations 
    for predictions. This integration enhances the interpretability of our statistical models by:
    
    1. **Contextualizing Predictions**: Explaining statistical outputs in cricket terminology
    2. **Highlighting Key Factors**: Identifying the most important elements influencing the prediction
    3. **Providing Nuanced Analysis**: Capturing aspects that purely statistical models might miss
    4. **Generating Narratives**: Creating comprehensive match previews based on data
    
    The LLM integration uses carefully designed prompts to extract sport-specific reasoning from the
    model while maintaining factual accuracy.
    """)
    
    # Prompt engineering examples
    st.subheader("Prompt Engineering Examples")
    
    st.code("""
# Match Prediction Explanation Prompt
def generate_match_explanation_prompt(team1, team2, winner, win_prob, team1_score, team2_score, 
                                      venue, toss_winner, toss_decision, pitch_conditions, weather_conditions):
    prompt = f'''
    You are a cricket analytics expert specializing in IPL predictions.
    
    Based on the following match information:
    - Team 1: {team1}
    - Team 2: {team2}
    - Venue: {venue}
    - Toss Winner: {toss_winner} (chose to {toss_decision})
    - Pitch Conditions: {pitch_conditions}
    - Weather Conditions: {weather_conditions}
    
    Our statistical model has predicted:
    - Match Winner: {winner} (Win Probability: {win_prob:.2%})
    - Predicted Score for {team1}: {team1_score}
    - Predicted Score for {team2}: {team2_score}
    
    Provide a concise, expert explanation of why {winner} is predicted to win this match.
    Focus on key factors like recent team form, head-to-head record, venue statistics,
    impact of toss, and how pitch/weather conditions might influence the outcome.
    
    Keep your explanation factual, balanced, and limited to 4-5 key points in bullet form.
    '''
    return prompt
    """, language="python")
    
    # Model training and updating
    st.header("Model Training and Updating")
    
    st.markdown("""
    Our prediction model follows a rigorous training and updating process:
    
    1. **Initial Training**: The model is trained on historical IPL data spanning multiple seasons
    2. **Cross-Validation**: We use k-fold cross-validation to ensure robust performance
    3. **Hyperparameter Tuning**: Model parameters are optimized for cricket prediction tasks
    4. **Regular Updates**: The model is updated after each match to incorporate new data
    5. **Seasonal Retraining**: Complete retraining is performed before each IPL season
    
    This approach ensures that the model remains accurate and captures the latest team dynamics,
    player form, and other relevant factors.
    """)
    
    # Training process visualization
    st.subheader("Model Training Process")
    
    # Create a simple flow chart of the training process
    st.markdown("""
    ```mermaid
    graph TD
        A[Historical Match Data] --> B[Data Preprocessing]
        B --> C[Feature Engineering]
        C --> D[Train Test Split]
        D --> E[Model Training]
        E --> F[Hyperparameter Tuning]
        F --> G[Model Evaluation]
        G --> H[Ensemble Creation]
        H --> I[Deploy Model]
        J[New Match Data] --> K[Update Features]
        K --> L[Incremental Update]
        L --> I
    ```
    """)
    
    # Limitations and future improvements
    st.header("Limitations and Future Improvements")
    
    st.markdown("""
    ### Current Limitations
    
    - **Player Injuries**: Limited real-time data on player fitness and injuries
    - **New Players**: Difficulty in accurately predicting performance of players with limited IPL history
    - **External Factors**: Challenging to account for all external factors like dew, exact pitch behavior
    - **Strategic Changes**: Teams may make unexpected strategic decisions during matches
    
    ### Planned Improvements
    
    - **Real-time Updating**: Incorporate live match data for in-game prediction adjustments
    - **Enhanced Player Modeling**: More sophisticated models for individual player contributions
    - **Video Analysis Integration**: Use computer vision to analyze pitch conditions and player technique
    - **Advanced Time Series Modeling**: Better capture team momentum and form trends
    - **Expanded LLM Capabilities**: More nuanced explanations and interactive Q&A about predictions
    """)
    
    # Disclaimer
    st.info("""
    **Disclaimer**: All prediction models have inherent uncertainty. Cricket is a dynamic sport with many
    variables, and actual match outcomes may differ from predictions. Our model provides probabilistic
    forecasts based on historical data and statistical patterns, not guarantees of match results.
    """)
