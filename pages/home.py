import streamlit as st
import random
from config import IPL_TEAMS, STADIUM_IMAGES, CRICKET_ACTION_IMAGES, VISUALIZATION_IMAGES

def show():
    """
    Home page of the IPL Prediction System
    """
    # Header with random cricket action image
    header_image = random.choice(CRICKET_ACTION_IMAGES)
    st.image(header_image, use_column_width=True)
    
    # Title and introduction
    st.title("IPL Cricket Prediction System")
    st.markdown("""
    Welcome to the IPL Cricket Prediction System - a comprehensive analytics platform that leverages 
    machine learning to forecast match outcomes, team scores, and player performance metrics for 
    the Indian Premier League.
    """)
    
    # Feature overview
    st.header("Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Match Predictions")
        st.markdown("""
        - Predict match winners
        - Forecast winning & losing team scores
        - Analyze team head-to-head statistics
        - Account for venue and weather conditions
        """)
        st.image(STADIUM_IMAGES[1], use_column_width=True)
    
    with col2:
        st.subheader("Player Analysis")
        st.markdown("""
        - Predict individual player performance
        - Track form over recent matches
        - Compare players across teams
        - Identify key performers for upcoming matches
        """)
        st.image(CRICKET_ACTION_IMAGES[2], use_column_width=True)
    
    with col3:
        st.subheader("Advanced Analytics")
        st.markdown("""
        - LLM-powered reasoning for predictions
        - Visualize prediction confidence levels
        - Understand key factors influencing outcomes
        - Track prediction accuracy over time
        """)
        st.image(VISUALIZATION_IMAGES[0], use_column_width=True)
    
    # How it works section
    st.header("How It Works")
    st.markdown("""
    Our system combines statistical machine learning models with advanced language model reasoning:
    
    1. **Data Collection**: We gather comprehensive statistics on teams, players, venues, and match conditions
    2. **Feature Engineering**: Raw data is transformed into meaningful predictive features
    3. **Ensemble Modeling**: Multiple algorithms work together to generate accurate predictions
    4. **LLM Reasoning**: Language models provide contextual explanations for statistical predictions
    5. **Continuous Learning**: The system improves over time as new match data becomes available
    """)
    
    # Teams section
    st.header("IPL Teams")
    team_cols = st.columns(5)
    
    for i, team in enumerate(IPL_TEAMS):
        with team_cols[i % 5]:
            st.markdown(f"**{team}**")
    
    # Quick start
    st.header("Quick Start")
    st.markdown("""
    Use the navigation panel on the left to explore different features:
    
    - **Match Predictions**: Forecast upcoming match outcomes
    - **Player Analysis**: Analyze individual player performance
    - **Team Analysis**: Compare team statistics and performance trends
    - **Model Explanation**: Understand how predictions are generated
    """)
    
    # Disclaimer
    st.info("""
    **Disclaimer**: This system provides predictions based on historical data and statistical models.
    While we strive for accuracy, cricket is inherently unpredictable, and actual results may vary.
    """)
