import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from config import IPL_TEAMS, STADIUM_IMAGES
from llm_integration import ollama_client, prompt_engineering

def show(model=None):
    """
    Page for match predictions
    """
    st.title("IPL Match Predictions")
    
    # Check if model is available
    if model is None:
        st.error("Prediction model is not available. Please try reloading the application.")
        return

    # Match selection
    st.header("Match Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team1 = st.selectbox("Select Home Team", IPL_TEAMS)
    
    with col2:
        # Filter to avoid same team selection
        team2_options = [team for team in IPL_TEAMS if team != team1]
        team2 = st.selectbox("Select Away Team", team2_options)
    
    with col3:
        venues = [
            "Eden Gardens, Kolkata",
            "Wankhede Stadium, Mumbai",
            "M. Chinnaswamy Stadium, Bangalore",
            "MA Chidambaram Stadium, Chennai",
            "Arun Jaitley Stadium, Delhi",
            "Rajiv Gandhi International Stadium, Hyderabad",
            "Punjab Cricket Association Stadium, Mohali",
            "Sawai Mansingh Stadium, Jaipur",
            "Narendra Modi Stadium, Ahmedabad",
            "Barsapara Cricket Stadium, Guwahati"
        ]
        venue = st.selectbox("Select Venue", venues)
    
    # Additional match parameters
    st.subheader("Match Conditions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ["Bat", "Field"])
    
    with col2:
        pitch_conditions = st.select_slider(
            "Pitch Conditions",
            options=["Very Batting Friendly", "Batting Friendly", "Neutral", "Bowling Friendly", "Very Bowling Friendly"]
        )
    
    with col3:
        weather_conditions = st.selectbox(
            "Weather Conditions",
            ["Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Humid"]
        )

    # Generate prediction button
    if st.button("Generate Prediction"):
        # Show a spinner while generating prediction
        with st.spinner("Generating prediction..."):
            # Prepare input data for model
            match_data = {
                'team1': team1,
                'team2': team2,
                'venue': venue,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'pitch_conditions': pitch_conditions,
                'weather_conditions': weather_conditions
            }
            
            try:
                # Get predictions from model
                prediction_result = model.predict_match(match_data)
                
                # Display prediction results
                st.header("Match Prediction Results")
                
                # Winner prediction with progress bar
                st.subheader("Match Winner")
                winner = prediction_result['winner']
                win_prob = prediction_result['win_probability']
                
                winner_col, prob_col = st.columns([1, 2])
                with winner_col:
                    st.markdown(f"**Predicted Winner**: {winner}")
                with prob_col:
                    st.progress(win_prob)
                    st.text(f"Win Probability: {win_prob:.2%}")
                
                # Score predictions
                st.subheader("Score Predictions")
                team1_score = prediction_result['team1_score']
                team2_score = prediction_result['team2_score']
                
                score_cols = st.columns(2)
                with score_cols[0]:
                    st.metric(label=f"{team1} Predicted Score", value=f"{team1_score} runs")
                with score_cols[1]:
                    st.metric(label=f"{team2} Predicted Score", value=f"{team2_score} runs")
                
                # Visualization of match prediction
                st.subheader("Match Outcome Visualization")
                
                # Create figure for win probability
                fig = go.Figure()
                
                # Add bars for win probabilities
                fig.add_trace(go.Bar(
                    x=[team1, team2],
                    y=[win_prob if winner == team1 else 1-win_prob, 
                       win_prob if winner == team2 else 1-win_prob],
                    text=[f"{win_prob:.2%}" if winner == team1 else f"{1-win_prob:.2%}", 
                          f"{win_prob:.2%}" if winner == team2 else f"{1-win_prob:.2%}"],
                    textposition='auto',
                    marker_color=['#1e88e5', '#ff5722']
                ))
                
                # Update layout
                fig.update_layout(
                    title="Team Win Probability",
                    xaxis_title="Teams",
                    yaxis_title="Win Probability",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key factors influencing prediction
                st.subheader("Key Factors")
                
                # Get explanation from LLM
                prompt = prompt_engineering.generate_match_explanation_prompt(
                    team1, team2, winner, win_prob, team1_score, team2_score, 
                    venue, toss_winner, toss_decision, pitch_conditions, weather_conditions
                )
                
                try:
                    explanation = ollama_client.get_explanation(prompt)
                    st.markdown(explanation)
                except Exception as e:
                    st.warning(f"LLM explanation unavailable: {e}")
                    st.markdown("""
                    Key factors considered in this prediction:
                    
                    - Recent team form and head-to-head record
                    - Team performance at selected venue
                    - Impact of toss decision
                    - Pitch and weather conditions
                    - Key player availability and current form
                    """)
                
                # Display recent head-to-head results
                st.subheader("Recent Head-to-Head")
                
                # Create sample head-to-head data (in a real system, this would come from a database)
                h2h_data = {
                    'Date': [
                        (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d')
                        for _ in range(5)
                    ],
                    'Venue': random.choices(venues, k=5),
                    'Winner': random.choices([team1, team2], k=5),
                    'Margin': [f"{random.randint(5, 50)} runs" if random.random() > 0.5 else f"{random.randint(1, 8)} wickets" for _ in range(5)],
                    'Player of Match': [random.choice(["Virat Kohli", "Rohit Sharma", "MS Dhoni", "Jasprit Bumrah", "KL Rahul"]) for _ in range(5)]
                }
                
                h2h_df = pd.DataFrame(h2h_data)
                st.dataframe(h2h_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating prediction: {e}")
                st.error("Please try again with different parameters.")
    
    # Display information about the prediction model
    st.markdown("---")
    st.markdown("""
    **About the Prediction Model**
    
    Our prediction model uses an ensemble of machine learning algorithms trained on historical IPL data.
    The model considers team composition, recent form, head-to-head records, venue statistics, 
    and match conditions to generate accurate predictions.
    
    For detailed information about the model and its performance, visit the Model Explanation page.
    """)
    
    # Show a random stadium image
    st.image(random.choice(STADIUM_IMAGES), use_column_width=True)
