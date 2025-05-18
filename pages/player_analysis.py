import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta

from config import CRICKET_ACTION_IMAGES

def show():
    """
    Page for player performance analysis and predictions
    """
    st.title("Player Performance Analysis")
    
    # Display a cricket action image
    st.image(random.choice(CRICKET_ACTION_IMAGES), use_column_width=True)
    
    # Player selection
    st.header("Player Selection")
    
    # Sample player data (in a real system, this would come from a database)
    player_list = [
        "Virat Kohli", "Rohit Sharma", "MS Dhoni", "Jasprit Bumrah", "KL Rahul",
        "Hardik Pandya", "Ravindra Jadeja", "Rishabh Pant", "Shreyas Iyer", 
        "Pat Cummins", "Jos Buttler", "Kagiso Rabada", "Suryakumar Yadav",
        "Yuzvendra Chahal", "Rashid Khan", "David Warner", "Andre Russell"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_player = st.selectbox("Select Player", player_list)
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type", 
            ["Batting Performance", "Bowling Performance", "Overall Performance", "Form Analysis"]
        )
    
    # Player profile
    st.header(f"{selected_player} - Player Profile")
    
    # Mock player data (in a real system, would be fetched from a database)
    player_team = random.choice([
        "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
        "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad"
    ])
    
    player_role = random.choice([
        "Batsman", "Bowler", "All-rounder", "Wicket-keeper Batsman"
    ])
    
    # Display player info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Team**: {player_team}")
        st.markdown(f"**Role**: {player_role}")
    
    with col2:
        batting_avg = round(random.uniform(20, 55), 2)
        strike_rate = round(random.uniform(120, 170), 2)
        st.markdown(f"**Batting Average**: {batting_avg}")
        st.markdown(f"**Strike Rate**: {strike_rate}")
    
    with col3:
        if player_role in ["Bowler", "All-rounder"]:
            bowling_avg = round(random.uniform(15, 35), 2)
            economy = round(random.uniform(6, 10), 2)
            st.markdown(f"**Bowling Average**: {bowling_avg}")
            st.markdown(f"**Economy Rate**: {economy}")
        else:
            matches = random.randint(50, 150)
            fifties = random.randint(10, 30)
            st.markdown(f"**Matches**: {matches}")
            st.markdown(f"**Fifties**: {fifties}")
    
    # Performance analysis based on selected type
    st.header(f"{analysis_type}")
    
    if analysis_type == "Batting Performance":
        # Generate mock batting data for the last 10 matches
        matches = 10
        dates = [(datetime.now() - timedelta(days=i*3)).strftime('%Y-%m-%d') for i in range(matches)]
        dates.reverse()
        
        opponents = random.choices([
            "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
            "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
            "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"
        ], k=matches)
        
        runs = [random.randint(0, 100) for _ in range(matches)]
        balls = [random.randint(max(1, runs[i]//2), max(1, runs[i] + 20)) for i in range(matches)]
        fours = [random.randint(0, runs[i]//6) for i in range(matches)]
        sixes = [random.randint(0, runs[i]//12) for i in range(matches)]
        
        # Create a dataframe
        batting_data = pd.DataFrame({
            'Date': dates,
            'Opponent': opponents,
            'Runs': runs,
            'Balls': balls,
            'Fours': fours,
            'Sixes': sixes,
            'Strike Rate': [round(runs[i]/balls[i]*100, 2) if balls[i] > 0 else 0 for i in range(matches)]
        })
        
        # Display recent batting performance
        st.subheader("Recent Batting Performance")
        st.dataframe(batting_data, use_container_width=True)
        
        # Runs visualization
        st.subheader("Runs in Last 10 Matches")
        fig = px.bar(
            batting_data,
            x='Date',
            y='Runs',
            color='Runs',
            labels={'Runs': 'Runs Scored', 'Date': 'Match Date'},
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strike rate visualization
        st.subheader("Strike Rate in Last 10 Matches")
        fig = px.line(
            batting_data,
            x='Date',
            y='Strike Rate',
            markers=True,
            labels={'Strike Rate': 'Strike Rate', 'Date': 'Match Date'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Bowling Performance":
        # Generate mock bowling data for the last 10 matches
        matches = 10
        dates = [(datetime.now() - timedelta(days=i*3)).strftime('%Y-%m-%d') for i in range(matches)]
        dates.reverse()
        
        opponents = random.choices([
            "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
            "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
            "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"
        ], k=matches)
        
        overs = [random.randint(2, 4) for _ in range(matches)]
        maidens = [random.randint(0, 1) for _ in range(matches)]
        runs_conceded = [random.randint(15, 50) for _ in range(matches)]
        wickets = [random.randint(0, 4) for _ in range(matches)]
        
        # Create a dataframe
        bowling_data = pd.DataFrame({
            'Date': dates,
            'Opponent': opponents,
            'Overs': overs,
            'Maidens': maidens,
            'Runs': runs_conceded,
            'Wickets': wickets,
            'Economy': [round(runs_conceded[i]/overs[i], 2) for i in range(matches)]
        })
        
        # Display recent bowling performance
        st.subheader("Recent Bowling Performance")
        st.dataframe(bowling_data, use_container_width=True)
        
        # Wickets visualization
        st.subheader("Wickets in Last 10 Matches")
        fig = px.bar(
            bowling_data,
            x='Date',
            y='Wickets',
            color='Wickets',
            labels={'Wickets': 'Wickets Taken', 'Date': 'Match Date'},
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Economy rate visualization
        st.subheader("Economy Rate in Last 10 Matches")
        fig = px.line(
            bowling_data,
            x='Date',
            y='Economy',
            markers=True,
            labels={'Economy': 'Economy Rate', 'Date': 'Match Date'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Overall Performance":
        # Create performance metrics for different IPL seasons
        seasons = [f"IPL {year}" for year in range(2018, 2024)]
        
        matches_played = [random.randint(10, 16) for _ in range(len(seasons))]
        
        if player_role in ["Batsman", "All-rounder", "Wicket-keeper Batsman"]:
            runs_scored = [random.randint(200, 600) for _ in range(len(seasons))]
            avg_sr = [random.uniform(120, 160) for _ in range(len(seasons))]
            
            # Create dataframe
            performance_data = pd.DataFrame({
                'Season': seasons,
                'Matches': matches_played,
                'Runs': runs_scored,
                'Average': [round(runs_scored[i]/(random.randint(5, 12)), 2) for i in range(len(seasons))],
                'Strike Rate': [round(avg_sr[i], 2) for i in range(len(seasons))]
            })
            
            # Display overall performance
            st.subheader("Season-wise Batting Performance")
            st.dataframe(performance_data, use_container_width=True)
            
            # Runs by season
            st.subheader("Runs by Season")
            fig = px.bar(
                performance_data,
                x='Season',
                y='Runs',
                color='Season',
                labels={'Runs': 'Runs Scored', 'Season': 'IPL Season'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Average and strike rate
            st.subheader("Batting Average and Strike Rate")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=performance_data['Average'],
                mode='lines+markers',
                name='Batting Average'
            ))
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=performance_data['Strike Rate'],
                mode='lines+markers',
                name='Strike Rate',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Batting Average and Strike Rate by Season',
                xaxis_title='Season',
                yaxis_title='Batting Average',
                yaxis2=dict(
                    title='Strike Rate',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        if player_role in ["Bowler", "All-rounder"]:
            wickets_taken = [random.randint(5, 25) for _ in range(len(seasons))]
            economy_rates = [random.uniform(6, 10) for _ in range(len(seasons))]
            
            # Create dataframe
            bowling_perf_data = pd.DataFrame({
                'Season': seasons,
                'Matches': matches_played,
                'Wickets': wickets_taken,
                'Economy': [round(er, 2) for er in economy_rates],
                'Average': [round(random.uniform(15, 35), 2) for _ in range(len(seasons))]
            })
            
            # Display overall bowling performance
            st.subheader("Season-wise Bowling Performance")
            st.dataframe(bowling_perf_data, use_container_width=True)
            
            # Wickets by season
            st.subheader("Wickets by Season")
            fig = px.bar(
                bowling_perf_data,
                x='Season',
                y='Wickets',
                color='Season',
                labels={'Wickets': 'Wickets Taken', 'Season': 'IPL Season'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Economy and average
            st.subheader("Bowling Economy and Average")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=bowling_perf_data['Economy'],
                mode='lines+markers',
                name='Economy Rate'
            ))
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=bowling_perf_data['Average'],
                mode='lines+markers',
                name='Bowling Average',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Bowling Economy and Average by Season',
                xaxis_title='Season',
                yaxis_title='Economy Rate',
                yaxis2=dict(
                    title='Bowling Average',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    elif analysis_type == "Form Analysis":
        # Recent form analysis - last 5 matches
        st.subheader("Recent Form Analysis (Last 5 Matches)")
        
        # Generate form data
        if player_role in ["Batsman", "All-rounder", "Wicket-keeper Batsman"]:
            dates = [(datetime.now() - timedelta(days=i*3)).strftime('%Y-%m-%d') for i in range(5)]
            dates.reverse()
            
            form_data = pd.DataFrame({
                'Date': dates,
                'Runs': [random.randint(0, 80) for _ in range(5)],
                'Strike Rate': [round(random.uniform(100, 180), 2) for _ in range(5)]
            })
            
            # Display form data
            st.subheader("Batting Form")
            
            # Visualization of recent form
            fig = px.line(
                form_data,
                x='Date',
                y=['Runs', 'Strike Rate'],
                markers=True,
                labels={'value': 'Value', 'Date': 'Match Date', 'variable': 'Metric'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Form metrics
            avg_runs = round(np.mean(form_data['Runs']), 2)
            avg_sr = round(np.mean(form_data['Strike Rate']), 2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Runs (Last 5 Matches)", avg_runs)
            
            with col2:
                st.metric("Average Strike Rate (Last 5 Matches)", avg_sr)
                
            # Form assessment
            if avg_runs > 40 and avg_sr > 140:
                form_assessment = "Excellent Form"
            elif avg_runs > 30 and avg_sr > 130:
                form_assessment = "Good Form"
            elif avg_runs > 20 and avg_sr > 120:
                form_assessment = "Average Form"
            else:
                form_assessment = "Below Average Form"
                
            st.success(f"Form Assessment: {form_assessment}")
        
        if player_role in ["Bowler", "All-rounder"]:
            dates = [(datetime.now() - timedelta(days=i*3)).strftime('%Y-%m-%d') for i in range(5)]
            dates.reverse()
            
            bowling_form_data = pd.DataFrame({
                'Date': dates,
                'Wickets': [random.randint(0, 3) for _ in range(5)],
                'Economy': [round(random.uniform(6, 12), 2) for _ in range(5)]
            })
            
            # Display bowling form data
            st.subheader("Bowling Form")
            
            # Visualization of recent bowling form
            fig = px.line(
                bowling_form_data,
                x='Date',
                y=['Wickets', 'Economy'],
                markers=True,
                labels={'value': 'Value', 'Date': 'Match Date', 'variable': 'Metric'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Form metrics
            avg_wickets = round(np.mean(bowling_form_data['Wickets']), 2)
            avg_economy = round(np.mean(bowling_form_data['Economy']), 2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Wickets (Last 5 Matches)", avg_wickets)
            
            with col2:
                st.metric("Average Economy (Last 5 Matches)", avg_economy)
                
            # Bowling form assessment
            if avg_wickets > 2 and avg_economy < 7.5:
                form_assessment = "Excellent Form"
            elif avg_wickets > 1.5 and avg_economy < 8.5:
                form_assessment = "Good Form"
            elif avg_wickets > 1 and avg_economy < 9.5:
                form_assessment = "Average Form"
            else:
                form_assessment = "Below Average Form"
                
            st.success(f"Bowling Form Assessment: {form_assessment}")
    
    # Prediction for next match
    st.header("Performance Prediction for Next Match")
    
    next_opponent = st.selectbox(
        "Select Opponent for Next Match",
        [team for team in [
            "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
            "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
            "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"
        ] if team != player_team]
    )
    
    next_venue = st.selectbox(
        "Select Venue",
        [
            "Eden Gardens, Kolkata",
            "Wankhede Stadium, Mumbai",
            "M. Chinnaswamy Stadium, Bangalore",
            "MA Chidambaram Stadium, Chennai",
            "Arun Jaitley Stadium, Delhi",
            "Rajiv Gandhi International Stadium, Hyderabad",
            "Punjab Cricket Association Stadium, Mohali",
            "Sawai Mansingh Stadium, Jaipur",
            "Narendra Modi Stadium, Ahmedabad"
        ]
    )
    
    if st.button("Predict Performance"):
        with st.spinner("Generating performance prediction..."):
            # Mock prediction logic (in a real system, this would use the ML model)
            if player_role in ["Batsman", "All-rounder", "Wicket-keeper Batsman"]:
                predicted_runs = random.randint(20, 70)
                predicted_sr = round(random.uniform(120, 170), 2)
                
                st.subheader("Batting Performance Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Runs", predicted_runs)
                
                with col2:
                    st.metric("Predicted Strike Rate", predicted_sr)
                    
                # Confidence interval
                runs_lower = max(0, predicted_runs - 15)
                runs_upper = predicted_runs + 15
                
                st.markdown(f"**Runs Range**: {runs_lower} - {runs_upper}")
                
                # Visualization
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_runs,
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#1e88e5"},
                        'steps': [
                            {'range': [0, 20], 'color': "#e57373"},
                            {'range': [20, 40], 'color': "#ffb74d"},
                            {'range': [40, 60], 'color': "#aed581"},
                            {'range': [60, 100], 'color': "#4caf50"}
                        ]
                    },
                    title = {'text': "Predicted Runs"}
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            if player_role in ["Bowler", "All-rounder"]:
                predicted_wickets = random.randint(0, 3)
                predicted_economy = round(random.uniform(7, 11), 2)
                
                st.subheader("Bowling Performance Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Wickets", predicted_wickets)
                
                with col2:
                    st.metric("Predicted Economy", predicted_economy)
                    
                # Confidence interval
                economy_lower = max(5, predicted_economy - 1.5)
                economy_upper = predicted_economy + 1.5
                
                st.markdown(f"**Economy Range**: {economy_lower:.2f} - {economy_upper:.2f}")
                
                # Visualization
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_wickets,
                    gauge = {
                        'axis': {'range': [None, 5]},
                        'bar': {'color': "#1e88e5"},
                        'steps': [
                            {'range': [0, 1], 'color': "#e57373"},
                            {'range': [1, 2], 'color': "#ffb74d"},
                            {'range': [2, 3], 'color': "#aed581"},
                            {'range': [3, 5], 'color': "#4caf50"}
                        ]
                    },
                    title = {'text': "Predicted Wickets"}
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Performance factors
            st.subheader("Key Performance Factors")
            
            factors = [
                f"Past performance against {next_opponent}",
                f"Recent form in the last 5 matches",
                f"Performance at {next_venue}",
                "Team matchups and opposition bowling/batting strength",
                "Historical performance in similar pitch conditions"
            ]
            
            for factor in factors:
                st.markdown(f"- {factor}")
