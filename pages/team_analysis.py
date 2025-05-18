import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

from config import IPL_TEAMS, STADIUM_IMAGES, VISUALIZATION_IMAGES

def show():
    """
    Page for team analysis and performance trends
    """
    st.title("IPL Team Analysis")
    
    # Display a visualization image
    st.image(random.choice(VISUALIZATION_IMAGES), use_container_width=True)
    
    # Team selection
    st.header("Team Selection")
    selected_team = st.selectbox("Select Team", IPL_TEAMS)
    
    # Team statistics overview
    st.header(f"{selected_team} - Team Overview")
    
    # Mock team data (in a real system, would be fetched from a database)
    # Generate random statistics for the team
    matches_played = random.randint(180, 250)
    matches_won = random.randint(90, 150)
    matches_lost = matches_played - matches_won
    win_percentage = round((matches_won / matches_played) * 100, 2)
    
    titles_won = random.randint(0, 5)
    highest_score = random.randint(220, 260)
    lowest_score = random.randint(60, 100)
    
    # Display team statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Matches Played", matches_played)
        st.metric("Titles Won", titles_won)
    
    with col2:
        st.metric("Matches Won", matches_won)
        st.metric("Highest Score", highest_score)
    
    with col3:
        st.metric("Win Percentage", f"{win_percentage}%")
        st.metric("Lowest Score", lowest_score)
    
    # Team performance by season
    st.header("Season Performance")
    
    # Generate mock season data
    seasons = [f"IPL {year}" for year in range(2016, 2024)]
    season_position = [random.randint(1, 10) for _ in range(len(seasons))]
    matches_per_season = [random.randint(14, 16) for _ in range(len(seasons))]
    wins_per_season = [random.randint(4, 12) for _ in range(len(seasons))]
    losses_per_season = [matches_per_season[i] - wins_per_season[i] for i in range(len(seasons))]
    nrr_per_season = [round(random.uniform(-1.5, 1.5), 2) for _ in range(len(seasons))]
    
    # Create dataframe
    season_data = pd.DataFrame({
        'Season': seasons,
        'Position': season_position,
        'Matches': matches_per_season,
        'Wins': wins_per_season,
        'Losses': losses_per_season,
        'NRR': nrr_per_season
    })
    
    # Display season data
    st.subheader("Season-wise Performance")
    st.dataframe(season_data, use_container_width=True)
    
    # Visualization of season performance
    st.subheader("Position by Season")
    
    # Create figure for team position (lower is better, so invert the y-axis)
    fig = px.line(
        season_data,
        x='Season',
        y='Position',
        markers=True,
        labels={'Position': 'Final Position', 'Season': 'IPL Season'}
    )
    
    # Invert y-axis so that 1st position is at the top
    fig.update_layout(yaxis=dict(autorange="reversed"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Wins and losses by season
    st.subheader("Wins and Losses by Season")
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=seasons,
        y=season_data['Wins'],
        name='Wins',
        marker_color='#2e7d32'
    ))
    
    fig.add_trace(go.Bar(
        x=seasons,
        y=season_data['Losses'],
        name='Losses',
        marker_color='#c62828'
    ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_title='Season',
        yaxis_title='Number of Matches'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Net Run Rate by season
    st.subheader("Net Run Rate (NRR) by Season")
    
    # Create bar chart with colorized bars based on NRR value
    fig = px.bar(
        season_data,
        x='Season',
        y='NRR',
        color='NRR',
        labels={'NRR': 'Net Run Rate', 'Season': 'IPL Season'},
        color_continuous_scale=['#c62828', '#ffa726', '#2e7d32']
    )
    
    # Add a reference line at y=0
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Venue analysis
    st.header("Venue Analysis")
    
    # Generate mock venue data
    venues = [
        "Eden Gardens", "Wankhede Stadium", "M. Chinnaswamy Stadium",
        "MA Chidambaram Stadium", "Arun Jaitley Stadium", 
        "Rajiv Gandhi Stadium", "Punjab Cricket Association Stadium"
    ]
    
    matches_at_venue = [random.randint(10, 30) for _ in range(len(venues))]
    wins_at_venue = [random.randint(3, matches_at_venue[i]) for i in range(len(venues))]
    win_pct_at_venue = [round((wins_at_venue[i] / matches_at_venue[i]) * 100, 2) for i in range(len(venues))]
    
    # Create dataframe
    venue_data = pd.DataFrame({
        'Venue': venues,
        'Matches': matches_at_venue,
        'Wins': wins_at_venue,
        'Win %': win_pct_at_venue
    })
    
    # Display venue data
    st.subheader("Performance by Venue")
    st.dataframe(venue_data, use_container_width=True)
    
    # Visualization of venue performance
    st.subheader("Win Percentage by Venue")
    
    # Create horizontal bar chart
    fig = px.bar(
        venue_data,
        y='Venue',
        x='Win %',
        orientation='h',
        color='Win %',
        labels={'Win %': 'Win Percentage', 'Venue': 'Stadium'},
        color_continuous_scale=['#c62828', '#ffa726', '#2e7d32']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Head-to-head analysis
    st.header("Head-to-Head Analysis")
    
    # Create mock head-to-head data
    opponents = [team for team in IPL_TEAMS if team != selected_team]
    matches_played_h2h = [random.randint(15, 30) for _ in range(len(opponents))]
    wins_h2h = [random.randint(5, matches_played_h2h[i]) for i in range(len(opponents))]
    losses_h2h = [matches_played_h2h[i] - wins_h2h[i] for i in range(len(opponents))]
    win_pct_h2h = [round((wins_h2h[i] / matches_played_h2h[i]) * 100, 2) for i in range(len(opponents))]
    
    # Create dataframe
    h2h_data = pd.DataFrame({
        'Opponent': opponents,
        'Matches': matches_played_h2h,
        'Wins': wins_h2h,
        'Losses': losses_h2h,
        'Win %': win_pct_h2h
    })
    
    # Display head-to-head data
    st.subheader(f"{selected_team} vs Other Teams")
    st.dataframe(h2h_data, use_container_width=True)
    
    # Visualization of head-to-head data
    st.subheader("Head-to-Head Win Percentage")
    
    # Create horizontal bar chart
    fig = px.bar(
        h2h_data,
        y='Opponent',
        x='Win %',
        orientation='h',
        color='Win %',
        labels={'Win %': 'Win Percentage', 'Opponent': 'Team'},
        color_continuous_scale=['#c62828', '#ffa726', '#2e7d32']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Batting and bowling analysis
    st.header("Team Batting and Bowling Analysis")
    
    # Generate mock batting and bowling data for recent seasons
    recent_seasons = [f"IPL {year}" for year in range(2020, 2024)]
    
    # Batting metrics
    avg_score = [random.randint(150, 190) for _ in range(len(recent_seasons))]
    run_rate = [round(random.uniform(7.5, 9.5), 2) for _ in range(len(recent_seasons))]
    
    # Bowling metrics
    avg_conceded = [random.randint(150, 190) for _ in range(len(recent_seasons))]
    economy_rate = [round(random.uniform(7.5, 9.5), 2) for _ in range(len(recent_seasons))]
    
    # Create dataframe
    team_perf_data = pd.DataFrame({
        'Season': recent_seasons,
        'Avg Score': avg_score,
        'Run Rate': run_rate,
        'Avg Conceded': avg_conceded,
        'Economy Rate': economy_rate
    })
    
    # Display team performance data
    st.subheader("Batting and Bowling Metrics by Season")
    st.dataframe(team_perf_data, use_container_width=True)
    
    # Visualization
    st.subheader("Batting Performance")
    
    # Create a dual-axis chart for batting metrics
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=recent_seasons,
        y=team_perf_data['Avg Score'],
        name='Average Score',
        marker_color='#1e88e5'
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_seasons,
        y=team_perf_data['Run Rate'],
        mode='lines+markers',
        name='Run Rate',
        yaxis='y2',
        marker_color='#f57c00'
    ))
    
    fig.update_layout(
        xaxis_title='Season',
        yaxis_title='Average Score',
        yaxis2=dict(
            title='Run Rate',
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
    
    st.subheader("Bowling Performance")
    
    # Create a dual-axis chart for bowling metrics
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=recent_seasons,
        y=team_perf_data['Avg Conceded'],
        name='Average Conceded',
        marker_color='#d32f2f'
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_seasons,
        y=team_perf_data['Economy Rate'],
        mode='lines+markers',
        name='Economy Rate',
        yaxis='y2',
        marker_color='#7b1fa2'
    ))
    
    fig.update_layout(
        xaxis_title='Season',
        yaxis_title='Average Conceded',
        yaxis2=dict(
            title='Economy Rate',
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
    
    # Key players
    st.header("Key Players")
    
    # Generate mock player data
    if selected_team == "Chennai Super Kings":
        key_players = ["MS Dhoni", "Ravindra Jadeja", "Devon Conway", "Ruturaj Gaikwad", "Deepak Chahar"]
    elif selected_team == "Mumbai Indians":
        key_players = ["Rohit Sharma", "Jasprit Bumrah", "Suryakumar Yadav", "Ishan Kishan", "Tim David"]
    elif selected_team == "Royal Challengers Bangalore":
        key_players = ["Virat Kohli", "Glenn Maxwell", "Faf du Plessis", "Mohammed Siraj", "Dinesh Karthik"]
    elif selected_team == "Kolkata Knight Riders":
        key_players = ["Shreyas Iyer", "Andre Russell", "Sunil Narine", "Venkatesh Iyer", "Varun Chakravarthy"]
    else:
        # Generic list for other teams
        key_players = [f"Player {i+1}" for i in range(5)]
    
    player_roles = ["Batsman", "All-rounder", "Bowler", "Wicket-keeper", "Batsman"]
    player_impact = [random.randint(70, 95) for _ in range(len(key_players))]
    
    # Create dataframe
    player_data = pd.DataFrame({
        'Player': key_players,
        'Role': player_roles,
        'Impact Score': player_impact
    })
    
    # Display key players
    st.subheader(f"{selected_team} Key Players")
    st.dataframe(player_data, use_container_width=True)
    
    # Visualization of player impact
    st.subheader("Player Impact Scores")
    
    # Create horizontal bar chart for player impact
    fig = px.bar(
        player_data,
        y='Player',
        x='Impact Score',
        orientation='h',
        color='Impact Score',
        labels={'Impact Score': 'Impact Rating (0-100)', 'Player': 'Player Name'},
        color_continuous_scale=['#ffa726', '#2e7d32']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display a random stadium image
    st.image(random.choice(STADIUM_IMAGES), use_column_width=True)
