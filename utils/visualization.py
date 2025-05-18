"""
Visualization utilities for IPL prediction system.
This module provides functions for generating plots and charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='visualization.log'
)
logger = logging.getLogger('visualization')

def plot_win_probability(team1, team2, win_probability):
    """
    Create a win probability chart for two teams.
    
    Args:
        team1 (str): Name of first team
        team2 (str): Name of second team
        win_probability (float): Probability of team1 winning (0-1)
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Ensure win_probability is between 0 and 1
        win_probability = max(0, min(1, win_probability))
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for win probabilities
        fig.add_trace(go.Bar(
            x=[team1, team2],
            y=[win_probability, 1-win_probability],
            text=[f"{win_probability:.1%}", f"{1-win_probability:.1%}"],
            textposition='auto',
            marker_color=['#1e88e5', '#ff5722']
        ))
        
        # Update layout
        fig.update_layout(
            title="Match Win Probability",
            xaxis_title="Teams",
            yaxis_title="Win Probability",
            yaxis=dict(range=[0, 1])
        )
        
        logger.info(f"Created win probability chart for {team1} vs {team2}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating win probability chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def plot_score_prediction(team1, team2, team1_score, team2_score):
    """
    Create a score prediction chart for two teams.
    
    Args:
        team1 (str): Name of first team
        team2 (str): Name of second team
        team1_score (int): Predicted score for team1
        team2_score (int): Predicted score for team2
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add bars for predicted scores
        fig.add_trace(go.Bar(
            x=[team1, team2],
            y=[team1_score, team2_score],
            text=[str(team1_score), str(team2_score)],
            textposition='auto',
            marker_color=['#1e88e5', '#ff5722']
        ))
        
        # Update layout
        fig.update_layout(
            title="Predicted Team Scores",
            xaxis_title="Teams",
            yaxis_title="Predicted Score (Runs)",
            yaxis=dict(range=[0, max(team1_score, team2_score) * 1.1])  # Add 10% headroom
        )
        
        logger.info(f"Created score prediction chart for {team1} vs {team2}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating score prediction chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def plot_player_performance_history(player_name, performance_data, metric_name='Runs'):
    """
    Create a chart showing a player's performance history.
    
    Args:
        player_name (str): Name of the player
        performance_data (pd.DataFrame): DataFrame with performance data
        metric_name (str): Name of metric to plot (Runs, Wickets, etc.)
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        if performance_data is None or len(performance_data) == 0:
            raise ValueError("No performance data provided")
        
        # Create figure
        fig = px.line(
            performance_data,
            x='Date',
            y=metric_name,
            markers=True,
            title=f"{player_name}'s {metric_name} History"
        )
        
        # Add average line
        avg_value = performance_data[metric_name].mean()
        fig.add_hline(
            y=avg_value,
            line_dash="dash",
            annotation_text=f"Avg: {avg_value:.1f}",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Match Date",
            yaxis_title=metric_name
        )
        
        logger.info(f"Created performance history chart for {player_name}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance history chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def plot_head_to_head(team1, team2, team1_wins, team2_wins, no_results=0):
    """
    Create a head-to-head comparison chart for two teams.
    
    Args:
        team1 (str): Name of first team
        team2 (str): Name of second team
        team1_wins (int): Number of wins for team1
        team2_wins (int): Number of wins for team2
        no_results (int): Number of matches with no result
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Calculate total matches and win percentages
        total_matches = team1_wins + team2_wins + no_results
        team1_pct = team1_wins / total_matches if total_matches > 0 else 0
        team2_pct = team2_wins / total_matches if total_matches > 0 else 0
        no_results_pct = no_results / total_matches if total_matches > 0 else 0
        
        # Create pie chart
        labels = [f"{team1} wins", f"{team2} wins", "No Results"]
        values = [team1_wins, team2_wins, no_results]
        colors = ['#1e88e5', '#ff5722', '#9e9e9e']
        
        fig = px.pie(
            names=labels,
            values=values,
            title=f"Head-to-Head: {team1} vs {team2}",
            color_discrete_sequence=colors,
            hover_data=[values]
        )
        
        # Add total matches annotation
        fig.add_annotation(
            text=f"Total Matches: {total_matches}",
            showarrow=False,
            font=dict(size=14),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1
        )
        
        # Update layout
        fig.update_layout(
            legend_title="Result",
            margin=dict(t=60, b=60, l=30, r=30)
        )
        
        logger.info(f"Created head-to-head chart for {team1} vs {team2}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating head-to-head chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def plot_team_form(team_name, matches_data):
    """
    Create a chart showing a team's recent form.
    
    Args:
        team_name (str): Name of the team
        matches_data (pd.DataFrame): DataFrame with recent match data
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        if matches_data is None or len(matches_data) == 0:
            raise ValueError("No matches data provided")
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for match results (assuming 1 for win, 0 for loss)
        # You would need to extract these values from your matches_data
        match_dates = matches_data['Date'].tolist()
        results = matches_data['Result'].tolist()
        
        # Convert results to numeric (1 for win, 0 for loss)
        numeric_results = []
        for result in results:
            if isinstance(result, str):
                numeric_results.append(1 if "won" in result.lower() and team_name.lower() in result.lower() else 0)
            else:
                numeric_results.append(result)
        
        # Create color map (green for wins, red for losses)
        colors = ['#4caf50' if r == 1 else '#f44336' for r in numeric_results]
        
        fig.add_trace(go.Bar(
            x=match_dates,
            y=numeric_results,
            marker_color=colors,
            text=["Win" if r == 1 else "Loss" for r in numeric_results],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{team_name}'s Recent Form",
            xaxis_title="Match Date",
            yaxis_title="Result (Win/Loss)",
            yaxis=dict(range=[0, 1], tickvals=[0, 1], ticktext=["Loss", "Win"])
        )
        
        # Calculate win percentage
        win_pct = sum(numeric_results) / len(numeric_results) if len(numeric_results) > 0 else 0
        
        # Add win percentage annotation
        fig.add_annotation(
            text=f"Win Percentage: {win_pct:.1%}",
            showarrow=False,
            font=dict(size=14),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.1
        )
        
        logger.info(f"Created team form chart for {team_name}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating team form chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def plot_feature_importance(feature_names, importance_values):
    """
    Create a feature importance chart for the prediction model.
    
    Args:
        feature_names (list): List of feature names
        importance_values (list): List of importance values
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        if len(feature_names) != len(importance_values):
            raise ValueError("Feature names and importance values must have same length")
        
        # Create dataframe
        data = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        # Sort by importance
        data = data.sort_values('Importance', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            data,
            y='Feature',
            x='Importance',
            orientation='h',
            title="Feature Importance in Prediction Model",
            color='Importance',
            color_continuous_scale=['#ffa726', '#2e7d32']
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Relative Importance",
            yaxis_title=""
        )
        
        logger.info(f"Created feature importance chart with {len(feature_names)} features")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def create_performance_gauge(value, title, min_val=0, max_val=100, 
                           threshold_ranges=None, threshold_colors=None):
    """
    Create a gauge chart for displaying a performance metric.
    
    Args:
        value (float): Value to display
        title (str): Chart title
        min_val (float): Minimum value for the gauge
        max_val (float): Maximum value for the gauge
        threshold_ranges (list): List of threshold ranges (e.g., [20, 40, 60, 80])
        threshold_colors (list): List of colors for each range
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Default threshold ranges and colors if not provided
        if threshold_ranges is None:
            threshold_ranges = [min_val + (max_val - min_val) * i / 4 for i in range(5)]
        
        if threshold_colors is None:
            threshold_colors = ["#e57373", "#ffb74d", "#aed581", "#4caf50"]
        
        # Create steps for gauge
        steps = []
        for i in range(len(threshold_ranges) - 1):
            steps.append(
                dict(
                    range=[threshold_ranges[i], threshold_ranges[i + 1]],
                    color=threshold_colors[i]
                )
            )
        
        # Create gauge chart
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "#1e88e5"},
                'steps': steps
            }
        ))
        
        # Update layout
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=0)
        )
        
        logger.info(f"Created performance gauge for {title} with value {value}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance gauge: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating gauge: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def plot_prediction_confidence_interval(value, lower_bound, upper_bound, title):
    """
    Create a chart showing a prediction with confidence interval.
    
    Args:
        value (float): Predicted value
        lower_bound (float): Lower bound of confidence interval
        upper_bound (float): Upper bound of confidence interval
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add predicted value
        fig.add_trace(go.Indicator(
            mode="number",
            value=value,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0.6, 1]}
        ))
        
        # Add confidence interval
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=value,
            delta={'reference': value, 'relative': False, 'valueformat': '.0f', 'position': "bottom"},
            title={'text': "Confidence Interval"},
            domain={'x': [0, 1], 'y': [0, 0.4]},
            number={'suffix': f" ({lower_bound} - {upper_bound})"}
        ))
        
        # Update layout
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=0)
        )
        
        logger.info(f"Created confidence interval chart for {title} with value {value}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating confidence interval chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig
