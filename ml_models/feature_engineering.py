import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_match_features(match_data, team_data, player_data, venue_data):
    """
    Create features for match prediction models.
    
    Args:
        match_data (pd.DataFrame): Historical match data
        team_data (pd.DataFrame): Team statistics
        player_data (pd.DataFrame): Player statistics
        venue_data (pd.DataFrame): Venue statistics
    
    Returns:
        pd.DataFrame: Feature matrix for match prediction
    """
    # This is a template function that would be implemented in a real system
    # with actual data sources. The implementation here is a placeholder.
    
    features = pd.DataFrame()
    
    # Team-based features
    
    # 1. Recent form (last 5 matches)
    def calculate_recent_form(team_matches, team_name):
        """Calculate recent form based on last 5 matches"""
        recent_matches = team_matches.sort_values('date', ascending=False).head(5)
        wins = sum(recent_matches['winner'] == team_name)
        return wins / len(recent_matches) if len(recent_matches) > 0 else 0.5
    
    # 2. Head-to-head record
    def calculate_head_to_head(matches, team1, team2):
        """Calculate head-to-head record between two teams"""
        h2h_matches = matches[
            ((matches['team1'] == team1) & (matches['team2'] == team2)) | 
            ((matches['team1'] == team2) & (matches['team2'] == team1))
        ]
        
        team1_wins = sum(h2h_matches['winner'] == team1)
        total_matches = len(h2h_matches)
        
        return team1_wins / total_matches if total_matches > 0 else 0.5
    
    # 3. Home advantage
    def calculate_home_advantage(match_row, venue_data):
        """Calculate home advantage based on venue"""
        venue = match_row['venue']
        team1 = match_row['team1']
        
        # Check if venue is home ground for team1
        venue_info = venue_data[venue_data['venue_name'] == venue]
        if not venue_info.empty and venue_info.iloc[0]['home_team'] == team1:
            return 1
        return 0
    
    # 4. Toss win advantage
    def toss_advantage(match_row):
        """Calculate advantage from winning the toss"""
        toss_winner = match_row['toss_winner']
        toss_decision = match_row['toss_decision']
        
        # Analyze if toss winner has advantage based on decision and venue
        # This would be more sophisticated in a real implementation
        if toss_decision == 'bat':
            return 0.6  # Slight advantage for batting first in certain conditions
        else:
            return 0.4  # Slight disadvantage for fielding first
    
    # Player-based features
    
    # 1. Team strength based on player ratings
    def calculate_team_strength(match_row, player_data):
        """Calculate team strength based on player ratings"""
        team1 = match_row['team1']
        team2 = match_row['team2']
        
        # Get players from each team
        team1_players = player_data[player_data['team'] == team1]
        team2_players = player_data[player_data['team'] == team2]
        
        # Calculate average player rating for each team
        team1_rating = team1_players['player_rating'].mean()
        team2_rating = team2_players['player_rating'].mean()
        
        # Return relative strength
        return team1_rating / (team1_rating + team2_rating)
    
    # 2. Key player availability
    def key_player_availability(match_row, player_data):
        """Check availability of key players"""
        team1 = match_row['team1']
        team2 = match_row['team2']
        
        # Get key players (high rating) from each team
        team1_key_players = player_data[(player_data['team'] == team1) & 
                                        (player_data['player_rating'] > 8)]
        team2_key_players = player_data[(player_data['team'] == team2) & 
                                        (player_data['player_rating'] > 8)]
        
        # Calculate availability ratio (would use actual availability data in real implementation)
        team1_availability = sum(team1_key_players['is_available']) / len(team1_key_players)
        team2_availability = sum(team2_key_players['is_available']) / len(team2_key_players)
        
        return team1_availability - team2_availability
    
    # Venue and condition features
    
    # 1. Pitch condition effect
    def pitch_condition_effect(match_row):
        """Analyze effect of pitch conditions"""
        pitch_type = match_row['pitch_type']
        
        # Effect based on pitch type
        if pitch_type == 'batting_friendly':
            return 0.7  # Advantage to better batting team
        elif pitch_type == 'bowling_friendly':
            return 0.3  # Advantage to better bowling team
        else:
            return 0.5  # Neutral
    
    # 2. Weather impact
    def weather_impact(match_row):
        """Analyze impact of weather conditions"""
        weather = match_row['weather']
        
        if weather == 'rainy':
            return 0.4  # Slight advantage to bowling first
        elif weather == 'humid':
            return 0.6  # Slight advantage to batting first
        else:
            return 0.5  # Neutral
    
    # In a real implementation, these functions would be applied to create
    # a feature matrix for all matches in the dataset
    
    # Return placeholder dataframe
    return pd.DataFrame({
        'team1_recent_form': [0.6],
        'team2_recent_form': [0.4],
        'head_to_head_ratio': [0.55],
        'home_advantage': [1],
        'toss_advantage': [0.6],
        'team_strength_ratio': [0.52],
        'key_player_availability': [0.1],
        'pitch_effect': [0.7],
        'weather_impact': [0.5]
    })


def create_player_features(player_data, match_data, team_data, opponent_data):
    """
    Create features for player performance prediction.
    
    Args:
        player_data (pd.DataFrame): Historical player performance data
        match_data (dict): Match information
        team_data (pd.DataFrame): Team statistics
        opponent_data (pd.DataFrame): Opponent team statistics
    
    Returns:
        pd.DataFrame: Feature matrix for player performance prediction
    """
    # This is a template function that would be implemented in a real system
    # with actual data sources. The implementation here is a placeholder.
    
    features = pd.DataFrame()
    
    # Player historical performance features
    
    # 1. Recent form (last 5 matches)
    def calculate_player_recent_form(player_id, player_data):
        """Calculate player's recent form based on last 5 matches"""
        player_matches = player_data[player_data['player_id'] == player_id]
        recent_matches = player_matches.sort_values('date', ascending=False).head(5)
        
        if len(recent_matches) == 0:
            return {
                'recent_avg_runs': 0,
                'recent_avg_sr': 0,
                'recent_avg_wickets': 0,
                'recent_avg_economy': 0
            }
        
        # Calculate batting metrics
        recent_avg_runs = recent_matches['runs'].mean()
        recent_avg_sr = recent_matches['strike_rate'].mean()
        
        # Calculate bowling metrics
        recent_avg_wickets = recent_matches['wickets'].mean()
        recent_avg_economy = recent_matches['economy_rate'].mean()
        
        return {
            'recent_avg_runs': recent_avg_runs,
            'recent_avg_sr': recent_avg_sr,
            'recent_avg_wickets': recent_avg_wickets,
            'recent_avg_economy': recent_avg_economy
        }
    
    # 2. Performance against specific opponent
    def calculate_vs_opponent(player_id, opponent_team, player_data):
        """Calculate player's performance against specific opponent"""
        player_matches = player_data[
            (player_data['player_id'] == player_id) & 
            (player_data['opponent'] == opponent_team)
        ]
        
        if len(player_matches) == 0:
            return {
                'vs_opp_avg_runs': 0,
                'vs_opp_avg_sr': 0,
                'vs_opp_avg_wickets': 0,
                'vs_opp_avg_economy': 0
            }
        
        # Calculate batting metrics
        vs_opp_avg_runs = player_matches['runs'].mean()
        vs_opp_avg_sr = player_matches['strike_rate'].mean()
        
        # Calculate bowling metrics
        vs_opp_avg_wickets = player_matches['wickets'].mean()
        vs_opp_avg_economy = player_matches['economy_rate'].mean()
        
        return {
            'vs_opp_avg_runs': vs_opp_avg_runs,
            'vs_opp_avg_sr': vs_opp_avg_sr,
            'vs_opp_avg_wickets': vs_opp_avg_wickets,
            'vs_opp_avg_economy': vs_opp_avg_economy
        }
    
    # 3. Performance at venue
    def calculate_venue_performance(player_id, venue, player_data):
        """Calculate player's performance at specific venue"""
        player_venue_matches = player_data[
            (player_data['player_id'] == player_id) & 
            (player_data['venue'] == venue)
        ]
        
        if len(player_venue_matches) == 0:
            return {
                'venue_avg_runs': 0,
                'venue_avg_sr': 0,
                'venue_avg_wickets': 0,
                'venue_avg_economy': 0
            }
        
        # Calculate venue-specific metrics
        venue_avg_runs = player_venue_matches['runs'].mean()
        venue_avg_sr = player_venue_matches['strike_rate'].mean()
        venue_avg_wickets = player_venue_matches['wickets'].mean()
        venue_avg_economy = player_venue_matches['economy_rate'].mean()
        
        return {
            'venue_avg_runs': venue_avg_runs,
            'venue_avg_sr': venue_avg_sr,
            'venue_avg_wickets': venue_avg_wickets,
            'venue_avg_economy': venue_avg_economy
        }
    
    # 4. Performance in different pitch conditions
    def calculate_pitch_performance(player_id, pitch_type, player_data):
        """Calculate player's performance in specific pitch conditions"""
        player_pitch_matches = player_data[
            (player_data['player_id'] == player_id) & 
            (player_data['pitch_type'] == pitch_type)
        ]
        
        if len(player_pitch_matches) == 0:
            return {
                'pitch_avg_runs': 0,
                'pitch_avg_sr': 0,
                'pitch_avg_wickets': 0,
                'pitch_avg_economy': 0
            }
        
        # Calculate pitch-specific metrics
        pitch_avg_runs = player_pitch_matches['runs'].mean()
        pitch_avg_sr = player_pitch_matches['strike_rate'].mean()
        pitch_avg_wickets = player_pitch_matches['wickets'].mean()
        pitch_avg_economy = player_pitch_matches['economy_rate'].mean()
        
        return {
            'pitch_avg_runs': pitch_avg_runs,
            'pitch_avg_sr': pitch_avg_sr,
            'pitch_avg_wickets': pitch_avg_wickets,
            'pitch_avg_economy': pitch_avg_economy
        }
    
    # In a real implementation, these functions would be applied to create
    # a feature matrix for player performance prediction
    
    # Return placeholder dataframe with sample features
    return pd.DataFrame({
        'recent_avg_runs': [35.2],
        'recent_avg_sr': [145.6],
        'recent_avg_wickets': [1.2],
        'recent_avg_economy': [8.3],
        'vs_opp_avg_runs': [42.1],
        'vs_opp_avg_sr': [152.3],
        'vs_opp_avg_wickets': [0.8],
        'vs_opp_avg_economy': [8.9],
        'venue_avg_runs': [38.7],
        'venue_avg_sr': [148.2],
        'venue_avg_wickets': [1.0],
        'venue_avg_economy': [8.5],
        'pitch_avg_runs': [33.5],
        'pitch_avg_sr': [140.8],
        'pitch_avg_wickets': [1.5],
        'pitch_avg_economy': [7.8]
    })
