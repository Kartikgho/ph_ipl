"""
Data processor for IPL cricket data.
This module handles cleaning, transformation, and feature extraction from raw data.
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_processor.log'
)
logger = logging.getLogger('data_processor')

class IPLDataProcessor:
    """
    Process and transform IPL cricket data for machine learning models.
    """
    
    def __init__(self):
        """Initialize the data processor"""
        logger.info("Initializing IPL Data Processor")
    
    def clean_match_data(self, match_data):
        """
        Clean and preprocess raw match data.
        
        Args:
            match_data (pd.DataFrame): Raw match data scraped from web
            
        Returns:
            pd.DataFrame: Cleaned match data
        """
        if match_data is None or match_data.empty:
            logger.warning("Empty match data received for cleaning")
            return pd.DataFrame()
            
        logger.info(f"Cleaning match data: {len(match_data)} records")
        
        try:
            # Create a copy to avoid modifying the original
            cleaned_data = match_data.copy()
            
            # Fill missing values
            cleaned_data.fillna({
                'result': 'No result',
                'winner': 'No result',
                'player_of_match': 'Unknown'
            }, inplace=True)
            
            # Convert date strings to datetime
            if 'date' in cleaned_data.columns:
                cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
            
            # Extract numeric values from score (e.g., "165/6" -> 165)
            if 'team1_score' in cleaned_data.columns:
                cleaned_data['team1_runs'] = cleaned_data['team1_score'].apply(
                    lambda x: int(re.search(r'(\d+)\/\d+', str(x)).group(1)) if pd.notna(x) and re.search(r'(\d+)\/\d+', str(x)) else np.nan
                )
                cleaned_data['team1_wickets'] = cleaned_data['team1_score'].apply(
                    lambda x: int(re.search(r'\d+\/(\d+)', str(x)).group(1)) if pd.notna(x) and re.search(r'\d+\/(\d+)', str(x)) else np.nan
                )
            
            if 'team2_score' in cleaned_data.columns:
                cleaned_data['team2_runs'] = cleaned_data['team2_score'].apply(
                    lambda x: int(re.search(r'(\d+)\/\d+', str(x)).group(1)) if pd.notna(x) and re.search(r'(\d+)\/\d+', str(x)) else np.nan
                )
                cleaned_data['team2_wickets'] = cleaned_data['team2_score'].apply(
                    lambda x: int(re.search(r'\d+\/(\d+)', str(x)).group(1)) if pd.notna(x) and re.search(r'\d+\/(\d+)', str(x)) else np.nan
                )
            
            # Create consistent venue names
            if 'venue' in cleaned_data.columns:
                cleaned_data['venue'] = cleaned_data['venue'].apply(
                    lambda x: x.split(',')[0].strip() if pd.notna(x) and ',' in x else x
                )
            
            # Add match_id if not present
            if 'match_id' not in cleaned_data.columns:
                cleaned_data['match_id'] = cleaned_data.index
            
            logger.info(f"Match data cleaning completed: {len(cleaned_data)} records")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning match data: {str(e)}")
            # Return original data if cleaning fails
            return match_data
    
    def clean_player_data(self, player_data):
        """
        Clean and preprocess raw player data.
        
        Args:
            player_data (pd.DataFrame): Raw player data scraped from web
            
        Returns:
            pd.DataFrame: Cleaned player data
        """
        if player_data is None or player_data.empty:
            logger.warning("Empty player data received for cleaning")
            return pd.DataFrame()
            
        logger.info(f"Cleaning player data: {len(player_data)} records")
        
        try:
            # Create a copy to avoid modifying the original
            cleaned_data = player_data.copy()
            
            # Convert numeric columns
            numeric_cols = ['Matches', 'Innings', 'Runs', 'Average', 'Strike Rate', 
                            'Balls Bowled', 'Wickets', 'Economy', 'Bowling Average']
            
            for col in numeric_cols:
                if col in cleaned_data.columns:
                    # Handle special characters and convert to numeric
                    cleaned_data[col] = cleaned_data[col].apply(
                        lambda x: str(x).replace('-', '0').replace('*', '') if pd.notna(x) else '0'
                    )
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Standardize player names
            if 'Player' in cleaned_data.columns:
                cleaned_data['Player'] = cleaned_data['Player'].apply(
                    lambda x: x.strip() if pd.notna(x) else 'Unknown'
                )
            
            # Add player_id if not present
            if 'player_id' not in cleaned_data.columns:
                cleaned_data['player_id'] = cleaned_data.index
            
            logger.info(f"Player data cleaning completed: {len(cleaned_data)} records")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning player data: {str(e)}")
            # Return original data if cleaning fails
            return player_data
    
    def clean_team_data(self, team_data):
        """
        Clean and preprocess raw team data.
        
        Args:
            team_data (pd.DataFrame): Raw team data scraped from web
            
        Returns:
            pd.DataFrame: Cleaned team data
        """
        if team_data is None or team_data.empty:
            logger.warning("Empty team data received for cleaning")
            return pd.DataFrame()
            
        logger.info(f"Cleaning team data: {len(team_data)} records")
        
        try:
            # Create a copy to avoid modifying the original
            cleaned_data = team_data.copy()
            
            # Convert numeric columns
            numeric_cols = ['Matches', 'Won', 'Lost', 'Tied', 'NR', 'Points', 
                            'For', 'Against', 'NRR']
            
            for col in numeric_cols:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Standardize team names
            if 'Team' in cleaned_data.columns:
                cleaned_data['Team'] = cleaned_data['Team'].apply(
                    lambda x: x.strip() if pd.notna(x) else 'Unknown'
                )
            
            # Add team_id if not present
            if 'team_id' not in cleaned_data.columns:
                cleaned_data['team_id'] = cleaned_data.index
            
            logger.info(f"Team data cleaning completed: {len(cleaned_data)} records")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning team data: {str(e)}")
            # Return original data if cleaning fails
            return team_data
    
    def extract_batting_features(self, batting_data):
        """
        Extract batting features from player performance data.
        
        Args:
            batting_data (pd.DataFrame): Cleaned batting performance data
            
        Returns:
            pd.DataFrame: DataFrame with batting features
        """
        if batting_data is None or batting_data.empty:
            logger.warning("Empty batting data received for feature extraction")
            return pd.DataFrame()
            
        logger.info(f"Extracting batting features from {len(batting_data)} records")
        
        try:
            # Create a copy to avoid modifying the original
            features = batting_data.copy()
            
            # Calculate additional features
            if all(col in features.columns for col in ['Runs', 'Innings']):
                features['runs_per_inning'] = features['Runs'] / features['Innings'].replace(0, 1)
            
            if all(col in features.columns for col in ['Runs', 'Balls']):
                features['strike_rate'] = (features['Runs'] / features['Balls'].replace(0, 1)) * 100
            
            if all(col in features.columns for col in ['Fours', 'Sixes', 'Runs']):
                features['boundary_percentage'] = ((features['Fours'] * 4 + features['Sixes'] * 6) / 
                                                 features['Runs'].replace(0, 1)) * 100
            
            if all(col in features.columns for col in ['Fifties', 'Hundreds', 'Innings']):
                features['milestone_rate'] = (features['Fifties'] + features['Hundreds']) / features['Innings'].replace(0, 1)
            
            logger.info(f"Batting feature extraction completed")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting batting features: {str(e)}")
            # Return original data if extraction fails
            return batting_data
    
    def extract_bowling_features(self, bowling_data):
        """
        Extract bowling features from player performance data.
        
        Args:
            bowling_data (pd.DataFrame): Cleaned bowling performance data
            
        Returns:
            pd.DataFrame: DataFrame with bowling features
        """
        if bowling_data is None or bowling_data.empty:
            logger.warning("Empty bowling data received for feature extraction")
            return pd.DataFrame()
            
        logger.info(f"Extracting bowling features from {len(bowling_data)} records")
        
        try:
            # Create a copy to avoid modifying the original
            features = bowling_data.copy()
            
            # Calculate additional features
            if all(col in features.columns for col in ['Wickets', 'Innings']):
                features['wickets_per_inning'] = features['Wickets'] / features['Innings'].replace(0, 1)
            
            if all(col in features.columns for col in ['Wickets', 'Balls Bowled']):
                features['strike_rate'] = features['Balls Bowled'] / features['Wickets'].replace(0, 1)
            
            if all(col in features.columns for col in ['Wickets', 'Runs Conceded']):
                features['bowling_avg'] = features['Runs Conceded'] / features['Wickets'].replace(0, 1)
            
            if all(col in features.columns for col in ['Runs Conceded', 'Overs']):
                features['economy_rate'] = features['Runs Conceded'] / features['Overs'].replace(0, 1)
            
            if all(col in features.columns for col in ['Maidens', 'Innings']):
                features['maiden_rate'] = features['Maidens'] / features['Innings'].replace(0, 1)
            
            logger.info(f"Bowling feature extraction completed")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting bowling features: {str(e)}")
            # Return original data if extraction fails
            return bowling_data
    
    def calculate_recent_form(self, player_data, num_matches=5):
        """
        Calculate recent form metrics based on last N matches.
        
        Args:
            player_data (pd.DataFrame): Player performance data
            num_matches (int): Number of recent matches to consider
            
        Returns:
            pd.DataFrame: DataFrame with recent form metrics
        """
        if player_data is None or player_data.empty:
            logger.warning("Empty player data received for form calculation")
            return pd.DataFrame()
            
        logger.info(f"Calculating recent form for {len(player_data)} players, last {num_matches} matches")
        
        try:
            # Group by player and sort by date
            if not all(col in player_data.columns for col in ['player_id', 'date']):
                logger.error("Required columns missing for form calculation")
                return pd.DataFrame()
                
            form_data = []
            
            # Process each player
            for player_id, player_matches in player_data.groupby('player_id'):
                # Sort matches by date (most recent first)
                player_matches = player_matches.sort_values('date', ascending=False)
                
                # Take only the most recent N matches
                recent_matches = player_matches.head(num_matches)
                
                if len(recent_matches) < 2:  # Need at least 2 matches for meaningful form
                    continue
                
                # Calculate form metrics for batsmen
                if 'runs' in recent_matches.columns:
                    recent_runs = recent_matches['runs'].fillna(0).astype(int).tolist()
                    avg_runs = sum(recent_runs) / len(recent_runs)
                    runs_trend = recent_runs[0] - avg_runs  # Positive means improving form
                    
                    form_data.append({
                        'player_id': player_id,
                        'recent_matches': len(recent_matches),
                        'avg_recent_runs': avg_runs,
                        'runs_trend': runs_trend,
                        'last_3_scores': recent_runs[:3],
                        'form_date': recent_matches['date'].max()
                    })
                
                # Calculate form metrics for bowlers
                if 'wickets' in recent_matches.columns:
                    recent_wickets = recent_matches['wickets'].fillna(0).astype(int).tolist()
                    avg_wickets = sum(recent_wickets) / len(recent_wickets)
                    wickets_trend = recent_wickets[0] - avg_wickets  # Positive means improving form
                    
                    if len(form_data) > 0 and form_data[-1]['player_id'] == player_id:
                        # Update existing player entry
                        form_data[-1].update({
                            'avg_recent_wickets': avg_wickets,
                            'wickets_trend': wickets_trend,
                            'last_3_wickets': recent_wickets[:3]
                        })
                    else:
                        # Create new player entry
                        form_data.append({
                            'player_id': player_id,
                            'recent_matches': len(recent_matches),
                            'avg_recent_wickets': avg_wickets,
                            'wickets_trend': wickets_trend,
                            'last_3_wickets': recent_wickets[:3],
                            'form_date': recent_matches['date'].max()
                        })
            
            form_df = pd.DataFrame(form_data)
            logger.info(f"Recent form calculation completed for {len(form_df)} players")
            return form_df
            
        except Exception as e:
            logger.error(f"Error calculating recent form: {str(e)}")
            # Return empty dataframe if calculation fails
            return pd.DataFrame()
    
    def prepare_match_features(self, match_data, team_data, player_data, recent_form_data):
        """
        Prepare features for match prediction models.
        
        Args:
            match_data (pd.DataFrame): Historical match data
            team_data (pd.DataFrame): Team statistics
            player_data (pd.DataFrame): Player statistics
            recent_form_data (pd.DataFrame): Recent form calculations
            
        Returns:
            pd.DataFrame: Feature matrix for match prediction
        """
        if match_data is None or match_data.empty:
            logger.warning("Empty match data received for feature preparation")
            return pd.DataFrame()
            
        logger.info(f"Preparing match features from {len(match_data)} matches")
        
        try:
            # Create features for each match
            feature_data = []
            
            for _, match in match_data.iterrows():
                team1 = match.get('team1')
                team2 = match.get('team2')
                
                if not team1 or not team2:
                    continue
                
                # Get team stats
                team1_stats = team_data[team_data['Team'] == team1].iloc[0] if len(team_data[team_data['Team'] == team1]) > 0 else None
                team2_stats = team_data[team_data['Team'] == team2].iloc[0] if len(team_data[team_data['Team'] == team2]) > 0 else None
                
                # Skip if team stats not available
                if team1_stats is None or team2_stats is None:
                    continue
                
                # Calculate win ratio
                team1_win_ratio = team1_stats.get('Won', 0) / team1_stats.get('Matches', 1)
                team2_win_ratio = team2_stats.get('Won', 0) / team2_stats.get('Matches', 1)
                
                # Calculate net run rate difference
                nrr_diff = team1_stats.get('NRR', 0) - team2_stats.get('NRR', 0)
                
                # Calculate head-to-head stats
                h2h_matches = match_data[
                    ((match_data['team1'] == team1) & (match_data['team2'] == team2)) |
                    ((match_data['team1'] == team2) & (match_data['team2'] == team1))
                ]
                
                team1_h2h_wins = h2h_matches[h2h_matches['winner'] == team1].shape[0]
                team2_h2h_wins = h2h_matches[h2h_matches['winner'] == team2].shape[0]
                h2h_ratio = team1_h2h_wins / max(1, team1_h2h_wins + team2_h2h_wins)
                
                # Check home advantage
                venue = match.get('venue', '')
                team1_home = 1 if team1.lower() in venue.lower() else 0
                team2_home = 1 if team2.lower() in venue.lower() else 0
                
                # Get toss information
                toss_winner = match.get('toss_winner', '')
                toss_advantage = 1 if toss_winner == team1 else -1 if toss_winner == team2 else 0
                
                # Get recent team form data (average of player form)
                team1_players = player_data[player_data['Team'] == team1]['player_id'].tolist()
                team2_players = player_data[player_data['Team'] == team2]['player_id'].tolist()
                
                team1_form = recent_form_data[recent_form_data['player_id'].isin(team1_players)]
                team2_form = recent_form_data[recent_form_data['player_id'].isin(team2_players)]
                
                team1_batting_form = team1_form['avg_recent_runs'].mean() if 'avg_recent_runs' in team1_form else 0
                team2_batting_form = team2_form['avg_recent_runs'].mean() if 'avg_recent_runs' in team2_form else 0
                
                team1_bowling_form = team1_form['avg_recent_wickets'].mean() if 'avg_recent_wickets' in team1_form else 0
                team2_bowling_form = team2_form['avg_recent_wickets'].mean() if 'avg_recent_wickets' in team2_form else 0
                
                # Build feature vector
                features = {
                    'match_id': match.get('match_id', 0),
                    'team1': team1,
                    'team2': team2,
                    'team1_win_ratio': team1_win_ratio,
                    'team2_win_ratio': team2_win_ratio,
                    'win_ratio_diff': team1_win_ratio - team2_win_ratio,
                    'nrr_diff': nrr_diff,
                    'h2h_ratio': h2h_ratio,
                    'team1_home': team1_home,
                    'team2_home': team2_home,
                    'home_advantage': team1_home - team2_home,
                    'toss_advantage': toss_advantage,
                    'team1_batting_form': team1_batting_form,
                    'team2_batting_form': team2_batting_form,
                    'batting_form_diff': team1_batting_form - team2_batting_form,
                    'team1_bowling_form': team1_bowling_form,
                    'team2_bowling_form': team2_bowling_form,
                    'bowling_form_diff': team1_bowling_form - team2_bowling_form
                }
                
                # Add target variable if available
                if 'winner' in match:
                    features['target'] = 1 if match['winner'] == team1 else 0
                
                feature_data.append(features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(feature_data)
            logger.info(f"Match feature preparation completed: {len(features_df)} records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing match features: {str(e)}")
            # Return empty dataframe if preparation fails
            return pd.DataFrame()
