"""
Data helper utilities for IPL prediction system.
This module provides functions for data loading, transformation, and manipulation.
"""

import pandas as pd
import numpy as np
import random
import os
import json
import logging
from datetime import datetime, timedelta
from config import IPL_TEAMS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_helpers.log'
)
logger = logging.getLogger('data_helpers')

def get_sample_player_data(player_name=None, team=None, role=None):
    """
    Get structured player data.
    
    Args:
        player_name (str, optional): Filter by player name
        team (str, optional): Filter by team
        role (str, optional): Filter by role
        
    Returns:
        dict: Player data structure or None if not found
    """
    try:
        # In a real implementation, this would query a database or API
        # Here we return a structured data object with empty/zero values
        if not player_name:
            return None
            
        player_data = {
            'name': player_name,
            'team': team or 'Unknown Team',
            'role': role or 'Unknown Role',
            'matches_played': 0,
            'batting_stats': {
                'innings': 0,
                'runs': 0,
                'average': 0.0,
                'strike_rate': 0.0,
                'highest_score': 0,
                'fifties': 0,
                'hundreds': 0
            },
            'bowling_stats': {
                'innings': 0,
                'wickets': 0,
                'economy': 0.0,
                'average': 0.0,
                'best_figures': '0/0'
            },
            'recent_performances': []
        }
        
        logger.info(f"Returned empty player data structure for {player_name}")
        return player_data
        
    except Exception as e:
        logger.error(f"Error in get_sample_player_data: {str(e)}")
        return None

def get_sample_team_data(team_name=None):
    """
    Get structured team data.
    
    Args:
        team_name (str, optional): Team name to filter by
        
    Returns:
        dict: Team data structure or None if not found
    """
    try:
        # In a real implementation, this would query a database or API
        # Here we return a structured data object with empty/zero values
        if not team_name or team_name not in IPL_TEAMS:
            return None
            
        team_data = {
            'name': team_name,
            'matches_played': 0,
            'wins': 0,
            'losses': 0,
            'points': 0,
            'nrr': 0.0,
            'titles': 0,
            'players': [],
            'recent_matches': []
        }
        
        logger.info(f"Returned empty team data structure for {team_name}")
        return team_data
        
    except Exception as e:
        logger.error(f"Error in get_sample_team_data: {str(e)}")
        return None

def prepare_match_data(team1, team2, venue, toss_winner, toss_decision, 
                      pitch_conditions, weather_conditions):
    """
    Prepare match data for prediction.
    
    Args:
        team1 (str): First team
        team2 (str): Second team
        venue (str): Match venue
        toss_winner (str): Team that won the toss
        toss_decision (str): Decision made by toss winner (bat/field)
        pitch_conditions (str): Pitch conditions
        weather_conditions (str): Weather conditions
        
    Returns:
        dict: Prepared match data for prediction
    """
    try:
        # Create match data dictionary
        match_data = {
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'pitch_conditions': pitch_conditions,
            'weather_conditions': weather_conditions
        }
        
        # Add additional computed fields that might be needed by the model
        match_data['home_ground_advantage'] = 1 if team1.lower() in venue.lower() else 0
        match_data['toss_advantage'] = 1 if toss_winner == team1 else 0
        
        # Encode pitch conditions (this would be more sophisticated in a real system)
        pitch_factor = 0.0
        if pitch_conditions == 'Very Batting Friendly':
            pitch_factor = 1.0
        elif pitch_conditions == 'Batting Friendly':
            pitch_factor = 0.75
        elif pitch_conditions == 'Neutral':
            pitch_factor = 0.5
        elif pitch_conditions == 'Bowling Friendly':
            pitch_factor = 0.25
        elif pitch_conditions == 'Very Bowling Friendly':
            pitch_factor = 0.0
        
        match_data['pitch_factor'] = pitch_factor
        
        # Encode weather conditions (this would be more sophisticated in a real system)
        weather_factor = 0.5
        if weather_conditions == 'Clear':
            weather_factor = 0.8
        elif weather_conditions == 'Partly Cloudy':
            weather_factor = 0.6
        elif weather_conditions == 'Cloudy':
            weather_factor = 0.4
        elif weather_conditions == 'Light Rain':
            weather_factor = 0.2
        elif weather_conditions == 'Humid':
            weather_factor = 0.5
        
        match_data['weather_factor'] = weather_factor
        
        logger.info(f"Prepared match data for {team1} vs {team2}")
        return match_data
        
    except Exception as e:
        logger.error(f"Error in prepare_match_data: {str(e)}")
        return {
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'pitch_conditions': pitch_conditions,
            'weather_conditions': weather_conditions
        }

def prepare_player_data(player_name, team, role, opponent, venue):
    """
    Prepare player data for prediction.
    
    Args:
        player_name (str): Player name
        team (str): Player's team
        role (str): Player's role
        opponent (str): Opponent team
        venue (str): Match venue
        
    Returns:
        dict: Prepared player data for prediction
    """
    try:
        # Create player data dictionary
        player_data = {
            'player_name': player_name,
            'team': team,
            'role': role,
            'opponent': opponent,
            'venue': venue
        }
        
        logger.info(f"Prepared player data for {player_name}")
        return player_data
        
    except Exception as e:
        logger.error(f"Error in prepare_player_data: {str(e)}")
        return {
            'player_name': player_name,
            'team': team,
            'role': role,
            'opponent': opponent,
            'venue': venue
        }

def load_match_data(file_path=None):
    """
    Load match data from file.
    
    Args:
        file_path (str, optional): Path to data file
        
    Returns:
        pd.DataFrame: Match data or empty DataFrame if file not found
    """
    try:
        if file_path and os.path.exists(file_path):
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return pd.DataFrame()
                
            logger.info(f"Loaded match data from {file_path}: {len(data)} records")
            return data
        else:
            logger.warning("File path not provided or file not found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading match data: {str(e)}")
        return pd.DataFrame()

def load_player_data(file_path=None):
    """
    Load player data from file.
    
    Args:
        file_path (str, optional): Path to data file
        
    Returns:
        pd.DataFrame: Player data or empty DataFrame if file not found
    """
    try:
        if file_path and os.path.exists(file_path):
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return pd.DataFrame()
                
            logger.info(f"Loaded player data from {file_path}: {len(data)} records")
            return data
        else:
            logger.warning("File path not provided or file not found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading player data: {str(e)}")
        return pd.DataFrame()

def load_team_data(file_path=None):
    """
    Load team data from file.
    
    Args:
        file_path (str, optional): Path to data file
        
    Returns:
        pd.DataFrame: Team data or empty DataFrame if file not found
    """
    try:
        if file_path and os.path.exists(file_path):
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return pd.DataFrame()
                
            logger.info(f"Loaded team data from {file_path}: {len(data)} records")
            return data
        else:
            logger.warning("File path not provided or file not found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading team data: {str(e)}")
        return pd.DataFrame()

def convert_datetime(date_str, format_str="%Y-%m-%d"):
    """
    Convert date string to datetime object.
    
    Args:
        date_str (str): Date string
        format_str (str): Date format string
        
    Returns:
        datetime: Datetime object or None if conversion fails
    """
    try:
        return datetime.strptime(date_str, format_str)
    except (ValueError, TypeError):
        logger.error(f"Error converting date: {date_str}")
        return None

def calculate_date_diff(date1, date2):
    """
    Calculate difference between two dates in days.
    
    Args:
        date1 (datetime/str): First date
        date2 (datetime/str): Second date
        
    Returns:
        int: Difference in days or None if calculation fails
    """
    try:
        # Convert string dates to datetime if needed
        if isinstance(date1, str):
            date1 = convert_datetime(date1)
        if isinstance(date2, str):
            date2 = convert_datetime(date2)
            
        if date1 and date2:
            return abs((date1 - date2).days)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calculating date difference: {str(e)}")
        return None

def filter_recent_data(data, date_column='date', days=30):
    """
    Filter data to include only recent records.
    
    Args:
        data (pd.DataFrame): Data to filter
        date_column (str): Column containing dates
        days (int): Number of days to consider recent
        
    Returns:
        pd.DataFrame: Filtered data
    """
    try:
        if data is None or data.empty or date_column not in data.columns:
            return pd.DataFrame()
            
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter data
        recent_data = data[data[date_column] >= cutoff_date]
        
        logger.info(f"Filtered data to {len(recent_data)} recent records (last {days} days)")
        return recent_data
        
    except Exception as e:
        logger.error(f"Error filtering recent data: {str(e)}")
        return data  # Return original data on error

def calculate_win_percentage(wins, total_matches):
    """
    Calculate win percentage.
    
    Args:
        wins (int): Number of wins
        total_matches (int): Total number of matches
        
    Returns:
        float: Win percentage (0-100)
    """
    try:
        if total_matches > 0:
            return (wins / total_matches) * 100
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating win percentage: {str(e)}")
        return 0.0

def calculate_average(values):
    """
    Calculate average of a list of values.
    
    Args:
        values (list): List of numeric values
        
    Returns:
        float: Average value
    """
    try:
        if values and len(values) > 0:
            # Filter out None and non-numeric values
            valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if valid_values:
                return sum(valid_values) / len(valid_values)
        return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating average: {str(e)}")
        return 0.0

def create_date_range(start_date, end_date, freq='D'):
    """
    Create a range of dates.
    
    Args:
        start_date (datetime/str): Start date
        end_date (datetime/str): End date
        freq (str): Frequency ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        pd.DatetimeIndex: Range of dates
    """
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = convert_datetime(start_date)
        if isinstance(end_date, str):
            end_date = convert_datetime(end_date)
            
        if start_date and end_date:
            return pd.date_range(start=start_date, end=end_date, freq=freq)
        else:
            return pd.DatetimeIndex([])
            
    except Exception as e:
        logger.error(f"Error creating date range: {str(e)}")
        return pd.DatetimeIndex([])

def save_data_to_file(data, file_path, file_type='csv'):
    """
    Save data to file.
    
    Args:
        data (pd.DataFrame/dict): Data to save
        file_path (str): Path to save file
        file_type (str): File type ('csv' or 'json')
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            
        # Save based on file type
        if file_type.lower() == 'csv':
            data.to_csv(file_path, index=False)
        elif file_type.lower() == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(file_path, orient='records')
            else:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return False
            
        logger.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to file: {str(e)}")
        return False

def create_dummy_match_result(team1, team2, venue, match_date=None):
    """
    Create a structured match result object for API responses where data is unavailable.
    This function does NOT generate mock data - it creates an empty data structure.
    
    Args:
        team1 (str): First team
        team2 (str): Second team
        venue (str): Match venue
        match_date (datetime/str, optional): Match date
        
    Returns:
        dict: Match result structure with empty/zero values
    """
    if match_date is None:
        match_date = datetime.now()
    elif isinstance(match_date, str):
        match_date = convert_datetime(match_date)
        
    match_result = {
        'match_id': None,
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'date': match_date,
        'result': "No result available",
        'winner': None,
        'team1_score': None,
        'team2_score': None,
        'player_of_match': None
    }
    
    logger.info(f"Created empty match result structure for {team1} vs {team2}")
    return match_result
