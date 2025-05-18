"""
Database utilities for the IPL prediction system.
This module handles data storage and retrieval operations.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from backend.models import Match, Player, Team, Prediction, PlayerPrediction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='db_utils.log'
)
logger = logging.getLogger('db_utils')

class DataManager:
    """
    Manage data storage and retrieval for the IPL prediction system.
    This implementation uses JSON files for storage, but could be extended to use a database.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the data manager with a data directory.
        
        Args:
            data_dir (str): Directory to store data files
        """
        self.data_dir = data_dir
        
        # Create data directory and subdirectories if they don't exist
        for subdir in ['matches', 'players', 'teams', 'predictions']:
            dir_path = os.path.join(data_dir, subdir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        logger.info(f"DataManager initialized with data directory: {data_dir}")
    
    def _get_file_path(self, category, file_id=None):
        """
        Get the file path for a specific data category and ID.
        
        Args:
            category (str): Data category (matches, players, teams, predictions)
            file_id (str, optional): Specific ID for the file
        
        Returns:
            str: File path
        """
        if file_id:
            return os.path.join(self.data_dir, category, f"{file_id}.json")
        else:
            return os.path.join(self.data_dir, category)
    
    def save_match(self, match):
        """
        Save a match to storage.
        
        Args:
            match (Match): Match object to save
            
        Returns:
            bool: Success status
        """
        try:
            if not isinstance(match, Match):
                raise TypeError("Expected Match object")
            
            # Create a unique ID if needed
            if not match.match_id:
                match.match_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Convert to dictionary
            match_dict = match.to_dict()
            
            # Save to file
            file_path = self._get_file_path('matches', match.match_id)
            with open(file_path, 'w') as f:
                json.dump(match_dict, f, default=str, indent=4)
            
            logger.info(f"Match saved: {match.match_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving match: {str(e)}")
            return False
    
    def save_player(self, player):
        """
        Save a player to storage.
        
        Args:
            player (Player): Player object to save
            
        Returns:
            bool: Success status
        """
        try:
            if not isinstance(player, Player):
                raise TypeError("Expected Player object")
            
            # Create a unique ID if needed
            if not player.player_id:
                player.player_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Convert to dictionary
            player_dict = player.to_dict()
            
            # Save to file
            file_path = self._get_file_path('players', player.player_id)
            with open(file_path, 'w') as f:
                json.dump(player_dict, f, default=str, indent=4)
            
            logger.info(f"Player saved: {player.player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving player: {str(e)}")
            return False
    
    def save_team(self, team):
        """
        Save a team to storage.
        
        Args:
            team (Team): Team object to save
            
        Returns:
            bool: Success status
        """
        try:
            if not isinstance(team, Team):
                raise TypeError("Expected Team object")
            
            # Create a unique ID if needed
            if not team.team_id:
                team.team_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Convert to dictionary
            team_dict = team.to_dict()
            
            # Save to file
            file_path = self._get_file_path('teams', team.team_id)
            with open(file_path, 'w') as f:
                json.dump(team_dict, f, default=str, indent=4)
            
            logger.info(f"Team saved: {team.team_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving team: {str(e)}")
            return False
    
    def save_prediction(self, prediction):
        """
        Save a match prediction to storage.
        
        Args:
            prediction (Prediction): Prediction object to save
            
        Returns:
            bool: Success status
        """
        try:
            if not isinstance(prediction, Prediction):
                raise TypeError("Expected Prediction object")
            
            # Create a unique ID if needed
            if not prediction.prediction_id:
                prediction.prediction_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Convert to dictionary
            prediction_dict = prediction.to_dict()
            
            # Save to file
            file_path = self._get_file_path('predictions', prediction.prediction_id)
            with open(file_path, 'w') as f:
                json.dump(prediction_dict, f, default=str, indent=4)
            
            logger.info(f"Prediction saved: {prediction.prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return False
    
    def save_player_prediction(self, prediction):
        """
        Save a player prediction to storage.
        
        Args:
            prediction (PlayerPrediction): PlayerPrediction object to save
            
        Returns:
            bool: Success status
        """
        try:
            if not isinstance(prediction, PlayerPrediction):
                raise TypeError("Expected PlayerPrediction object")
            
            # Create a unique ID if needed
            if not prediction.prediction_id:
                prediction.prediction_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Convert to dictionary
            prediction_dict = prediction.to_dict()
            
            # Save to file
            file_path = self._get_file_path('predictions', f"player_{prediction.prediction_id}")
            with open(file_path, 'w') as f:
                json.dump(prediction_dict, f, default=str, indent=4)
            
            logger.info(f"Player prediction saved: {prediction.prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving player prediction: {str(e)}")
            return False
    
    def get_match(self, match_id):
        """
        Get a match by ID.
        
        Args:
            match_id (str): Match ID
            
        Returns:
            Match: Match object if found, None otherwise
        """
        try:
            file_path = self._get_file_path('matches', match_id)
            
            if not os.path.exists(file_path):
                logger.warning(f"Match not found: {match_id}")
                return None
            
            with open(file_path, 'r') as f:
                match_dict = json.load(f)
            
            match = Match.from_dict(match_dict)
            logger.info(f"Match retrieved: {match_id}")
            return match
            
        except Exception as e:
            logger.error(f"Error getting match {match_id}: {str(e)}")
            return None
    
    def get_player(self, player_id):
        """
        Get a player by ID.
        
        Args:
            player_id (str): Player ID
            
        Returns:
            Player: Player object if found, None otherwise
        """
        try:
            file_path = self._get_file_path('players', player_id)
            
            if not os.path.exists(file_path):
                logger.warning(f"Player not found: {player_id}")
                return None
            
            with open(file_path, 'r') as f:
                player_dict = json.load(f)
            
            player = Player.from_dict(player_dict)
            logger.info(f"Player retrieved: {player_id}")
            return player
            
        except Exception as e:
            logger.error(f"Error getting player {player_id}: {str(e)}")
            return None
    
    def get_team(self, team_id):
        """
        Get a team by ID.
        
        Args:
            team_id (str): Team ID
            
        Returns:
            Team: Team object if found, None otherwise
        """
        try:
            file_path = self._get_file_path('teams', team_id)
            
            if not os.path.exists(file_path):
                logger.warning(f"Team not found: {team_id}")
                return None
            
            with open(file_path, 'r') as f:
                team_dict = json.load(f)
            
            team = Team.from_dict(team_dict)
            logger.info(f"Team retrieved: {team_id}")
            return team
            
        except Exception as e:
            logger.error(f"Error getting team {team_id}: {str(e)}")
            return None
    
    def get_prediction(self, prediction_id):
        """
        Get a prediction by ID.
        
        Args:
            prediction_id (str): Prediction ID
            
        Returns:
            Prediction: Prediction object if found, None otherwise
        """
        try:
            file_path = self._get_file_path('predictions', prediction_id)
            
            if not os.path.exists(file_path):
                logger.warning(f"Prediction not found: {prediction_id}")
                return None
            
            with open(file_path, 'r') as f:
                prediction_dict = json.load(f)
            
            prediction = Prediction.from_dict(prediction_dict)
            logger.info(f"Prediction retrieved: {prediction_id}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction {prediction_id}: {str(e)}")
            return None
    
    def get_all_teams(self):
        """
        Get all teams.
        
        Returns:
            list: List of Team objects
        """
        try:
            teams_dir = self._get_file_path('teams')
            team_files = [f for f in os.listdir(teams_dir) if f.endswith('.json')]
            
            teams = []
            for file_name in team_files:
                team_id = file_name.replace('.json', '')
                team = self.get_team(team_id)
                if team:
                    teams.append(team)
            
            logger.info(f"Retrieved {len(teams)} teams")
            return teams
            
        except Exception as e:
            logger.error(f"Error getting all teams: {str(e)}")
            return []
    
    def get_all_players(self, team=None, role=None):
        """
        Get all players, optionally filtered by team or role.
        
        Args:
            team (str, optional): Filter by team name
            role (str, optional): Filter by player role
            
        Returns:
            list: List of Player objects
        """
        try:
            players_dir = self._get_file_path('players')
            player_files = [f for f in os.listdir(players_dir) if f.endswith('.json')]
            
            players = []
            for file_name in player_files:
                player_id = file_name.replace('.json', '')
                player = self.get_player(player_id)
                
                if player:
                    # Apply filters if specified
                    if team and player.team != team:
                        continue
                    if role and player.role != role:
                        continue
                    
                    players.append(player)
            
            logger.info(f"Retrieved {len(players)} players")
            return players
            
        except Exception as e:
            logger.error(f"Error getting all players: {str(e)}")
            return []
    
    def get_matches(self, team=None, venue=None, limit=10):
        """
        Get matches, optionally filtered by team or venue.
        
        Args:
            team (str, optional): Filter by team name
            venue (str, optional): Filter by venue
            limit (int, optional): Maximum number of matches to return
            
        Returns:
            list: List of Match objects
        """
        try:
            matches_dir = self._get_file_path('matches')
            match_files = [f for f in os.listdir(matches_dir) if f.endswith('.json')]
            
            # Sort by filename (which should include timestamp)
            match_files.sort(reverse=True)
            
            matches = []
            for file_name in match_files:
                if len(matches) >= limit:
                    break
                    
                match_id = file_name.replace('.json', '')
                match = self.get_match(match_id)
                
                if match:
                    # Apply filters if specified
                    if team and match.team1 != team and match.team2 != team:
                        continue
                    if venue and match.venue != venue:
                        continue
                    
                    matches.append(match)
            
            logger.info(f"Retrieved {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error getting matches: {str(e)}")
            return []
    
    def get_prediction_history(self, team=None, limit=10):
        """
        Get prediction history, optionally filtered by team.
        
        Args:
            team (str, optional): Filter by team name
            limit (int, optional): Maximum number of predictions to return
            
        Returns:
            list: List of Prediction objects
        """
        try:
            predictions_dir = self._get_file_path('predictions')
            prediction_files = [f for f in os.listdir(predictions_dir) 
                              if f.endswith('.json') and not f.startswith('player_')]
            
            # Sort by filename (which should include timestamp)
            prediction_files.sort(reverse=True)
            
            predictions = []
            for file_name in prediction_files:
                if len(predictions) >= limit:
                    break
                    
                prediction_id = file_name.replace('.json', '')
                prediction = self.get_prediction(prediction_id)
                
                if prediction:
                    # Apply filters if specified
                    if team and prediction.team1 != team and prediction.team2 != team:
                        continue
                    
                    predictions.append(prediction)
            
            logger.info(f"Retrieved {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting prediction history: {str(e)}")
            return []
    
    def get_head_to_head(self, team1, team2):
        """
        Get head-to-head statistics between two teams.
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            
        Returns:
            dict: Head-to-head statistics
        """
        try:
            # Get all matches between the two teams
            matches_dir = self._get_file_path('matches')
            match_files = [f for f in os.listdir(matches_dir) if f.endswith('.json')]
            
            h2h_matches = []
            for file_name in match_files:
                match_id = file_name.replace('.json', '')
                match = self.get_match(match_id)
                
                if match and ((match.team1 == team1 and match.team2 == team2) or 
                             (match.team1 == team2 and match.team2 == team1)):
                    h2h_matches.append(match)
            
            # Calculate statistics
            total_matches = len(h2h_matches)
            team1_wins = sum(1 for m in h2h_matches if m.winner == team1)
            team2_wins = sum(1 for m in h2h_matches if m.winner == team2)
            no_results = total_matches - team1_wins - team2_wins
            
            # Calculate average scores
            team1_batting_first = [m for m in h2h_matches if 
                                  (m.team1 == team1 and m.toss_winner == team1 and m.toss_decision == 'Bat') or
                                  (m.team2 == team1 and m.toss_winner == team2 and m.toss_decision == 'Field')]
            
            team2_batting_first = [m for m in h2h_matches if 
                                  (m.team1 == team2 and m.toss_winner == team2 and m.toss_decision == 'Bat') or
                                  (m.team2 == team2 and m.toss_winner == team1 and m.toss_decision == 'Field')]
            
            # Get recent matches (last 5)
            recent_matches = sorted(h2h_matches, 
                                  key=lambda x: x.date if isinstance(x.date, datetime) 
                                  else datetime.strptime(str(x.date), "%Y-%m-%d"), 
                                  reverse=True)[:5]
            
            # Format recent match results
            recent_results = []
            for match in recent_matches:
                result = {
                    'match_id': match.match_id,
                    'date': match.date,
                    'venue': match.venue,
                    'winner': match.winner,
                    'team1_score': match.team1_score,
                    'team2_score': match.team2_score,
                    'result_summary': match.match_result
                }
                recent_results.append(result)
            
            # Create head-to-head stats
            h2h_stats = {
                'team1': team1,
                'team2': team2,
                'total_matches': total_matches,
                'team1_wins': team1_wins,
                'team2_wins': team2_wins,
                'no_results': no_results,
                'team1_win_percentage': (team1_wins / total_matches * 100) if total_matches > 0 else 0,
                'team2_win_percentage': (team2_wins / total_matches * 100) if total_matches > 0 else 0,
                'recent_matches': recent_results
            }
            
            logger.info(f"Retrieved head-to-head stats between {team1} and {team2}")
            return h2h_stats
            
        except Exception as e:
            logger.error(f"Error getting head-to-head stats: {str(e)}")
            return {
                'team1': team1,
                'team2': team2,
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'no_results': 0,
                'recent_matches': []
            }
