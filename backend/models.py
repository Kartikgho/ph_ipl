"""
Database models for IPL prediction system.
This module defines the data structures used in the application.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='models.log'
)
logger = logging.getLogger('models')

class Match:
    """Model for cricket match data"""
    
    def __init__(self, match_id=None, team1=None, team2=None, venue=None, date=None, 
                 toss_winner=None, toss_decision=None, winner=None, 
                 team1_score=None, team2_score=None, match_result=None):
        """
        Initialize a Match object.
        
        Args:
            match_id (int/str): Unique identifier for the match
            team1 (str): Name of first team
            team2 (str): Name of second team
            venue (str): Match venue
            date (datetime/str): Match date
            toss_winner (str): Team that won the toss
            toss_decision (str): Decision made by toss winner (bat/field)
            winner (str): Team that won the match
            team1_score (str): Score of team1 (e.g., "165/6")
            team2_score (str): Score of team2 (e.g., "142/8")
            match_result (str): Description of match result
        """
        self.match_id = match_id
        self.team1 = team1
        self.team2 = team2
        self.venue = venue
        self.date = date
        self.toss_winner = toss_winner
        self.toss_decision = toss_decision
        self.winner = winner
        self.team1_score = team1_score
        self.team2_score = team2_score
        self.match_result = match_result
        
        # Parse scores into runs and wickets if provided
        self.team1_runs = None
        self.team1_wickets = None
        self.team2_runs = None
        self.team2_wickets = None
        
        if team1_score and '/' in team1_score:
            parts = team1_score.split('/')
            if len(parts) == 2:
                try:
                    self.team1_runs = int(parts[0])
                    self.team1_wickets = int(parts[1])
                except ValueError:
                    pass
        
        if team2_score and '/' in team2_score:
            parts = team2_score.split('/')
            if len(parts) == 2:
                try:
                    self.team2_runs = int(parts[0])
                    self.team2_wickets = int(parts[1])
                except ValueError:
                    pass
    
    @classmethod
    def from_dict(cls, data):
        """Create a Match object from a dictionary"""
        return cls(
            match_id=data.get('match_id'),
            team1=data.get('team1'),
            team2=data.get('team2'),
            venue=data.get('venue'),
            date=data.get('date'),
            toss_winner=data.get('toss_winner'),
            toss_decision=data.get('toss_decision'),
            winner=data.get('winner'),
            team1_score=data.get('team1_score'),
            team2_score=data.get('team2_score'),
            match_result=data.get('match_result')
        )
    
    def to_dict(self):
        """Convert Match object to dictionary"""
        return {
            'match_id': self.match_id,
            'team1': self.team1,
            'team2': self.team2,
            'venue': self.venue,
            'date': self.date,
            'toss_winner': self.toss_winner,
            'toss_decision': self.toss_decision,
            'winner': self.winner,
            'team1_score': self.team1_score,
            'team2_score': self.team2_score,
            'match_result': self.match_result,
            'team1_runs': self.team1_runs,
            'team1_wickets': self.team1_wickets,
            'team2_runs': self.team2_runs,
            'team2_wickets': self.team2_wickets
        }


class Player:
    """Model for cricket player data"""
    
    def __init__(self, player_id=None, name=None, team=None, role=None):
        """
        Initialize a Player object.
        
        Args:
            player_id (int/str): Unique identifier for the player
            name (str): Player's name
            team (str): Player's team
            role (str): Player's role (Batsman, Bowler, All-rounder, Wicket-keeper)
        """
        self.player_id = player_id
        self.name = name
        self.team = team
        self.role = role
        
        # Performance statistics
        self.batting_stats = {}
        self.bowling_stats = {}
        self.recent_form = []
    
    def update_batting_stats(self, matches=None, innings=None, runs=None, 
                            average=None, strike_rate=None, fifties=None, 
                            hundreds=None, fours=None, sixes=None):
        """Update batting statistics for the player"""
        self.batting_stats = {
            'matches': matches,
            'innings': innings,
            'runs': runs,
            'average': average,
            'strike_rate': strike_rate,
            'fifties': fifties,
            'hundreds': hundreds,
            'fours': fours,
            'sixes': sixes
        }
    
    def update_bowling_stats(self, matches=None, innings=None, overs=None,
                           wickets=None, economy=None, average=None, 
                           strike_rate=None, best_figures=None):
        """Update bowling statistics for the player"""
        self.bowling_stats = {
            'matches': matches,
            'innings': innings,
            'overs': overs,
            'wickets': wickets,
            'economy': economy,
            'average': average,
            'strike_rate': strike_rate,
            'best_figures': best_figures
        }
    
    def add_recent_performance(self, match_id, date, opponent, runs=None, 
                              balls=None, wickets=None, overs=None, 
                              economy=None):
        """Add a recent performance entry for the player"""
        performance = {
            'match_id': match_id,
            'date': date,
            'opponent': opponent,
            'runs': runs,
            'balls': balls,
            'wickets': wickets,
            'overs': overs,
            'economy': economy
        }
        
        self.recent_form.append(performance)
        
        # Keep only the most recent 10 performances
        if len(self.recent_form) > 10:
            self.recent_form = sorted(self.recent_form, 
                                     key=lambda x: x['date'] if isinstance(x['date'], datetime) 
                                     else datetime.strptime(str(x['date']), "%Y-%m-%d"), 
                                     reverse=True)[:10]
    
    @classmethod
    def from_dict(cls, data):
        """Create a Player object from a dictionary"""
        player = cls(
            player_id=data.get('player_id'),
            name=data.get('name'),
            team=data.get('team'),
            role=data.get('role')
        )
        
        # Add batting stats if present
        if 'batting_stats' in data:
            player.batting_stats = data['batting_stats']
        
        # Add bowling stats if present
        if 'bowling_stats' in data:
            player.bowling_stats = data['bowling_stats']
        
        # Add recent form if present
        if 'recent_form' in data:
            player.recent_form = data['recent_form']
        
        return player
    
    def to_dict(self):
        """Convert Player object to dictionary"""
        return {
            'player_id': self.player_id,
            'name': self.name,
            'team': self.team,
            'role': self.role,
            'batting_stats': self.batting_stats,
            'bowling_stats': self.bowling_stats,
            'recent_form': self.recent_form
        }


class Team:
    """Model for cricket team data"""
    
    def __init__(self, team_id=None, name=None, short_name=None):
        """
        Initialize a Team object.
        
        Args:
            team_id (int/str): Unique identifier for the team
            name (str): Full team name
            short_name (str): Short team name or abbreviation
        """
        self.team_id = team_id
        self.name = name
        self.short_name = short_name
        
        # Team statistics
        self.stats = {}
        self.players = []
        self.recent_matches = []
    
    def update_stats(self, matches=None, wins=None, losses=None, no_results=None,
                    points=None, titles=None, win_percentage=None, 
                    net_run_rate=None):
        """Update team statistics"""
        self.stats = {
            'matches': matches,
            'wins': wins,
            'losses': losses,
            'no_results': no_results,
            'points': points,
            'titles': titles,
            'win_percentage': win_percentage,
            'net_run_rate': net_run_rate
        }
    
    def add_player(self, player_id, name, role):
        """Add a player to the team"""
        self.players.append({
            'player_id': player_id,
            'name': name,
            'role': role
        })
    
    def add_recent_match(self, match_id, date, opponent, venue,
                        result, score, opponent_score):
        """Add a recent match result for the team"""
        match = {
            'match_id': match_id,
            'date': date,
            'opponent': opponent,
            'venue': venue,
            'result': result,
            'score': score,
            'opponent_score': opponent_score
        }
        
        self.recent_matches.append(match)
        
        # Keep only the most recent 10 matches
        if len(self.recent_matches) > 10:
            self.recent_matches = sorted(self.recent_matches, 
                                       key=lambda x: x['date'] if isinstance(x['date'], datetime) 
                                       else datetime.strptime(str(x['date']), "%Y-%m-%d"), 
                                       reverse=True)[:10]
    
    @classmethod
    def from_dict(cls, data):
        """Create a Team object from a dictionary"""
        team = cls(
            team_id=data.get('team_id'),
            name=data.get('name'),
            short_name=data.get('short_name')
        )
        
        # Add stats if present
        if 'stats' in data:
            team.stats = data['stats']
        
        # Add players if present
        if 'players' in data:
            team.players = data['players']
        
        # Add recent matches if present
        if 'recent_matches' in data:
            team.recent_matches = data['recent_matches']
        
        return team
    
    def to_dict(self):
        """Convert Team object to dictionary"""
        return {
            'team_id': self.team_id,
            'name': self.name,
            'short_name': self.short_name,
            'stats': self.stats,
            'players': self.players,
            'recent_matches': self.recent_matches
        }


class Prediction:
    """Model for match prediction data"""
    
    def __init__(self, prediction_id=None, match_id=None, match_date=None,
                team1=None, team2=None, venue=None):
        """
        Initialize a Prediction object.
        
        Args:
            prediction_id (int/str): Unique identifier for the prediction
            match_id (int/str): ID of the match being predicted
            match_date (datetime/str): Date of the match
            team1 (str): Name of first team
            team2 (str): Name of second team
            venue (str): Match venue
        """
        self.prediction_id = prediction_id
        self.match_id = match_id
        self.match_date = match_date
        self.team1 = team1
        self.team2 = team2
        self.venue = venue
        
        # Prediction results
        self.predicted_winner = None
        self.win_probability = None
        self.team1_predicted_score = None
        self.team2_predicted_score = None
        
        # Reasoning and additional info
        self.key_factors = []
        self.confidence_level = None
        self.prediction_timestamp = datetime.now()
        self.llm_reasoning = None
    
    def set_prediction(self, winner, probability, team1_score, team2_score):
        """Set the main prediction results"""
        self.predicted_winner = winner
        self.win_probability = probability
        self.team1_predicted_score = team1_score
        self.team2_predicted_score = team2_score
    
    def add_key_factor(self, factor, importance):
        """Add a key factor influencing the prediction"""
        self.key_factors.append({
            'factor': factor,
            'importance': importance
        })
    
    def set_reasoning(self, reasoning_text):
        """Set the LLM reasoning text"""
        self.llm_reasoning = reasoning_text
    
    @classmethod
    def from_dict(cls, data):
        """Create a Prediction object from a dictionary"""
        prediction = cls(
            prediction_id=data.get('prediction_id'),
            match_id=data.get('match_id'),
            match_date=data.get('match_date'),
            team1=data.get('team1'),
            team2=data.get('team2'),
            venue=data.get('venue')
        )
        
        # Add prediction results if present
        prediction.predicted_winner = data.get('predicted_winner')
        prediction.win_probability = data.get('win_probability')
        prediction.team1_predicted_score = data.get('team1_predicted_score')
        prediction.team2_predicted_score = data.get('team2_predicted_score')
        
        # Add reasoning and additional info if present
        prediction.key_factors = data.get('key_factors', [])
        prediction.confidence_level = data.get('confidence_level')
        prediction.prediction_timestamp = data.get('prediction_timestamp', datetime.now())
        prediction.llm_reasoning = data.get('llm_reasoning')
        
        return prediction
    
    def to_dict(self):
        """Convert Prediction object to dictionary"""
        return {
            'prediction_id': self.prediction_id,
            'match_id': self.match_id,
            'match_date': self.match_date,
            'team1': self.team1,
            'team2': self.team2,
            'venue': self.venue,
            'predicted_winner': self.predicted_winner,
            'win_probability': self.win_probability,
            'team1_predicted_score': self.team1_predicted_score,
            'team2_predicted_score': self.team2_predicted_score,
            'key_factors': self.key_factors,
            'confidence_level': self.confidence_level,
            'prediction_timestamp': self.prediction_timestamp,
            'llm_reasoning': self.llm_reasoning
        }


class PlayerPrediction:
    """Model for player performance prediction data"""
    
    def __init__(self, prediction_id=None, match_id=None, player_id=None,
                player_name=None, team=None, opponent=None):
        """
        Initialize a PlayerPrediction object.
        
        Args:
            prediction_id (int/str): Unique identifier for the prediction
            match_id (int/str): ID of the match being predicted
            player_id (int/str): ID of the player
            player_name (str): Name of the player
            team (str): Player's team
            opponent (str): Opponent team
        """
        self.prediction_id = prediction_id
        self.match_id = match_id
        self.player_id = player_id
        self.player_name = player_name
        self.team = team
        self.opponent = opponent
        
        # Batting prediction
        self.predicted_runs = None
        self.predicted_strike_rate = None
        self.runs_confidence_interval = None
        
        # Bowling prediction
        self.predicted_wickets = None
        self.predicted_economy = None
        self.wickets_confidence_interval = None
        
        # Reasoning and additional info
        self.key_factors = []
        self.confidence_level = None
        self.prediction_timestamp = datetime.now()
    
    def set_batting_prediction(self, runs, strike_rate, confidence_interval=None):
        """Set the batting prediction"""
        self.predicted_runs = runs
        self.predicted_strike_rate = strike_rate
        self.runs_confidence_interval = confidence_interval or [max(0, runs-15), runs+15]
    
    def set_bowling_prediction(self, wickets, economy, confidence_interval=None):
        """Set the bowling prediction"""
        self.predicted_wickets = wickets
        self.predicted_economy = economy
        self.wickets_confidence_interval = confidence_interval or [max(0, wickets-1), wickets+1]
    
    def add_key_factor(self, factor, importance):
        """Add a key factor influencing the prediction"""
        self.key_factors.append({
            'factor': factor,
            'importance': importance
        })
    
    @classmethod
    def from_dict(cls, data):
        """Create a PlayerPrediction object from a dictionary"""
        prediction = cls(
            prediction_id=data.get('prediction_id'),
            match_id=data.get('match_id'),
            player_id=data.get('player_id'),
            player_name=data.get('player_name'),
            team=data.get('team'),
            opponent=data.get('opponent')
        )
        
        # Add batting prediction if present
        prediction.predicted_runs = data.get('predicted_runs')
        prediction.predicted_strike_rate = data.get('predicted_strike_rate')
        prediction.runs_confidence_interval = data.get('runs_confidence_interval')
        
        # Add bowling prediction if present
        prediction.predicted_wickets = data.get('predicted_wickets')
        prediction.predicted_economy = data.get('predicted_economy')
        prediction.wickets_confidence_interval = data.get('wickets_confidence_interval')
        
        # Add reasoning and additional info if present
        prediction.key_factors = data.get('key_factors', [])
        prediction.confidence_level = data.get('confidence_level')
        prediction.prediction_timestamp = data.get('prediction_timestamp', datetime.now())
        
        return prediction
    
    def to_dict(self):
        """Convert PlayerPrediction object to dictionary"""
        return {
            'prediction_id': self.prediction_id,
            'match_id': self.match_id,
            'player_id': self.player_id,
            'player_name': self.player_name,
            'team': self.team,
            'opponent': self.opponent,
            'predicted_runs': self.predicted_runs,
            'predicted_strike_rate': self.predicted_strike_rate,
            'runs_confidence_interval': self.runs_confidence_interval,
            'predicted_wickets': self.predicted_wickets,
            'predicted_economy': self.predicted_economy,
            'wickets_confidence_interval': self.wickets_confidence_interval,
            'key_factors': self.key_factors,
            'confidence_level': self.confidence_level,
            'prediction_timestamp': self.prediction_timestamp
        }
