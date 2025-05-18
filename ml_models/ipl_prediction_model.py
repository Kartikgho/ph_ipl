import numpy as np
import pandas as pd
import random
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score

from config import FEATURE_COLUMNS, TRAIN_TEST_SPLIT_RATIO, RANDOM_STATE, MODEL_SAVE_PATH

class IPLPredictionModel:
    """
    Main prediction model class for IPL match outcomes.
    This model uses an ensemble approach to predict match winners, scores, and player performance.
    """
    
    def __init__(self):
        """Initialize the IPL prediction model"""
        self.match_winner_model = None
        self.team1_score_model = None
        self.team2_score_model = None
        self.player_performance_model = None
        self.feature_transformer = None
        self.is_trained = False
    
    def load_or_train_model(self):
        """
        Load pre-trained models if available, otherwise train new ones.
        
        In a real implementation, this would load saved model files or
        train models on historical match data.
        """
        try:
            # In a real implementation, we would try to load saved models
            # self.match_winner_model = pickle.load(open(f"{MODEL_SAVE_PATH}winner_model.pkl", "rb"))
            # self.team1_score_model = pickle.load(open(f"{MODEL_SAVE_PATH}team1_score_model.pkl", "rb"))
            # self.team2_score_model = pickle.load(open(f"{MODEL_SAVE_PATH}team2_score_model.pkl", "rb"))
            
            # For this implementation, we'll create simple models
            self._create_mock_models()
            self.is_trained = True
            
        except (FileNotFoundError, EOFError) as e:
            # If models don't exist, create and train new ones
            self._create_mock_models()
            self.is_trained = True
    
    def _create_mock_models(self):
        """
        Create mock models for demonstration purposes.
        
        In a real implementation, these would be trained on actual IPL data.
        """
        # Mock categorical and numerical feature columns
        categorical_features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 
                                'pitch_conditions', 'weather_conditions']
        numerical_features = ['team1_recent_form', 'team2_recent_form', 'head_to_head_ratio']
        
        # Create a mock feature transformer
        self.feature_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ]
        )
        
        # Create mock models
        self.match_winner_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.team1_score_model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
        self.team2_score_model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    
    def predict_match(self, match_data):
        """
        Predict the outcome of a match based on the provided data.
        
        Args:
            match_data (dict): Dictionary containing match information
        
        Returns:
            dict: Prediction results including winner, win probability, and scores
        """
        if not self.is_trained:
            raise Exception("Model not trained yet. Call load_or_train_model() first.")
        
        # In a real implementation, we would transform the input data and make predictions
        # using the trained models. For this demonstration, we'll create simulated predictions.
        
        team1 = match_data['team1']
        team2 = match_data['team2']
        toss_winner = match_data['toss_winner']
        toss_decision = match_data['toss_decision']
        venue = match_data['venue']
        pitch_conditions = match_data['pitch_conditions']
        
        # Simulate various factors affecting the prediction
        team1_strength = random.uniform(0.4, 0.6)
        team2_strength = random.uniform(0.4, 0.6)
        
        # Toss advantage (slight advantage to toss winner)
        toss_advantage = 0.05 if toss_winner == team1 else -0.05
        
        # Home advantage (if venue includes team name, small advantage)
        venue_lower = venue.lower()
        team1_lower = team1.lower()
        home_advantage = 0.03 if any(part in venue_lower for part in team1_lower.split()) else -0.03
        
        # Pitch condition factor (batting or bowling friendly)
        if pitch_conditions in ["Very Batting Friendly", "Batting Friendly"]:
            # Advantage to stronger batting team (random for demo)
            pitch_factor = 0.04 if random.random() > 0.5 else -0.04
        elif pitch_conditions in ["Very Bowling Friendly", "Bowling Friendly"]:
            # Advantage to stronger bowling team (random for demo)
            pitch_factor = 0.04 if random.random() > 0.5 else -0.04
        else:
            pitch_factor = 0
        
        # Calculate win probability
        win_probability_raw = 0.5 + team1_strength - team2_strength + toss_advantage + home_advantage + pitch_factor
        
        # Ensure probability is between 0.2 and 0.8 (to avoid extremely certain predictions)
        win_probability = max(0.2, min(0.8, win_probability_raw))
        
        # Determine winner based on probability
        if win_probability > 0.5:
            winner = team1
        else:
            winner = team2
            win_probability = 1 - win_probability
        
        # Generate score predictions
        batting_conditions_factor = 1.1 if pitch_conditions in ["Very Batting Friendly", "Batting Friendly"] else 0.9
        
        # Base scores in range typical for T20
        team1_base_score = random.randint(140, 180)
        team2_base_score = random.randint(140, 180)
        
        # Adjust scores based on conditions
        team1_score = int(team1_base_score * batting_conditions_factor)
        team2_score = int(team2_base_score * batting_conditions_factor)
        
        # Ensure winning team has higher score
        if winner == team1 and team1_score <= team2_score:
            team1_score = team2_score + random.randint(1, 20)
        elif winner == team2 and team2_score <= team1_score:
            team2_score = team1_score + random.randint(1, 20)
        
        # Return prediction results
        return {
            'winner': winner,
            'win_probability': win_probability,
            'team1_score': team1_score,
            'team2_score': team2_score
        }
    
    def predict_player_performance(self, player_data):
        """
        Predict player performance based on provided data.
        
        Args:
            player_data (dict): Dictionary containing player and match information
        
        Returns:
            dict: Predicted performance metrics
        """
        # In a real implementation, this would use a trained model to predict
        # player-specific metrics based on historical data and match conditions.
        
        player_role = player_data.get('role', 'Batsman')
        
        performance = {}
        
        if player_role in ['Batsman', 'All-rounder', 'Wicket-keeper Batsman']:
            # Predict batting performance
            performance['predicted_runs'] = random.randint(15, 60)
            performance['predicted_strike_rate'] = random.uniform(120, 170)
            
            # Add confidence intervals
            performance['runs_lower'] = max(0, performance['predicted_runs'] - 15)
            performance['runs_upper'] = performance['predicted_runs'] + 15
            
        if player_role in ['Bowler', 'All-rounder']:
            # Predict bowling performance
            performance['predicted_wickets'] = random.randint(0, 3)
            performance['predicted_economy'] = random.uniform(7, 11)
            
            # Add confidence intervals
            performance['economy_lower'] = max(5, performance['predicted_economy'] - 1.5)
            performance['economy_upper'] = performance['predicted_economy'] + 1.5
        
        # In a real model, we would include many more metrics and their confidence intervals
        return performance
