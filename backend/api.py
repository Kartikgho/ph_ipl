"""
API endpoints for the IPL prediction system.
This module handles the server-side API logic for predictions and data retrieval.
"""

import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from ml_models.ipl_prediction_model import IPLPredictionModel
from backend.models import Match, Player, Team, Prediction, PlayerPrediction
from backend.db_utils import DataManager
from llm_integration.ollama_client import get_explanation
from llm_integration.prompt_engineering import generate_match_explanation_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api.log'
)
logger = logging.getLogger('api')

# Initialize Flask app
app = Flask(__name__)

# Initialize prediction model
prediction_model = IPLPredictionModel()
try:
    prediction_model.load_or_train_model()
    logger.info("Prediction model loaded successfully")
except Exception as e:
    logger.error(f"Error loading prediction model: {str(e)}")

# Initialize data manager
data_manager = DataManager()

@app.route('/api/predict/match', methods=['POST'])
def predict_match():
    """
    API endpoint to predict match outcome.
    
    Expects JSON with match details:
    {
        "team1": "Team A",
        "team2": "Team B",
        "venue": "Stadium",
        "toss_winner": "Team A",
        "toss_decision": "Bat",
        "pitch_conditions": "Batting Friendly",
        "weather_conditions": "Clear"
    }
    
    Returns:
        JSON with prediction results
    """
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate prediction
        prediction_result = prediction_model.predict_match(data)
        
        # Get LLM explanation
        try:
            prompt = generate_match_explanation_prompt(
                data['team1'], data['team2'], 
                prediction_result['winner'], prediction_result['win_probability'],
                prediction_result['team1_score'], prediction_result['team2_score'],
                data['venue'], data['toss_winner'], data['toss_decision'],
                data.get('pitch_conditions', 'Unknown'), data.get('weather_conditions', 'Unknown')
            )
            
            explanation = get_explanation(prompt)
            prediction_result['explanation'] = explanation
        except Exception as e:
            logger.error(f"Error getting LLM explanation: {str(e)}")
            prediction_result['explanation'] = "Explanation not available at this time."
        
        # Save prediction to database
        try:
            prediction = Prediction(
                prediction_id=datetime.now().strftime("%Y%m%d%H%M%S"),
                match_id=None,  # No specific match ID for hypothetical predictions
                match_date=datetime.now(),
                team1=data['team1'],
                team2=data['team2'],
                venue=data['venue']
            )
            
            prediction.set_prediction(
                prediction_result['winner'],
                prediction_result['win_probability'],
                prediction_result['team1_score'],
                prediction_result['team2_score']
            )
            
            prediction.set_reasoning(prediction_result.get('explanation', ''))
            
            # Add key factors (would be more sophisticated in a real system)
            prediction.add_key_factor("Head-to-Head Record", 0.25)
            prediction.add_key_factor("Recent Form", 0.2)
            prediction.add_key_factor("Home Advantage", 0.15)
            prediction.add_key_factor("Toss Result", 0.1)
            
            data_manager.save_prediction(prediction)
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
        
        return jsonify(prediction_result), 200
        
    except Exception as e:
        logger.error(f"Error in match prediction API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/player', methods=['POST'])
def predict_player_performance():
    """
    API endpoint to predict player performance.
    
    Expects JSON with player and match details:
    {
        "player_id": "123",
        "player_name": "Player Name",
        "team": "Team A",
        "opponent": "Team B",
        "venue": "Stadium",
        "role": "Batsman"
    }
    
    Returns:
        JSON with prediction results
    """
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['player_name', 'team', 'opponent', 'role']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Generate prediction
        player_prediction = prediction_model.predict_player_performance(data)
        
        # Save prediction to database
        try:
            pred = PlayerPrediction(
                prediction_id=datetime.now().strftime("%Y%m%d%H%M%S"),
                match_id=None,  # No specific match ID for hypothetical predictions
                player_id=data.get('player_id'),
                player_name=data['player_name'],
                team=data['team'],
                opponent=data['opponent']
            )
            
            # Set appropriate predictions based on role
            if data['role'] in ['Batsman', 'All-rounder', 'Wicket-keeper Batsman']:
                pred.set_batting_prediction(
                    player_prediction.get('predicted_runs'),
                    player_prediction.get('predicted_strike_rate'),
                    [player_prediction.get('runs_lower', 0), player_prediction.get('runs_upper', 0)]
                )
            
            if data['role'] in ['Bowler', 'All-rounder']:
                pred.set_bowling_prediction(
                    player_prediction.get('predicted_wickets'),
                    player_prediction.get('predicted_economy'),
                    [player_prediction.get('economy_lower', 0), player_prediction.get('economy_upper', 0)]
                )
            
            # Add key factors (would be more sophisticated in a real system)
            pred.add_key_factor("Recent Form", 0.3)
            pred.add_key_factor("Head-to-Head vs Opponent", 0.25)
            pred.add_key_factor("Venue Performance", 0.2)
            
            data_manager.save_player_prediction(pred)
        except Exception as e:
            logger.error(f"Error saving player prediction: {str(e)}")
        
        return jsonify(player_prediction), 200
        
    except Exception as e:
        logger.error(f"Error in player prediction API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """
    API endpoint to get all teams.
    
    Returns:
        JSON with team data
    """
    try:
        teams = data_manager.get_all_teams()
        return jsonify([team.to_dict() for team in teams]), 200
    except Exception as e:
        logger.error(f"Error getting teams: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams/<team_id>', methods=['GET'])
def get_team(team_id):
    """
    API endpoint to get team details.
    
    Args:
        team_id (str): ID of team to retrieve
    
    Returns:
        JSON with team data
    """
    try:
        team = data_manager.get_team(team_id)
        if team:
            return jsonify(team.to_dict()), 200
        else:
            return jsonify({"error": "Team not found"}), 404
    except Exception as e:
        logger.error(f"Error getting team {team_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/players', methods=['GET'])
def get_players():
    """
    API endpoint to get all players.
    
    Query parameters:
        team (str, optional): Filter by team
        role (str, optional): Filter by role
    
    Returns:
        JSON with player data
    """
    try:
        team = request.args.get('team')
        role = request.args.get('role')
        
        players = data_manager.get_all_players(team=team, role=role)
        return jsonify([player.to_dict() for player in players]), 200
    except Exception as e:
        logger.error(f"Error getting players: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/players/<player_id>', methods=['GET'])
def get_player(player_id):
    """
    API endpoint to get player details.
    
    Args:
        player_id (str): ID of player to retrieve
    
    Returns:
        JSON with player data
    """
    try:
        player = data_manager.get_player(player_id)
        if player:
            return jsonify(player.to_dict()), 200
        else:
            return jsonify({"error": "Player not found"}), 404
    except Exception as e:
        logger.error(f"Error getting player {player_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """
    API endpoint to get match data.
    
    Query parameters:
        team (str, optional): Filter by team
        venue (str, optional): Filter by venue
        limit (int, optional): Limit number of matches returned
    
    Returns:
        JSON with match data
    """
    try:
        team = request.args.get('team')
        venue = request.args.get('venue')
        limit = request.args.get('limit', default=10, type=int)
        
        matches = data_manager.get_matches(team=team, venue=venue, limit=limit)
        return jsonify([match.to_dict() for match in matches]), 200
    except Exception as e:
        logger.error(f"Error getting matches: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/history', methods=['GET'])
def get_prediction_history():
    """
    API endpoint to get prediction history.
    
    Query parameters:
        team (str, optional): Filter by team
        limit (int, optional): Limit number of predictions returned
    
    Returns:
        JSON with prediction history
    """
    try:
        team = request.args.get('team')
        limit = request.args.get('limit', default=10, type=int)
        
        predictions = data_manager.get_prediction_history(team=team, limit=limit)
        return jsonify([pred.to_dict() for pred in predictions]), 200
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/head-to-head', methods=['GET'])
def get_head_to_head():
    """
    API endpoint to get head-to-head statistics.
    
    Query parameters:
        team1 (str): First team
        team2 (str): Second team
    
    Returns:
        JSON with head-to-head statistics
    """
    try:
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')
        
        if not team1 or not team2:
            return jsonify({"error": "Both team1 and team2 parameters are required"}), 400
        
        h2h_stats = data_manager.get_head_to_head(team1, team2)
        return jsonify(h2h_stats), 200
    except Exception as e:
        logger.error(f"Error getting head-to-head stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Start the server if run directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
