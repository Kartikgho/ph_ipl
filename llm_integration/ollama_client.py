"""
Ollama LLM client for IPL prediction system.
This module handles interactions with the local LLM for prediction explanations.
"""

import os
import requests
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ollama_client.log'
)
logger = logging.getLogger('ollama_client')

# Configuration
OLLAMA_API_ENDPOINT = os.getenv("OLLAMA_API_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

def get_explanation(prompt, model=None, max_tokens=None):
    """
    Get an explanation from the Ollama LLM.
    
    Args:
        prompt (str): The prompt to send to the LLM
        model (str, optional): Model to use, defaults to OLLAMA_MODEL
        max_tokens (int, optional): Maximum tokens to generate, defaults to MAX_TOKENS
        
    Returns:
        str: Generated explanation text
        
    Raises:
        Exception: If API request fails
    """
    if model is None:
        model = OLLAMA_MODEL
    
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    
    logger.info(f"Requesting explanation using model: {model}")
    
    try:
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens
        }
        
        # Make request to Ollama API
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        explanation = result.get('response', '')
        
        logger.info(f"Received explanation ({len(explanation)} chars)")
        return explanation
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        
        # Return a reasonable explanation in case of Ollama connection issues
        return """
        Based on my cricket analysis expertise, here are likely factors influencing this prediction:
        
        • Recent team form will be significant - teams with momentum often maintain it
        • Head-to-head history typically predicts future performance patterns
        • Venue statistics matter - teams perform differently at different grounds
        • Toss advantage varies by venue but generally impacts strategy
        • Key player availability and current form would affect team strength
        
        While I can't access the live model now, these fundamental cricket analytics principles generally drive IPL match predictions.
        """
    except Exception as e:
        logger.error(f"Error processing explanation: {str(e)}")
        raise

def get_player_analysis(player_name, stats, opponent=None, venue=None):
    """
    Get a player analysis from the Ollama LLM.
    
    Args:
        player_name (str): Player name
        stats (dict): Player statistics
        opponent (str, optional): Opponent team
        venue (str, optional): Match venue
        
    Returns:
        str: Generated player analysis
    """
    # Create a prompt for player analysis
    prompt = f"""
    As a cricket analytics expert, analyze {player_name}'s IPL performance and prospects.
    
    Player stats:
    - Batting average: {stats.get('batting_avg', 'N/A')}
    - Strike rate: {stats.get('strike_rate', 'N/A')}
    - Bowling average: {stats.get('bowling_avg', 'N/A')}
    - Economy rate: {stats.get('economy_rate', 'N/A')}
    
    Recent form summary:
    {stats.get('recent_form', 'N/A')}
    """
    
    if opponent:
        prompt += f"\nUpcoming match against: {opponent}"
    
    if venue:
        prompt += f"\nVenue: {venue}"
    
    prompt += """
    
    Provide a brief analysis of the player's form, strengths, weaknesses, and potential performance.
    Focus on concrete statistical insights rather than general statements.
    Limit your response to 4-5 concise bullet points.
    """
    
    try:
        return get_explanation(prompt)
    except Exception as e:
        logger.error(f"Error getting player analysis: {str(e)}")
        return f"Player analysis unavailable for {player_name} at this time."

def get_team_analysis(team_name, stats, opponent=None, venue=None):
    """
    Get a team analysis from the Ollama LLM.
    
    Args:
        team_name (str): Team name
        stats (dict): Team statistics
        opponent (str, optional): Opponent team
        venue (str, optional): Match venue
        
    Returns:
        str: Generated team analysis
    """
    # Create a prompt for team analysis
    prompt = f"""
    As a cricket analytics expert, analyze {team_name}'s IPL performance and prospects.
    
    Team stats:
    - Win percentage: {stats.get('win_percentage', 'N/A')}
    - Recent form: {stats.get('recent_form', 'N/A')}
    - Net run rate: {stats.get('net_run_rate', 'N/A')}
    - Titles won: {stats.get('titles', 'N/A')}
    
    Key players:
    {stats.get('key_players', 'N/A')}
    """
    
    if opponent:
        prompt += f"\nUpcoming match against: {opponent}"
        
        if 'head_to_head' in stats:
            prompt += f"\nHead-to-head record: {stats['head_to_head']}"
    
    if venue:
        prompt += f"\nVenue: {venue}"
        
        if 'venue_record' in stats:
            prompt += f"\nRecord at this venue: {stats['venue_record']}"
    
    prompt += """
    
    Provide a brief analysis of the team's strengths, weaknesses, and potential performance.
    Focus on concrete statistical insights rather than general statements.
    Limit your response to 4-5 concise bullet points.
    """
    
    try:
        return get_explanation(prompt)
    except Exception as e:
        logger.error(f"Error getting team analysis: {str(e)}")
        return f"Team analysis unavailable for {team_name} at this time."
