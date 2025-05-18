"""
Prompt engineering for IPL prediction system.
This module creates optimized prompts for the LLM to generate insights.
"""

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='prompt_engineering.log'
)
logger = logging.getLogger('prompt_engineering')

def generate_match_explanation_prompt(team1, team2, winner, win_prob, team1_score, team2_score, 
                                     venue, toss_winner, toss_decision, pitch_conditions, weather_conditions):
    """
    Generate a prompt for match prediction explanation.
    
    Args:
        team1 (str): Name of first team
        team2 (str): Name of second team
        winner (str): Predicted winner
        win_prob (float): Win probability
        team1_score (int): Predicted score for team1
        team2_score (int): Predicted score for team2
        venue (str): Match venue
        toss_winner (str): Team that won the toss
        toss_decision (str): Decision made by toss winner (bat/field)
        pitch_conditions (str): Description of pitch conditions
        weather_conditions (str): Description of weather conditions
        
    Returns:
        str: Formatted prompt for the LLM
    """
    logger.info(f"Generating match explanation prompt for {team1} vs {team2}")
    
    prompt = f"""
    You are a cricket analytics expert specializing in IPL predictions.
    
    Based on the following match information:
    - Team 1: {team1}
    - Team 2: {team2}
    - Venue: {venue}
    - Toss Winner: {toss_winner} (chose to {toss_decision})
    - Pitch Conditions: {pitch_conditions}
    - Weather Conditions: {weather_conditions}
    
    Our statistical model has predicted:
    - Match Winner: {winner} (Win Probability: {win_prob:.2%})
    - Predicted Score for {team1}: {team1_score}
    - Predicted Score for {team2}: {team2_score}
    
    Provide a concise, expert explanation of why {winner} is predicted to win this match.
    Focus on key factors like recent team form, head-to-head record, venue statistics,
    impact of toss, and how pitch/weather conditions might influence the outcome.
    
    Keep your explanation factual, balanced, and limited to 4-5 key points in bullet form.
    """
    
    return prompt

def generate_player_performance_prompt(player_name, team, opponent, venue, role,
                                     recent_form, predicted_runs=None, predicted_wickets=None):
    """
    Generate a prompt for player performance prediction explanation.
    
    Args:
        player_name (str): Name of the player
        team (str): Player's team
        opponent (str): Opponent team
        venue (str): Match venue
        role (str): Player's role
        recent_form (str): Description of player's recent form
        predicted_runs (int, optional): Predicted runs
        predicted_wickets (int, optional): Predicted wickets
        
    Returns:
        str: Formatted prompt for the LLM
    """
    logger.info(f"Generating player performance prompt for {player_name}")
    
    # Create base prompt
    prompt = f"""
    You are a cricket analytics expert specializing in player performance predictions for IPL.
    
    Based on the following player and match information:
    - Player: {player_name}
    - Team: {team}
    - Role: {role}
    - Opponent: {opponent}
    - Venue: {venue}
    - Recent Form: {recent_form}
    """
    
    # Add prediction information if available
    if predicted_runs is not None and role in ["Batsman", "All-rounder", "Wicket-keeper Batsman"]:
        prompt += f"\nOur statistical model has predicted {player_name} will score around {predicted_runs} runs."
    
    if predicted_wickets is not None and role in ["Bowler", "All-rounder"]:
        prompt += f"\nOur statistical model has predicted {player_name} will take around {predicted_wickets} wickets."
    
    # Add the instruction
    prompt += f"""
    
    Provide a concise, expert explanation of the factors influencing {player_name}'s predicted performance in this match.
    Focus on key aspects like:
    - The player's historical performance against this opponent
    - Performance at this venue
    - Current form and confidence
    - Match-up against specific bowlers/batsmen
    - Impact of pitch and weather conditions
    
    Keep your explanation factual, data-driven, and limited to 3-4 key points in bullet form.
    """
    
    return prompt

def generate_team_analysis_prompt(team, matches_played, win_percentage, 
                                key_players, recent_form, team_strengths=None, team_weaknesses=None):
    """
    Generate a prompt for team analysis.
    
    Args:
        team (str): Team name
        matches_played (int): Total matches played
        win_percentage (float): Win percentage
        key_players (list): List of key players
        recent_form (str): Description of recent form
        team_strengths (list, optional): List of team strengths
        team_weaknesses (list, optional): List of team weaknesses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    logger.info(f"Generating team analysis prompt for {team}")
    
    # Format key players as string
    if isinstance(key_players, list):
        key_players_str = ", ".join(key_players)
    else:
        key_players_str = str(key_players)
    
    # Create base prompt
    prompt = f"""
    You are a cricket analytics expert specializing in IPL team analysis.
    
    Based on the following team information for {team}:
    - Matches Played: {matches_played}
    - Win Percentage: {win_percentage:.2%}
    - Key Players: {key_players_str}
    - Recent Form: {recent_form}
    """
    
    # Add strengths and weaknesses if available
    if team_strengths:
        if isinstance(team_strengths, list):
            strengths_str = ", ".join(team_strengths)
        else:
            strengths_str = str(team_strengths)
        prompt += f"- Key Strengths: {strengths_str}\n"
    
    if team_weaknesses:
        if isinstance(team_weaknesses, list):
            weaknesses_str = ", ".join(team_weaknesses)
        else:
            weaknesses_str = str(team_weaknesses)
        prompt += f"- Key Weaknesses: {weaknesses_str}\n"
    
    # Add the instruction
    prompt += f"""
    
    Provide a comprehensive analysis of {team}'s performance and prospects in the current IPL season.
    Include:
    1. An evaluation of their current form and trajectory
    2. Analysis of their key strengths and how they leverage them
    3. Assessment of their vulnerabilities and how opponents exploit them
    4. Impact of their key players on team performance
    5. Strategic recommendations for improvement
    
    Keep your analysis insightful, data-driven, and balanced. Format as 5-6 paragraphs with clear headings.
    """
    
    return prompt

def generate_head_to_head_prompt(team1, team2, total_matches, team1_wins, team2_wins,
                               recent_matches, venue=None):
    """
    Generate a prompt for head-to-head analysis.
    
    Args:
        team1 (str): First team name
        team2 (str): Second team name
        total_matches (int): Total matches played between teams
        team1_wins (int): Number of wins for team1
        team2_wins (int): Number of wins for team2
        recent_matches (list): List of recent match results
        venue (str, optional): Upcoming match venue
        
    Returns:
        str: Formatted prompt for the LLM
    """
    logger.info(f"Generating head-to-head prompt for {team1} vs {team2}")
    
    # Format recent matches as string
    recent_matches_str = ""
    if recent_matches:
        for match in recent_matches[:5]:  # Limit to most recent 5
            if isinstance(match, dict):
                winner = match.get('winner', 'Unknown')
                date = match.get('date', 'Unknown date')
                result = match.get('result_summary', f"{winner} won")
                recent_matches_str += f"- {date}: {result}\n"
            else:
                recent_matches_str += f"- {match}\n"
    
    # Create base prompt
    prompt = f"""
    You are a cricket analytics expert specializing in IPL team match-ups.
    
    Based on the following head-to-head information between {team1} and {team2}:
    - Total matches played: {total_matches}
    - {team1} wins: {team1_wins} ({team1_wins/total_matches*100 if total_matches > 0 else 0:.1f}%)
    - {team2} wins: {team2_wins} ({team2_wins/total_matches*100 if total_matches > 0 else 0:.1f}%)
    - Draws/No Results: {total_matches - team1_wins - team2_wins}
    
    Recent match results:
    {recent_matches_str}
    """
    
    # Add venue information if available
    if venue:
        prompt += f"The upcoming match will be played at: {venue}\n"
    
    # Add the instruction
    prompt += f"""
    
    Provide a comprehensive head-to-head analysis between {team1} and {team2} in IPL history.
    Include:
    1. Historical rivalry and trends
    2. Key match-ups between important players
    3. Venue-specific performance patterns (if applicable)
    4. Recent momentum and form considerations
    5. Strategic factors that have influenced past results
    
    Keep your analysis insightful, data-driven, and balanced. Format as 4-5 paragraphs with clear headings.
    """
    
    return prompt
