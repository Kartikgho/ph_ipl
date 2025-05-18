"""
Data storage module for IPL prediction system.
This module handles saving and loading data from various storage formats.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import pickle
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_storage.log'
)
logger = logging.getLogger('data_storage')

class IPLDataStorage:
    """
    Store and retrieve IPL cricket data in various formats.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the data storage with a data directory.
        
        Args:
            data_dir (str): Directory to store data files
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
        
        # Create subdirectories for different data types
        self.match_dir = os.path.join(data_dir, "matches")
        self.player_dir = os.path.join(data_dir, "players")
        self.team_dir = os.path.join(data_dir, "teams")
        self.model_dir = os.path.join(data_dir, "models")
        
        for directory in [self.match_dir, self.player_dir, self.team_dir, self.model_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def save_matches(self, match_data, filename=None):
        """
        Save match data to CSV.
        
        Args:
            match_data (pd.DataFrame): Match data to save
            filename (str, optional): Filename to use, default is timestamp-based
            
        Returns:
            str: Path to saved file
        """
        if match_data is None or match_data.empty:
            logger.warning("Empty match data provided, not saving")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"matches_{timestamp}.csv"
        
        file_path = os.path.join(self.match_dir, filename)
        
        try:
            match_data.to_csv(file_path, index=False)
            logger.info(f"Match data saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving match data: {str(e)}")
            return None
    
    def save_players(self, player_data, filename=None):
        """
        Save player data to CSV.
        
        Args:
            player_data (pd.DataFrame): Player data to save
            filename (str, optional): Filename to use, default is timestamp-based
            
        Returns:
            str: Path to saved file
        """
        if player_data is None or player_data.empty:
            logger.warning("Empty player data provided, not saving")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"players_{timestamp}.csv"
        
        file_path = os.path.join(self.player_dir, filename)
        
        try:
            player_data.to_csv(file_path, index=False)
            logger.info(f"Player data saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving player data: {str(e)}")
            return None
    
    def save_teams(self, team_data, filename=None):
        """
        Save team data to CSV.
        
        Args:
            team_data (pd.DataFrame): Team data to save
            filename (str, optional): Filename to use, default is timestamp-based
            
        Returns:
            str: Path to saved file
        """
        if team_data is None or team_data.empty:
            logger.warning("Empty team data provided, not saving")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"teams_{timestamp}.csv"
        
        file_path = os.path.join(self.team_dir, filename)
        
        try:
            team_data.to_csv(file_path, index=False)
            logger.info(f"Team data saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving team data: {str(e)}")
            return None
    
    def save_model(self, model, model_name):
        """
        Save a trained model using pickle.
        
        Args:
            model: Trained model object
            model_name (str): Name to identify the model
            
        Returns:
            str: Path to saved model
        """
        if model is None:
            logger.warning("Empty model provided, not saving")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        file_path = os.path.join(self.model_dir, filename)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_matches(self, filename=None):
        """
        Load match data from CSV.
        
        Args:
            filename (str, optional): Specific file to load, default is most recent
            
        Returns:
            pd.DataFrame: Loaded match data
        """
        try:
            if filename is None:
                # Find most recent file
                files = os.listdir(self.match_dir)
                csv_files = [f for f in files if f.endswith('.csv') and f.startswith('matches_')]
                
                if not csv_files:
                    logger.warning("No match data files found")
                    return pd.DataFrame()
                
                # Sort by name (contains timestamp)
                csv_files.sort(reverse=True)
                filename = csv_files[0]
            
            file_path = os.path.join(self.match_dir, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"Match data file not found: {file_path}")
                return pd.DataFrame()
            
            match_data = pd.read_csv(file_path)
            logger.info(f"Loaded match data from {file_path}: {len(match_data)} records")
            return match_data
            
        except Exception as e:
            logger.error(f"Error loading match data: {str(e)}")
            return pd.DataFrame()
    
    def load_players(self, filename=None):
        """
        Load player data from CSV.
        
        Args:
            filename (str, optional): Specific file to load, default is most recent
            
        Returns:
            pd.DataFrame: Loaded player data
        """
        try:
            if filename is None:
                # Find most recent file
                files = os.listdir(self.player_dir)
                csv_files = [f for f in files if f.endswith('.csv') and f.startswith('players_')]
                
                if not csv_files:
                    logger.warning("No player data files found")
                    return pd.DataFrame()
                
                # Sort by name (contains timestamp)
                csv_files.sort(reverse=True)
                filename = csv_files[0]
            
            file_path = os.path.join(self.player_dir, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"Player data file not found: {file_path}")
                return pd.DataFrame()
            
            player_data = pd.read_csv(file_path)
            logger.info(f"Loaded player data from {file_path}: {len(player_data)} records")
            return player_data
            
        except Exception as e:
            logger.error(f"Error loading player data: {str(e)}")
            return pd.DataFrame()
    
    def load_teams(self, filename=None):
        """
        Load team data from CSV.
        
        Args:
            filename (str, optional): Specific file to load, default is most recent
            
        Returns:
            pd.DataFrame: Loaded team data
        """
        try:
            if filename is None:
                # Find most recent file
                files = os.listdir(self.team_dir)
                csv_files = [f for f in files if f.endswith('.csv') and f.startswith('teams_')]
                
                if not csv_files:
                    logger.warning("No team data files found")
                    return pd.DataFrame()
                
                # Sort by name (contains timestamp)
                csv_files.sort(reverse=True)
                filename = csv_files[0]
            
            file_path = os.path.join(self.team_dir, filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"Team data file not found: {file_path}")
                return pd.DataFrame()
            
            team_data = pd.read_csv(file_path)
            logger.info(f"Loaded team data from {file_path}: {len(team_data)} records")
            return team_data
            
        except Exception as e:
            logger.error(f"Error loading team data: {str(e)}")
            return pd.DataFrame()
    
    def load_model(self, model_name=None, specific_file=None):
        """
        Load a trained model using pickle.
        
        Args:
            model_name (str, optional): Name pattern to identify the model
            specific_file (str, optional): Specific model file to load
            
        Returns:
            object: Loaded model object
        """
        try:
            if specific_file is not None:
                file_path = os.path.join(self.model_dir, specific_file)
            elif model_name is not None:
                # Find most recent file with model_name
                files = os.listdir(self.model_dir)
                model_files = [f for f in files if f.endswith('.pkl') and f.startswith(f"{model_name}_")]
                
                if not model_files:
                    logger.warning(f"No model files found for {model_name}")
                    return None
                
                # Sort by name (contains timestamp)
                model_files.sort(reverse=True)
                file_path = os.path.join(self.model_dir, model_files[0])
            else:
                logger.warning("Either model_name or specific_file must be provided")
                return None
            
            if not os.path.exists(file_path):
                logger.warning(f"Model file not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded model from {file_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_data_summary(self):
        """
        Get a summary of available data files.
        
        Returns:
            dict: Summary of data files by type
        """
        try:
            match_files = [f for f in os.listdir(self.match_dir) if f.endswith('.csv')]
            player_files = [f for f in os.listdir(self.player_dir) if f.endswith('.csv')]
            team_files = [f for f in os.listdir(self.team_dir) if f.endswith('.csv')]
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            
            summary = {
                'match_files': sorted(match_files, reverse=True),
                'player_files': sorted(player_files, reverse=True),
                'team_files': sorted(team_files, reverse=True),
                'model_files': sorted(model_files, reverse=True),
                'total_files': len(match_files) + len(player_files) + len(team_files) + len(model_files)
            }
            
            logger.info(f"Data summary generated: {summary['total_files']} total files")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {
                'match_files': [],
                'player_files': [],
                'team_files': [],
                'model_files': [],
                'total_files': 0,
                'error': str(e)
            }
