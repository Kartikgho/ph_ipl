import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib.util

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import IPL_TEAMS, STADIUM_IMAGES
from ml_models import ipl_prediction_model
from utils import visualization, data_helpers
from llm_integration import ollama_client

# Function to load modules from file path
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load our page modules
home_page = load_module("home", "pages/home.py")
match_predictions_page = load_module("match_predictions", "pages/match_predictions.py")
player_analysis_page = load_module("player_analysis", "pages/player_analysis.py")
team_analysis_page = load_module("team_analysis", "pages/team_analysis.py")
model_explanation_page = load_module("model_explanation", "pages/model_explanation.py")

st.set_page_config(
    page_title="IPL Cricket Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'model' not in st.session_state:
    # Initialize prediction model
    st.session_state.model = ipl_prediction_model.IPLPredictionModel()
    try:
        st.session_state.model.load_or_train_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.model = None

# Navigation in sidebar
st.sidebar.title("IPL Prediction System")
st.sidebar.image(STADIUM_IMAGES[0], use_column_width=True)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Match Predictions", "Player Analysis", "Team Analysis", "Model Explanation"]
)

# Display appropriate page based on selection
if page == "Home":
    home_page.show()
elif page == "Match Predictions":
    match_predictions_page.show(model=st.session_state.model)
elif page == "Player Analysis":
    player_analysis_page.show()
elif page == "Team Analysis":
    team_analysis_page.show()
elif page == "Model Explanation":
    model_explanation_page.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "IPL Cricket Prediction System developed by Artizence Systems LLP. "
    "This system uses machine learning to predict match outcomes, scores, and player performance."
)

if __name__ == "__main__":
    # This block will be executed when the script is run directly
    pass
