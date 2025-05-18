"""
Configuration settings for IPL Prediction System
"""

# Data Collection Configuration
IPL_STATS_URL = "https://www.iplt20.com/stats"
PLAYER_STATS_URL = "https://www.iplt20.com/stats/all-time/most-runs"
TEAM_STATS_URL = "https://www.iplt20.com/teams"
MATCH_STATS_URL = "https://www.iplt20.com/matches"

# Model Configuration
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42
MODEL_SAVE_PATH = "saved_models/"
FEATURE_COLUMNS = [
    'batting_avg', 'strike_rate', 'bowling_avg', 'economy_rate', 
    'home_ground_advantage', 'head_to_head_wins', 'recent_form',
    'toss_winner', 'venue_stats', 'player_of_match_freq'
]

# LLM Configuration
OLLAMA_API_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"
MAX_TOKENS = 512

# Teams 
IPL_TEAMS = [
    "Chennai Super Kings", 
    "Delhi Capitals", 
    "Gujarat Titans", 
    "Kolkata Knight Riders", 
    "Lucknow Super Giants", 
    "Mumbai Indians", 
    "Punjab Kings", 
    "Rajasthan Royals", 
    "Royal Challengers Bangalore", 
    "Sunrisers Hyderabad"
]

# Page Configuration
STADIUM_IMAGES = [
    "https://pixabay.com/get/g19894cf47a12e95c6e04a2bae4c3af19a2e60e77de492103b53de0949a934dcd0ae07a26e705928158fb49b5d18842a471ea533ca186fccefdf2a8f8de0f7987_1280.jpg",
    "https://pixabay.com/get/gfc876ca1c6779d502f7ff6d656aa326f6ea5548b5b170b56c6e15bff8f74ff578ce6793bfb47c0eac7f2741b16c537ebe4b7675c2053caacf05fd0237dab68b6_1280.jpg",
    "https://pixabay.com/get/gd00fab437b21d05d7369164115e52f15a57ee4cf63211a19ccf541ba14490c2eca0aae112a659938ce538e957550a671b79cc335afc033361fa45701948359db_1280.jpg"
]

CRICKET_ACTION_IMAGES = [
    "https://pixabay.com/get/gbc767d832a5522f9e040e5dba57a001ca4f5c29004d1ff0877437d11c3c91b551e495fb20fe5d4fcd622bece4e5cadb393e34ffe711a452f4c22664e6e690451_1280.jpg",
    "https://pixabay.com/get/gd5cb9a946448a9b67cdf6231ecaa999b92b49c4975e01880051bfb9c48f7114021ae04ba9bd8d01b5e8201955db430d58d9107b599200f2512604496f9a2f845_1280.jpg",
    "https://pixabay.com/get/g7af4812e9f67b237278ee27bc9cf5fec1db08e09e12a0b29860e6f5645b322122d21b80b3002310d1f9c729d0d3e7a3a8902a79f854edc3041d62cab4b5ada02_1280.jpg",
    "https://pixabay.com/get/gb9372d84cd681ecc2da5ef46ba71b80ab3ed098b3ee6bba8ab4ae5ab5973a50367c0552c891cc6321453bd5c5d9f50f8691f1e4ca61f3ff5aa033c08d95fa0cd_1280.jpg"
]

VISUALIZATION_IMAGES = [
    "https://pixabay.com/get/g64f7337d270e6642f2b05abd47f36e425ece968d5048fbbe7cdc9194b96dd6236ced4d3aaa99d95669b88e02f4a97112a92b63ce4d93f40b00196dcea8d01e7a_1280.jpg",
    "https://pixabay.com/get/g65adeb5f8b805b5b2a0c44979da29819b48320dac200c08e685f96f9bb7623fa171f09c71156b401d91dc2fe4b1e9b855c8cc467b2244a69d17f49580561b3d6_1280.jpg",
    "https://pixabay.com/get/gd505651bfb04fa8551e18cf0c9ee913d7a9568e4b9ab1fc3bab8c6d35f691aef5b0154298d4aebb779ca555fc505650d1c2b8856fd14e7e8a728ff7b11870caf_1280.jpg"
]
