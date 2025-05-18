import streamlit as st
import random
import pandas as pd
from config import IPL_TEAMS, STADIUM_IMAGES, CRICKET_ACTION_IMAGES, VISUALIZATION_IMAGES

def show():
    """
    Enhanced home page of the IPL Prediction System with better visual design
    """
    # Custom CSS for enhanced visual styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #1e3c72;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 800;
        background: linear-gradient(to right, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-text {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1e3c72;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #1e3c72;
        height: 100%;
    }
    .feature-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 15px;
    }
    .team-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 5px;
        transition: transform 0.3s;
    }
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 10px rgba(0,0,0,0.1);
    }
    .step-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: #f0f7ff;
        border-left: 3px solid #1e88e5;
    }
    .disclaimer {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with cricket action image in a centered, optimized layout
    header_image = random.choice(CRICKET_ACTION_IMAGES)
    st.image(header_image, use_container_width=True)
    
    # Title and introduction with enhanced styling
    st.markdown('<h1 class="main-title">🏏 IPL Cricket Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">A state-of-the-art analytics platform that leverages machine learning to forecast match outcomes, team scores, and player performance for the Indian Premier League.</p>', unsafe_allow_html=True)
    
    # Quick stats row
    st.markdown('<h2 class="section-header">📊 IPL 2025 Stats</h2>', unsafe_allow_html=True)
    
    stats_cols = st.columns(4)
    
    with stats_cols[0]:
        st.metric("Total Matches", "74", "8%")
    
    with stats_cols[1]:
        st.metric("Avg. First Innings Score", "172.4", "3.8%")
    
    with stats_cols[2]:
        st.metric("Highest Score", "246/5", "CSK vs RCB")
    
    with stats_cols[3]:
        st.metric("Super Overs", "3", "+1")
    
    # Feature overview with enhanced cards
    st.markdown('<h2 class="section-header">🔑 Key Features</h2>', unsafe_allow_html=True)
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="feature-title">🎯 Match Predictions</h3>', unsafe_allow_html=True)
        st.markdown("""
        - Predict winners with confidence scores
        - Forecast team scores and margins
        - Analyze head-to-head statistics
        - Account for venue and conditions
        """)
        st.image(STADIUM_IMAGES[1], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with feature_cols[1]:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="feature-title">👤 Player Analysis</h3>', unsafe_allow_html=True)
        st.markdown("""
        - Predict individual performance
        - Track form across seasons
        - Compare players by role
        - Identify key match-winners
        """)
        st.image(CRICKET_ACTION_IMAGES[2], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with feature_cols[2]:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="feature-title">📈 Advanced Analytics</h3>', unsafe_allow_html=True)
        st.markdown("""
        - LLM-powered reasoning
        - Visualize confidence levels
        - Understand key factors
        - Track prediction accuracy
        """)
        st.image(VISUALIZATION_IMAGES[0], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # How it works section with step cards
    st.markdown('<h2 class="section-header">⚙️ How It Works</h2>', unsafe_allow_html=True)
    
    # Create steps with visual appeal
    steps = [
        {
            "title": "🔍 Data Collection", 
            "description": "We gather comprehensive statistics on teams, players, venues, and match conditions"
        },
        {
            "title": "⚗️ Feature Engineering", 
            "description": "Raw data is transformed into meaningful predictive features"
        },
        {
            "title": "🧮 Ensemble Modeling", 
            "description": "Multiple algorithms work together to generate accurate predictions"
        },
        {
            "title": "💬 LLM Reasoning", 
            "description": "Language models provide contextual explanations for statistical predictions"
        },
        {
            "title": "📚 Continuous Learning", 
            "description": "The system improves over time as new match data becomes available"
        }
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-card">
            <h3>{i}. {step['title']}</h3>
            <p>{step['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Teams section with hover effects
    st.markdown('<h2 class="section-header">🏆 IPL Teams</h2>', unsafe_allow_html=True)
    
    # Team colors (for visual appeal)
    team_colors = {
        "Chennai Super Kings": "#FFFF00",
        "Delhi Capitals": "#0078BC",
        "Gujarat Titans": "#1D49B0",
        "Kolkata Knight Riders": "#3A225D",
        "Lucknow Super Giants": "#A0E6FF",
        "Mumbai Indians": "#004BA0",
        "Punjab Kings": "#ED1B24",
        "Rajasthan Royals": "#FF69B4",
        "Royal Challengers Bangalore": "#FF0000",
        "Sunrisers Hyderabad": "#FF822A"
    }
    
    # Create a dataframe with team stats
    team_stats = []
    for team in IPL_TEAMS:
        titles = random.randint(0, 5)
        win_rate = round(random.uniform(40, 65), 1)
        team_stats.append({
            "Team": team,
            "Titles": titles,
            "Win Rate": f"{win_rate}%",
            "Color": team_colors.get(team, "#1e3c72")
        })
    
    team_df = pd.DataFrame(team_stats)
    
    # Display team cards
    team_cols = st.columns(5)
    for i, team in enumerate(IPL_TEAMS):
        with team_cols[i % 5]:
            color = team_colors.get(team, "#1e3c72")
            titles = team_df[team_df["Team"] == team]["Titles"].values[0]
            win_rate = team_df[team_df["Team"] == team]["Win Rate"].values[0]
            
            st.markdown(f"""
            <div class="team-card" style="border-top: 3px solid {color};">
                <strong>{team}</strong><br>
                🏆 Titles: {titles}<br>
                📊 Win Rate: {win_rate}
            </div>
            """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown('<h2 class="section-header">🚀 Quick Start</h2>', unsafe_allow_html=True)
    
    # Create columns for quick start options
    quick_cols = st.columns(4)
    
    with quick_cols[0]:
        if st.button("🎯 Match Predictions", use_container_width=True):
            st.session_state.page = "Match Predictions"
            st.rerun()
    
    with quick_cols[1]:
        if st.button("👤 Player Analysis", use_container_width=True):
            st.session_state.page = "Player Analysis"
            st.experimental_rerun()
    
    with quick_cols[2]:
        if st.button("🏏 Team Analysis", use_container_width=True):
            st.session_state.page = "Team Analysis"
            st.experimental_rerun()
    
    with quick_cols[3]:
        if st.button("📊 Model Explanation", use_container_width=True):
            st.session_state.page = "Model Explanation"
            st.experimental_rerun()
    
    # Disclaimer with enhanced styling
    st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
    st.markdown("""
    **Disclaimer**: This system provides predictions based on historical data and statistical models.
    While we strive for accuracy, cricket is inherently unpredictable, and actual results may vary.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Latest updates ticker
    st.markdown("---")
    st.markdown("#### 📰 Latest IPL Updates")
    
    # Sample news items
    news_items = [
        "Mumbai Indians secure crucial win against Chennai Super Kings in last-over thriller",
        "Virat Kohli scores century, leads RCB to commanding victory",
        "Rajasthan Royals announce replacement for injured bowler",
        "Gujarat Titans move to top of table with impressive NRR",
        "IPL Governing Council announces schedule changes for final week"
    ]
    
    # Display as a news ticker
    news_placeholder = st.empty()
    news_index = int(pd.Timestamp.now().timestamp()) % len(news_items)  # Simple way to rotate news
    news_placeholder.info(f"**Latest**: {news_items[news_index]}")
    
    # Footer
    st.markdown("<div style='text-align: center; margin-top: 30px; color: #777;'>© 2025 IPL Cricket Prediction System | Developed by Artizence Systems LLP</div>", unsafe_allow_html=True)
