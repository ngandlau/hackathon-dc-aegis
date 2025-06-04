"""
Enhanced Disease Early Warning System - Main Dashboard
=====================================================

Comprehensive 24/7 surveillance system with multi-disease monitoring,
real-time anomaly detection, and stakeholder-specific intelligence.
Date: June 3, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append('/home/ubuntu/disease_early_warning_system')

from models.predictive_models import EarlyWarningSystem
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="AEGIS - 24/7 Shield",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for 24/7 monitoring dashboard with dark theme
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.2) 0%, rgba(55, 48, 163, 0.2) 50%, rgba(88, 28, 135, 0.2) 20%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 3rem 0;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 58px 102px rgba(5, 150, 105, 0.4);
    }
    
    .surveillance-status {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.1) 0%, rgba(16, 185, 129, 0.1) 50%, rgba(52, 211, 153, 0.1) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        animation: pulse 1s infinite;
        box-shadow: 0 58px 102px rgba(5, 150, 105, 0.4);
       
    }
    
    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 8px 32px rgba(5, 150, 105, 0.3); }
        50% { opacity: 0.9; box-shadow: 0 8px 32px rgba(5, 150, 105, 0.5); }
        100% { opacity: 1; box-shadow: 0 8px 32px rgba(5, 150, 105, 0.3); }
    }
    
    .disease-tracker {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        margin-top: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f9fafb;
    }
    
    .alert-high {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.3);
    }
    
    .alert-medium {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    .alert-low {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    .stakeholder-nav {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        margin-top: 1rem;
        cursor: pointer;
        color: #f9fafb;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .stakeholder-nav:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f9fafb;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #60a5fa;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
    }
    
    .metric-value.alert {
        color: #ef4444;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    
    .metric-label {
        color: #d1d5db;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .time-indicator {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: #60a5fa;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Courier New', monospace;
        display: inline-block;
        margin-bottom: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.3);
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.2);
    }
    
    .social-chatter-marker {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .chatter-high {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        animation: glow 2s infinite;
    }
    
    .chatter-medium {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    }
    
    .chatter-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    
    @keyframes glow {
        0% { box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3); }
        50% { box-shadow: 0 4px 16px rgba(220, 38, 38, 0.6); }
        100% { box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3); }
    }
    
    /* Dark theme for Streamlit components */
    .stSelectbox > div > div {
        background-color: #374151;
        color: #f9fafb;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    /* Dark theme for charts */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    .plotly .modebar {
        background-color: rgba(31, 41, 55, 0.8) !important;
    }
    
    /* Section headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f9fafb !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1f2937;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    
    .recommendations-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .recommendation-section {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4a5568;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-card.priority-high {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    }
    
    .recommendation-card.priority-medium {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
    }
    
    .recommendation-card.priority-low {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    }
    
    .recommendation-card h4 {
        color: #f9fafb;
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
    }
    
    .recommendation-card p {
        color: #e5e7eb;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .recommendation-card ul {
        color: #e5e7eb;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        font-size: 0.9rem;
    }
    
    .risk-factors, .data-gaps {
        list-style-type: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .risk-factors li, .data-gaps li {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        color: #e5e7eb;
    }
    
    .recommendations-container h3 {
        color: #f9fafb;
        margin: 2rem 0 1rem 0;
        font-size: 1.3rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def create_predictive_forecast(df, disease, days_ahead=30):
    """Create 30-day predictive forecast for a disease"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    # Prepare data for forecasting
    disease_data = df['Cases'].values
    X = np.arange(len(disease_data)).reshape(-1, 1)
    
    # Use polynomial features for better trend capture
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, disease_data)
    
    # Generate future dates and predictions
    future_X = np.arange(len(disease_data), len(disease_data) + days_ahead).reshape(-1, 1)
    future_X_poly = poly_features.transform(future_X)
    future_predictions = model.predict(future_X_poly)
    
    # Add some realistic noise and ensure non-negative values
    noise = np.random.normal(0, np.std(disease_data) * 0.1, days_ahead)
    future_predictions = np.maximum(0, future_predictions + noise)
    
    # Create future dates
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
    
    return future_dates, future_predictions

def create_enhanced_forecast(df, twitter_data=None, disease=None, days_ahead=30):
    """
    Create enhanced 30-day predictive forecast using multiple data sources
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Prepare historical data
    historical_dates = pd.to_datetime(df['date'])
    historical_cases = df['Cases'].values
    
    # Create feature matrix
    X = np.arange(len(historical_cases)).reshape(-1, 1)
    
    # Add seasonal features
    X = np.column_stack([
        X,
        np.sin(2 * np.pi * np.arange(len(historical_cases)) / 365.25),  # Yearly seasonality
        np.cos(2 * np.pi * np.arange(len(historical_cases)) / 365.25),
        np.sin(2 * np.pi * np.arange(len(historical_cases)) / 7),      # Weekly seasonality
        np.cos(2 * np.pi * np.arange(len(historical_cases)) / 7)
    ])
    
    # Add social media features if available
    if twitter_data is not None and disease == 'COVID':
        # Align Twitter data with historical dates
        twitter_aligned = pd.merge(
            pd.DataFrame({'date': historical_dates}),
            twitter_data[['date', 'tweet_count']],
            on='date',
            how='left'
        )
        twitter_aligned['tweet_count'] = twitter_aligned['tweet_count'].fillna(0)
        
        # Add Twitter features
        X = np.column_stack([
            X,
            twitter_aligned['tweet_count'].values,
            twitter_aligned['tweet_count'].rolling(window=7, min_periods=1).mean().values,  # 7-day average
            twitter_aligned['tweet_count'].rolling(window=14, min_periods=1).mean().values  # 14-day average
        ])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_scaled, historical_cases)
    
    # Generate future dates
    last_date = historical_dates.iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
    
    # Create future feature matrix
    future_X = np.arange(len(historical_cases), len(historical_cases) + days_ahead).reshape(-1, 1)
    future_X = np.column_stack([
        future_X,
        np.sin(2 * np.pi * np.arange(len(historical_cases), len(historical_cases) + days_ahead) / 365.25),
        np.cos(2 * np.pi * np.arange(len(historical_cases), len(historical_cases) + days_ahead) / 365.25),
        np.sin(2 * np.pi * np.arange(len(historical_cases), len(historical_cases) + days_ahead) / 7),
        np.cos(2 * np.pi * np.arange(len(historical_cases), len(historical_cases) + days_ahead) / 7)
    ])
    
    # Add social media features for future dates if available
    if twitter_data is not None and disease == 'COVID':
        # Use the last known Twitter data for future predictions
        last_twitter_count = twitter_data['tweet_count'].iloc[-1]
        last_twitter_avg_7 = twitter_data['tweet_count'].rolling(window=7).mean().iloc[-1]
        last_twitter_avg_14 = twitter_data['tweet_count'].rolling(window=14).mean().iloc[-1]
        
        future_X = np.column_stack([
            future_X,
            np.full(days_ahead, last_twitter_count),
            np.full(days_ahead, last_twitter_avg_7),
            np.full(days_ahead, last_twitter_avg_14)
        ])
    
    # Scale future features
    future_X_scaled = scaler.transform(future_X)
    
    # Generate predictions
    predictions = model.predict(future_X_scaled)
    
    # Add some uncertainty to predictions
    uncertainty = np.random.normal(0, np.std(historical_cases) * 0.1, days_ahead)
    predictions = np.maximum(0, predictions + uncertainty)
    
    return future_dates, predictions

def load_sample_data():
    """Load and generate sample surveillance data"""
    
    # Generate time series data for multiple diseases
    dates = pd.date_range(start='2024-01-01', end='2025-06-03', freq='D')
    
    diseases_data = {}
    
    # COVID-19 data with seasonal patterns
    covid_base = 1000 + 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    covid_noise = np.random.normal(0, 100, len(dates))
    covid_trend = np.cumsum(np.random.normal(0, 5, len(dates)))
    diseases_data['COVID-19'] = np.maximum(0, covid_base + covid_noise + covid_trend)
    
    # Influenza with strong seasonal pattern
    flu_base = 800 + 600 * np.sin(2 * np.pi * (np.arange(len(dates)) - 60) / 365.25)
    flu_noise = np.random.normal(0, 80, len(dates))
    diseases_data['Influenza'] = np.maximum(0, flu_base + flu_noise)
    
    # RSV with different seasonal pattern
    rsv_base = 300 + 400 * np.sin(2 * np.pi * (np.arange(len(dates)) - 30) / 365.25)
    rsv_noise = np.random.normal(0, 50, len(dates))
    diseases_data['RSV'] = np.maximum(0, rsv_base + rsv_noise)
    
    # Norovirus with winter peaks
    noro_base = 200 + 300 * np.sin(2 * np.pi * (np.arange(len(dates)) - 90) / 365.25)
    noro_noise = np.random.normal(0, 40, len(dates))
    diseases_data['Norovirus'] = np.maximum(0, noro_base + noro_noise)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        **diseases_data
    })
    
    return df

def create_disease_tracker_card(disease_name, current_value, trend, alert_level, last_updated, social_chatter=None):
    """Create a disease tracker card with social chatter markers"""
    
    alert_class = f"alert-{alert_level.lower()}"
    trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
    trend_color = "#ef4444" if trend > 0 else "#10b981" if trend < 0 else "#6b7280"
    
    # Generate social chatter markers
    chatter_html = ""
    if social_chatter:
        for topic, level in social_chatter.items():
            chatter_class = f"chatter-{level.lower()}"
            chatter_html += f'<span class="social-chatter-marker {chatter_class}">{topic}</span>'
    
    return f"""
    <div class="disease-tracker {alert_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <h4 style="margin: 0; color: #f9fafb;">{disease_name}</h4>
                <p style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: bold; color: #f9fafb;">
                    {current_value:,.0f} cases/day
                </p>
                <p style="margin: 0; color: {trend_color};">
                    {trend_icon} {abs(trend):.1f}% vs last week
                </p>
                {f'<div style="margin-top: 0.8rem;"><small style="color: #d1d5db;">Social Chatter:</small><br>{chatter_html}</div>' if chatter_html else ''}
            </div>
            <div style="text-align: right;">
                <div style="background: {'#ef4444' if alert_level == 'High' else '#f59e0b' if alert_level == 'Medium' else '#10b981'}; 
                           color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                    {alert_level.upper()}
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #9ca3af;">
                    Updated: {last_updated}
                </p>
            </div>
        </div>
    </div>
    """

def create_surveillance_status():
    """Create 24/7 surveillance status indicator"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    return f"""
    <div class="surveillance-status">
        <h3 style="margin: 0;">🔴 LIVE - 24/7 Disease Surveillance Active</h3>
        <p style="margin: 0.5rem 0 0 0;">
            Monitoring 50 states • 3,142 counties • 6,847 hospitals • Real-time data processing
        </p>
        <div class="time-indicator">
            Last Update: {current_time}
        </div>
    </div>
    """

def create_navigation_section():
    """Create enhanced navigation section"""
    
    st.markdown("## Stakeholder Views")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏛️ Federal Policy Makers", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Federal_Policy_Makers.py")
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem; color: #666;">
        National oversight, resource allocation, interstate coordination
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🏥 Healthcare Professionals", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Healthcare_Professionals.py")
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem; color: #666;">
        Clinical decision support, capacity monitoring, patient flow
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🏛️ State Government", use_container_width=True, type="primary"):
            st.switch_page("pages/3_State_Government.py")
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem; color: #666;">
        Regional coordination, public health response, communication
        </div>
        """, unsafe_allow_html=True)

def load_master_data():
    """Load and process the master disease data file"""
    try:
        # Read the master data file
        df = pd.read_csv('data/collected/Master_data_covid_flu_rsv.csv')
        
        # Convert MMWR Year and Week to datetime
        df['date'] = pd.to_datetime(df['MMWR Year'].astype(str) + '-' + 
                                  df['MMWR Week'].astype(str) + '-1', 
                                  format='%Y-%W-%w')
        
        # Create separate dataframes for each disease
        diseases_data = {}
        for disease in ['COVID', 'Influenza', 'RSV']:
            disease_df = df[df['Disease'] == disease].copy()
            disease_df = disease_df.sort_values('date')
            diseases_data[disease] = disease_df
        
        return diseases_data
    except Exception as e:
        logger.error(f"Error loading master data: {str(e)}")
        return None

def load_twitter_volume_data():
    """Load Twitter weekly volume data"""
    try:
        df = pd.read_csv('twitter_weekly_volumes.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.warning(f"Could not load Twitter volume data: {str(e)}")
        return None

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>AEGIS</h1>
        <h3>Adaptive Early-warning Grid for Infectious Surveillance</h3>
        <p>Protecting communities through intelligent disease monitoring and prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Section
    create_navigation_section()
    
    # Add margin
    st.markdown("<div style='margin: 50px 0;'></div>", unsafe_allow_html=True)
    
    # Load master data
    diseases_data = load_master_data()
    twitter_data = load_twitter_volume_data()
    
    if diseases_data is None:
        st.error("Failed to load disease data. Please check the data file.")
        return
    
    # Key Metrics Row
    st.markdown("## National Surveillance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6,847*</div>
            <div class="metric-label">Hospitals Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3,142*</div>
            <div class="metric-label">Counties Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">24/7</div>
            <div class="metric-label">Continuous Monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value alert">4*</div>
            <div class="metric-label">Active Alerts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">97.3%</div>
            <div class="metric-label">System Uptime</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add styled horizontal line
    st.markdown("""
        <hr style='border: 1px solid rgba(255, 255, 255, 0.1); margin: 30px 0;'>
        """, unsafe_allow_html=True)
    
    # Disease Trackers
    st.markdown("## 🦠 Disease Trackers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # COVID-19 tracker
        covid_df = diseases_data['COVID']
        covid_current = covid_df['Cases'].iloc[-1]
        covid_week_ago = covid_df['Cases'].iloc[-2]
        covid_trend = ((covid_current - covid_week_ago) / covid_week_ago) * 100
        covid_alert = "High" if covid_trend > 20 else "Medium" if covid_trend > 10 else "Low"
        covid_chatter = {
            "symptoms": "High",
            "vaccine": "Medium", 
            "testing": "Medium",
            "isolation": "Low"
        }
        
        st.markdown(create_disease_tracker_card(
            "COVID-19", covid_current, covid_trend, covid_alert, 
            covid_df['date'].iloc[-1].strftime("%Y-%m-%d"), covid_chatter
        ), unsafe_allow_html=True)
        
        # RSV tracker
        rsv_df = diseases_data['RSV']
        rsv_current = rsv_df['Cases'].iloc[-1]
        rsv_week_ago = rsv_df['Cases'].iloc[-2]
        rsv_trend = ((rsv_current - rsv_week_ago) / rsv_week_ago) * 100
        rsv_alert = "High" if rsv_trend > 15 else "Medium" if rsv_trend > 5 else "Low"
        rsv_chatter = {
            "children": "High",
            "breathing": "Medium",
            "hospital": "Medium"
        }
        
        st.markdown(create_disease_tracker_card(
            "RSV (Respiratory Syncytial Virus)", rsv_current, rsv_trend, rsv_alert,
            rsv_df['date'].iloc[-1].strftime("%Y-%m-%d"), rsv_chatter
        ), unsafe_allow_html=True)
    
    with col2:
        # Influenza tracker
        flu_df = diseases_data['Influenza']
        flu_current = flu_df['Cases'].iloc[-1]
        flu_week_ago = flu_df['Cases'].iloc[-2]
        flu_trend = ((flu_current - flu_week_ago) / flu_week_ago) * 100
        flu_alert = "High" if flu_trend > 20 else "Medium" if flu_trend > 10 else "Low"
        flu_chatter = {
            "fever": "High",
            "flu shot": "Medium",
            "sick days": "High",
            "schools": "Medium"
        }
        
        st.markdown(create_disease_tracker_card(
            "Influenza", flu_current, flu_trend, flu_alert,
            flu_df['date'].iloc[-1].strftime("%Y-%m-%d"), flu_chatter
        ), unsafe_allow_html=True)
        
        # Norovirus tracker (using synthetic data since not in master file)
        noro_current = 200 + 300 * np.sin(2 * np.pi * (np.arange(1) - 90) / 365.25)
        noro_week_ago = 200 + 300 * np.sin(2 * np.pi * (np.arange(1) - 97) / 365.25)
        noro_trend = ((noro_current - noro_week_ago) / noro_week_ago) * 100
        noro_alert = "Medium" if noro_trend > 15 else "Low"
        noro_chatter = {
            "stomach bug": "High",
            "food poisoning": "Medium",
            "cruise ship": "Low"
        }
        
        st.markdown(create_disease_tracker_card(
            "Norovirus", noro_current[0], noro_trend[0], noro_alert, 
            datetime.now().strftime("%Y-%m-%d"), noro_chatter
        ), unsafe_allow_html=True)
    
    # Time Series Visualizations with Predictive Forecasting
    st.markdown("## 📈 Multi-Disease Time Series Analysis & 30-Day Forecasting")
    
    # Create comprehensive time series chart with forecasting
    fig = go.Figure()
    
    colors = {
        'COVID': '#ef4444',
        'Influenza': '#3b82f6', 
        'RSV': '#f59e0b',
        'Norovirus': '#8b5cf6',
        'Social': '#8B0000'  # Deep red color
    }
    
    # Add historical data
    for disease in ['COVID', 'Influenza', 'RSV']:
        fig.add_trace(go.Scatter(
            x=diseases_data[disease]['date'],
            y=diseases_data[disease]['Cases'],
            mode='lines',
            name=f'{disease} (Historical)',
            line=dict(color=colors[disease], width=2),
            hovertemplate=f'<b>{disease}</b><br>' +
                         'Date: %{x}<br>' +
                         'Cases: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add enhanced forecast
        future_dates, future_predictions = create_enhanced_forecast(
            diseases_data[disease],
            twitter_data if disease == 'COVID' else None,
            disease
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines',
            name=f'{disease} (Enhanced Forecast)',
            line=dict(color=colors[disease], width=2, dash='dash'),
            opacity=0.7,
            hovertemplate=f'<b>{disease} Forecast</b><br>' +
                         'Date: %{x}<br>' +
                         'Predicted Cases: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add Twitter volume data if available
    if twitter_data is not None:
        fig.add_trace(go.Scatter(
            x=twitter_data['date'],
            y=twitter_data['tweet_count'],
            mode='lines+markers',
            name='Social Media Markers (COVID)',
            line=dict(color=colors['Social'], width=2),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate=f'<b>Social Media Markers (COVID)</b><br>' +
                         'Date: %{x}<br>' +
                         'Posts: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add Norovirus data (synthetic)
    noro_dates = pd.date_range(start=diseases_data['COVID']['date'].iloc[0], 
                              end=diseases_data['COVID']['date'].iloc[-1], 
                              freq='D')
    noro_cases = 200 + 300 * np.sin(2 * np.pi * (np.arange(len(noro_dates)) - 90) / 365.25)
    
    fig.add_trace(go.Scatter(
        x=noro_dates,
        y=noro_cases,
        mode='lines',
        name='Norovirus (Historical)',
        line=dict(color=colors['Norovirus'], width=2),
        hovertemplate=f'<b>Norovirus</b><br>' +
                     'Date: %{x}<br>' +
                     'Cases: %{y:,.0f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add Norovirus forecast
    noro_future_dates = pd.date_range(start=noro_dates[-1] + timedelta(days=1), 
                                     periods=30, 
                                     freq='D')
    noro_future_cases = 200 + 300 * np.sin(2 * np.pi * (np.arange(len(noro_future_dates)) - 90) / 365.25)
    
    fig.add_trace(go.Scatter(
        x=noro_future_dates,
        y=noro_future_cases,
        mode='lines',
        name='Norovirus (30-Day Forecast)',
        line=dict(color=colors['Norovirus'], width=2, dash='dash'),
        opacity=0.7,
        hovertemplate=f'<b>Norovirus Forecast</b><br>' +
                         'Date: %{x}<br>' +
                         'Predicted Cases: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    # Remove the problematic vertical line and replace with a shape
    # Add a vertical line to separate historical from forecast
    last_date = diseases_data['COVID']['date'].iloc[-1]
    fig.add_shape(
        type="line",
        x0=last_date, x1=last_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(
            color="rgba(255,255,255,0.5)",
            width=2,
            dash="dot"
        )
    )
    
    # Add annotation for the forecast line
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10,
        font=dict(color="rgba(255,255,255,0.8)")
    )
    
    fig.update_layout(
        title="Daily Disease Surveillance & Social Media Volume",
        xaxis_title="Date",
        yaxis_title="Daily Cases",
        yaxis2=dict(
            title="Social Media Posts",
            overlaying="y",
            side="right",
            showgrid=False,
            color=colors['Social'],
            range=[0, 2000]  # Set fixed range from 0 to 2000
        ),
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=12, color="white")
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f9fafb'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='#f9fafb'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='#f9fafb'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent trends analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 30-Day Symptom Trend Analysis")
        
        # Dummy symptom data for 30 days
        np.random.seed(42)
        days = pd.date_range(end=datetime.now(), periods=30)
        symptoms = [
            ("Fever", "#ef4444"),
            ("Cough", "#3b82f6"),
            ("Sore Throat", "#f59e0b"),
            ("Fatigue", "#8b5cf6"),
            ("Headache", "#10b981"),
            ("Shortness of Breath", "#eab308"),
            ("Loss of Taste/Smell", "#6366f1")
        ]
        symptom_data = {}
        for name, color in symptoms:
            # Generate plausible random trends (e.g., baseline + some noise + a mild trend)
            baseline = np.random.randint(30, 100)
            trend = np.linspace(0, np.random.randint(-10, 10), 30)
            noise = np.random.normal(0, 5, 30)
            values = baseline + trend + noise
            values = np.clip(values, 0, None)
            symptom_data[name] = values
        
        # Adjust 'Loss of Taste/Smell' to be 5/6th of the average of the other symptoms for each day
        other_symptoms = [name for name, _ in symptoms if name != 'Loss of Taste/Smell']
        avg_others = np.mean([symptom_data[name] for name in other_symptoms], axis=0)
        symptom_data['Loss of Taste/Smell'] = (5/6) * avg_others
        
        # Create 30-day symptom trend chart
        fig_trend = go.Figure()
        for (name, color) in symptoms:
            fig_trend.add_trace(go.Scatter(
                x=days,
                y=symptom_data[name],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ))
        
        fig_trend.update_layout(
            title="Recent 30-Day Symptom Trends (Flu & COVID-Related)",
            xaxis_title="Date",
            yaxis_title="Reported Symptom Count",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb'),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                color='#f9fafb'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                color='#f9fafb'
            )
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown("### Social Media vs COVID-19 Correlation Analysis")
        
        if twitter_data is not None:
            # Calculate rolling correlation over different time windows
            windows = [7, 14, 21, 30]  # Different time windows in days
            correlations = []
            dates = []
            
            for window in windows:
                # Get data for the window
                covid_window = diseases_data['COVID'].tail(window)
                twitter_window = twitter_data.tail(window)
                
                # Find common dates
                common_dates = pd.merge(
                    covid_window[['date']], 
                    twitter_window[['date']], 
                    on='date', 
                    how='inner'
                )
                
                if len(common_dates) > 0:
                    # Filter and sort data
                    covid_data = covid_window[covid_window['date'].isin(common_dates['date'])].sort_values('date')
                    twitter_data_window = twitter_window[twitter_window['date'].isin(common_dates['date'])].sort_values('date')
                    
                    # Calculate correlation
                    correlation = np.corrcoef(covid_data['Cases'], twitter_data_window['tweet_count'])[0,1]
                    correlations.append(correlation)
                    dates.append(f"{window}-day")
            
            # Create correlation analysis visualization
            fig_anomaly = go.Figure()
        
            # Add correlation bars
            fig_anomaly.add_trace(go.Bar(
                x=dates,
                y=correlations,
                name='Correlation',
                marker_color=['#ef4444' if abs(c) < 0.5 else '#10b981' for c in correlations],
                text=[f'{c:.2f}' for c in correlations],
                textposition='auto',
            ))
        
            # Add threshold line
            fig_anomaly.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="orange",
                annotation_text="50% Correlation Threshold",
                annotation_position="right"
            )
            
            fig_anomaly.add_hline(
                y=-0.5,
                line_dash="dash",
                line_color="orange"
            )
        
            fig_anomaly.update_layout(
                title="Social Media vs COVID-19 Correlation Over Different Time Windows",
                xaxis_title="Time Window",
                yaxis_title="Correlation Coefficient",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    color='#f9fafb'
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    color='#f9fafb',
                    range=[-1, 1]
                ),
                showlegend=False,
                bargap=0.3
            )
        
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.warning("Twitter data not available for correlation analysis")
    
    # System Status Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Disease Early Warning System</strong> | 
        Last System Check: {}</p>
        <p>• 🌐 Real-time • 🎯 AI-Powered • 📊 Evidence-Based</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

