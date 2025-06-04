"""
State Government Dashboard
=========================

Streamlit dashboard tailored for state government officials with:
- State-specific outbreak monitoring
- Public health response coordination
- Economic impact tracking
- Communication strategy support
- Regional comparison analysis

Date: June 3, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/ubuntu/disease_early_warning_system')

from models.predictive_models import EarlyWarningSystem
from models.advanced_analytics import GeographicClusterAnalyzer
from models.recommendation_agent import RecommendationAgent

# Page configuration
st.set_page_config(
    page_title="State Government Dashboard - Disease Early Warning System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add navigation buttons
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown("### 🏛️ State Government Dashboard")
with col2:
    if st.button("🏠 Main Dashboard", use_container_width=True):
        st.switch_page("dashboards/main_dashboard.py")
with col3:
    if st.button("🏛️ Federal", use_container_width=True):
        st.switch_page("pages/1_Federal_Policy_Makers.py")
with col4:
    if st.button("🏥 Healthcare", use_container_width=True):
        st.switch_page("pages/2_Healthcare_Professionals.py")

# Custom CSS for state government styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, rgba(111, 66, 193, 0.2) 0%, rgba(232, 62, 140, 0.2) 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .state-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6f42c1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .response-critical {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .response-elevated {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .response-normal {
        background: #d1ecf1;
        border-left: 4px solid #0dcaf0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .communication-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .economic-impact {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for recommendations
if 'state_recommendations' not in st.session_state:
    st.session_state.state_recommendations = None

@st.cache_data
def load_state_data():
    """Load and cache data for state dashboard"""
    # Get the current directory and project root
    current_dir = Path(__file__).parent.parent.parent
    data_dir = current_dir / "data"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    simulated_dir = data_dir / "simulated"
    simulated_dir.mkdir(exist_ok=True)
    
    data = {}
    
    # Generate sample CDC data if it doesn't exist
    cdc_file = simulated_dir / "cdc_simulated_2021-12-01_2022-03-31.csv"
    if not cdc_file.exists():
        # Create sample CDC data
        dates = pd.date_range(start='2021-12-01', end='2022-03-31', freq='D')
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
                 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa']
        
        cdc_data = []
        for date in dates:
            for state in states:
                new_admissions = np.random.randint(50, 500)
                total_hospitalized = np.random.randint(100, 1000)
                bed_occupancy = np.random.uniform(40, 90)
                icu_occupancy = np.random.uniform(30, 80)
                
                cdc_data.append({
                    'date': date,
                    'jurisdiction': state,
                    'new_covid_19_hospital_admissions': new_admissions,
                    'total_hospitalized_covid_19_patients': total_hospitalized,
                    'covid_19_inpatient_bed_occupancy_7_day_average': bed_occupancy,
                    'covid_19_icu_bed_occupancy_7_day_average': icu_occupancy
                })
        
        cdc_df = pd.DataFrame(cdc_data)
        cdc_df.to_csv(cdc_file, index=False)
    
    # Load CDC data
    if cdc_file.exists():
        data['cdc'] = pd.read_csv(cdc_file)
        data['cdc']['date'] = pd.to_datetime(data['cdc']['date'])
    
    # Generate and load social media data if needed
    social_file = simulated_dir / "reddit_2021-12-01_2022-03-31.csv"
    if not social_file.exists():
        # Create sample social media data
        dates = pd.date_range(start='2021-12-01', end='2022-03-31', freq='D')
        symptoms = ['fever', 'cough', 'fatigue', 'breathing', 'taste_smell', 'headache', 'body_aches']
        
        social_data = []
        for date in dates:
            # Generate 10-20 posts per day
            num_posts = np.random.randint(10, 20)
            for _ in range(num_posts):
                # Randomly select 1-3 symptoms
                post_symptoms = np.random.choice(symptoms, size=np.random.randint(1, 4), replace=False)
                content = f"I'm experiencing {' and '.join(post_symptoms)}. "
                social_data.append({
                    'date': date,
                    'title': f"COVID symptoms - {', '.join(post_symptoms)}",
                    'content': content
                })
        
        social_df = pd.DataFrame(social_data)
        social_df.to_csv(social_file, index=False)
    
    if social_file.exists():
        data['social'] = pd.read_csv(social_file)
        data['social']['date'] = pd.to_datetime(data['social']['date'])
    
    # Generate and load economic data if needed
    economic_file = simulated_dir / "economic_2021-12-01_2022-03-31.csv"
    if not economic_file.exists():
        # Create sample economic data
        dates = pd.date_range(start='2021-12-01', end='2022-03-31', freq='D')
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
                 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa']
        
        economic_data = []
        for date in dates:
            for state in states:
                economic_data.append({
                    'date': date,
                    'state': state,
                    'gdp_growth': np.random.uniform(-2, 2),
                    'unemployment_rate': np.random.uniform(3, 8),
                    'retail_sales': np.random.uniform(-10, 10),
                    'business_confidence': np.random.uniform(40, 60)
                })
        
        economic_df = pd.DataFrame(economic_data)
        economic_df.to_csv(economic_file, index=False)
    
    if economic_file.exists():
        data['economic'] = pd.read_csv(economic_file)
        data['economic']['date'] = pd.to_datetime(data['economic']['date'])
    
    # Generate and load mobility data if needed
    mobility_file = simulated_dir / "mobility_2021-12-01_2022-03-31.csv"
    if not mobility_file.exists():
        # Create sample mobility data
        dates = pd.date_range(start='2021-12-01', end='2022-03-31', freq='D')
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
                 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa']
        
        mobility_data = []
        for date in dates:
            for state in states:
                mobility_data.append({
                    'date': date,
                    'state': state,
                    'retail_and_recreation': np.random.uniform(-40, 20),
                    'grocery_and_pharmacy': np.random.uniform(-20, 10),
                    'parks': np.random.uniform(-30, 30),
                    'transit_stations': np.random.uniform(-50, 10),
                    'workplaces': np.random.uniform(-40, 20),
                    'residential': np.random.uniform(-10, 30)
                })
        
        mobility_df = pd.DataFrame(mobility_data)
        mobility_df.to_csv(mobility_file, index=False)
    
    if mobility_file.exists():
        data['mobility'] = pd.read_csv(mobility_file)
        data['mobility']['date'] = pd.to_datetime(data['mobility']['date'])
    
    return data

def calculate_state_metrics(cdc_data, state):
    """Calculate key metrics for a specific state"""
    if cdc_data.empty:
        return None
    
    state_data = cdc_data[cdc_data['jurisdiction'] == state]
    if state_data.empty:
        return None
    
    state_data = state_data.sort_values('date')
    latest = state_data.iloc[-1]
    
    # Calculate trends
    if len(state_data) >= 14:
        two_weeks_ago = state_data.iloc[-14]
        admission_trend = ((latest['new_covid_19_hospital_admissions'] - two_weeks_ago['new_covid_19_hospital_admissions']) / 
                          two_weeks_ago['new_covid_19_hospital_admissions'] * 100) if two_weeks_ago['new_covid_19_hospital_admissions'] > 0 else 0
    else:
        admission_trend = 0
    
    # Response level determination
    bed_occupancy = latest['covid_19_inpatient_bed_occupancy_7_day_average']
    icu_occupancy = latest['covid_19_icu_bed_occupancy_7_day_average']
    
    if bed_occupancy > 80 or icu_occupancy > 85 or admission_trend > 30:
        response_level = "Critical"
        response_color = "#dc3545"
    elif bed_occupancy > 65 or icu_occupancy > 70 or admission_trend > 15:
        response_level = "Elevated"
        response_color = "#ffc107"
    else:
        response_level = "Normal"
        response_color = "#0dcaf0"
    
    return {
        'current_admissions': latest['new_covid_19_hospital_admissions'],
        'total_hospitalized': latest.get('total_hospitalized_covid_19_patients', 0),
        'bed_occupancy': bed_occupancy,
        'icu_occupancy': icu_occupancy,
        'admission_trend': admission_trend,
        'response_level': response_level,
        'response_color': response_color,
        'data': state_data
    }

def create_regional_comparison(cdc_data, target_state):
    """Create comparison with neighboring/similar states"""
    if cdc_data.empty:
        return None
    
    # Get latest data for all states
    latest_data = cdc_data.groupby('jurisdiction').last().reset_index()
    
    # Calculate metrics for comparison
    comparison_data = []
    for _, row in latest_data.iterrows():
        state = row['jurisdiction']
        
        # Get state trend
        state_data = cdc_data[cdc_data['jurisdiction'] == state].tail(7)
        if len(state_data) >= 2:
            trend = ((state_data['new_covid_19_hospital_admissions'].iloc[-1] - 
                     state_data['new_covid_19_hospital_admissions'].iloc[0]) / 
                     len(state_data))
        else:
            trend = 0
        
        comparison_data.append({
            'state': state,
            'admissions': row['new_covid_19_hospital_admissions'],
            'bed_occupancy': row['covid_19_inpatient_bed_occupancy_7_day_average'],
            'icu_occupancy': row['covid_19_icu_bed_occupancy_7_day_average'],
            'trend': trend,
            'is_target': state == target_state
        })
    
    return pd.DataFrame(comparison_data)

def generate_public_health_actions(metrics, economic_data, mobility_data):
    """Generate recommended public health actions"""
    actions = []
    
    if metrics:
        response_level = metrics['response_level']
        
        if response_level == "Critical":
            actions.append({
                'priority': 'Immediate',
                'category': 'Healthcare System',
                'action': 'Activate emergency hospital surge protocols',
                'timeline': 'Within 24 hours',
                'rationale': f"Bed occupancy at {metrics['bed_occupancy']:.1f}%, ICU at {metrics['icu_occupancy']:.1f}%"
            })
            
            actions.append({
                'priority': 'Immediate',
                'category': 'Public Communication',
                'action': 'Issue public health emergency advisory',
                'timeline': 'Within 6 hours',
                'rationale': 'Critical healthcare capacity requires immediate public awareness'
            })
            
            actions.append({
                'priority': 'High',
                'category': 'Resource Mobilization',
                'action': 'Request federal assistance and mutual aid',
                'timeline': 'Within 48 hours',
                'rationale': 'State resources may be insufficient for current demand'
            })
        
        elif response_level == "Elevated":
            actions.append({
                'priority': 'High',
                'category': 'Preparedness',
                'action': 'Prepare surge capacity and review response plans',
                'timeline': 'Within 72 hours',
                'rationale': f"Approaching critical thresholds: {metrics['bed_occupancy']:.1f}% bed occupancy"
            })
            
            actions.append({
                'priority': 'Medium',
                'category': 'Public Communication',
                'action': 'Increase public health messaging and awareness campaigns',
                'timeline': 'Within 1 week',
                'rationale': 'Proactive communication to prevent further escalation'
            })
        
        # Trend-based actions
        if metrics['admission_trend'] > 25:
            actions.append({
                'priority': 'High',
                'category': 'Intervention',
                'action': 'Consider implementing targeted public health measures',
                'timeline': 'Within 1 week',
                'rationale': f"Rapid increase in admissions: {metrics['admission_trend']:.1f}% over 2 weeks"
            })
    
    # Economic considerations
    if not economic_data.empty:
        latest_economic = economic_data.iloc[-1]
        if latest_economic['unemployment_claims'] > 250000:
            actions.append({
                'priority': 'Medium',
                'category': 'Economic Support',
                'action': 'Coordinate with federal agencies on economic relief programs',
                'timeline': 'Within 2 weeks',
                'rationale': f"High unemployment claims: {latest_economic['unemployment_claims']:,.0f}"
            })
    
    # Mobility considerations
    if not mobility_data.empty:
        latest_mobility = mobility_data.groupby('date').last().iloc[-1]
        if latest_mobility['mobility_change_pct'] < -30:
            actions.append({
                'priority': 'Medium',
                'category': 'Economic Recovery',
                'action': 'Develop plans for safe economic reopening',
                'timeline': 'Within 1 month',
                'rationale': f"Significant mobility reduction: {latest_mobility['mobility_change_pct']:.1f}%"
            })
    
    # Default action
    if not actions:
        actions.append({
            'priority': 'Routine',
            'category': 'Monitoring',
            'action': 'Continue routine surveillance and monitoring',
            'timeline': 'Ongoing',
            'rationale': 'No immediate high-priority actions required'
        })
    
    return actions

def create_communication_strategy(metrics, social_data):
    """Generate communication strategy recommendations"""
    strategies = []
    
    if metrics:
        response_level = metrics['response_level']
        
        if response_level == "Critical":
            strategies.append({
                'audience': 'General Public',
                'message': 'Healthcare system under strain - follow all health guidelines',
                'channels': ['Emergency Alert System', 'Social Media', 'Press Conference'],
                'frequency': 'Daily updates'
            })
            
            strategies.append({
                'audience': 'Healthcare Workers',
                'message': 'Surge protocols activated - additional resources being deployed',
                'channels': ['Professional Networks', 'Direct Communication'],
                'frequency': 'Twice daily'
            })
        
        elif response_level == "Elevated":
            strategies.append({
                'audience': 'General Public',
                'message': 'Increased vigilance needed - practice preventive measures',
                'channels': ['Social Media', 'Local Media', 'Community Leaders'],
                'frequency': 'Every 2-3 days'
            })
        
        else:
            strategies.append({
                'audience': 'General Public',
                'message': 'Situation stable - continue following health guidelines',
                'channels': ['Social Media', 'Weekly Updates'],
                'frequency': 'Weekly'
            })
    
    # Social media sentiment considerations
    if not social_data.empty:
        # Analyze recent sentiment (simplified)
        recent_posts = social_data.tail(1000)  # Last 1000 posts
        if 'sentiment' in recent_posts.columns:
            negative_pct = (recent_posts['sentiment'] == 'negative').mean() * 100
            
            if negative_pct > 60:
                strategies.append({
                    'audience': 'Social Media Users',
                    'message': 'Address misinformation and provide factual updates',
                    'channels': ['Social Media', 'Influencer Partnerships'],
                    'frequency': 'As needed'
                })
    
    return strategies

def load_master_data():
    """Load and process the master disease data file"""
    try:
        df = pd.read_csv('data/collected/Master_data_covid_flu_rsv.csv')
        df['date'] = pd.to_datetime(df['MMWR Year'].astype(str) + '-' + 
                                  df['MMWR Week'].astype(str) + '-1', 
                                  format='%Y-%W-%w')
        return df
    except Exception as e:
        st.error(f"Error loading master data: {str(e)}")
        return None

def get_forecasts(df):
    """Generate forecasts for each disease"""
    forecasts = {}
    for disease in df['Disease'].unique():
        disease_data = df[df['Disease'] == disease]
        if len(disease_data) >= 14:  # Need at least 2 weeks of data
            # Simple forecast: average of last 2 weeks
            last_two_weeks = disease_data['Cases'].tail(2).mean()
            forecasts[disease] = {
                'prediction': last_two_weeks,
                'confidence': 0.8
            }
    return forecasts

def main():
    """Main state government dashboard function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ State Government Dashboard</h1>
        <p>Disease Early Warning System - Public Health Response Coordination</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading state data..."):
        data = load_state_data()
    
    if not data:
        st.error("No data available. Please ensure data collection has been completed.")
        return
    
    # Sidebar controls
    st.sidebar.header("State Dashboard Controls")
    
    # State selector
    cdc_data = data.get('cdc', pd.DataFrame())
    if not cdc_data.empty:
        states = sorted(cdc_data['jurisdiction'].unique().tolist())
        selected_state = st.sidebar.selectbox(
            "Select State",
            states,
            index=0 if states else 0
        )
    else:
        selected_state = None
        st.sidebar.error("No state data available")
    
    # Dashboard mode
    dashboard_mode = st.sidebar.selectbox(
        "Dashboard Mode",
        ["Executive Summary", "Detailed Analysis", "Response Planning"]
    )
    
    # Alert threshold
    alert_threshold = st.sidebar.slider(
        "Alert Sensitivity",
        min_value=1,
        max_value=5,
        value=3,
        help="1=Most sensitive, 5=Least sensitive"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    if not selected_state:
        st.error("Please select a state to view dashboard.")
        return
    
    social_data = data.get('social', pd.DataFrame())
    economic_data = data.get('economic', pd.DataFrame())
    mobility_data = data.get('mobility', pd.DataFrame())
    
    # State Overview Section
    st.header(f"📊 {selected_state} State Overview")
    
    if not cdc_data.empty:
        metrics = calculate_state_metrics(cdc_data, selected_state)
        
        if metrics:
            # Key metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Daily Admissions",
                    f"{metrics['current_admissions']:,.0f}",
                    f"{metrics['admission_trend']:+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Total Hospitalized",
                    f"{metrics['total_hospitalized']:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Bed Occupancy",
                    f"{metrics['bed_occupancy']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "ICU Occupancy",
                    f"{metrics['icu_occupancy']:.1f}%"
                )
            
            # Response level alert
            response_class = f"response-{metrics['response_level'].lower()}"
            st.markdown(f"""
            <div class="{response_class}">
                <strong>Response Level: {metrics['response_level']}</strong><br>
                Current situation requires {metrics['response_level'].lower()} level response protocols.<br>
                Bed occupancy: {metrics['bed_occupancy']:.1f}% | ICU occupancy: {metrics['icu_occupancy']:.1f}% | 2-week trend: {metrics['admission_trend']:+.1f}%
            </div>
            """, unsafe_allow_html=True)
            
            # State trend visualization
            if dashboard_mode in ["Executive Summary", "Detailed Analysis"]:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Daily Admissions', 'Bed Occupancy', 'ICU Occupancy', 'Trend Analysis'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Daily admissions
                fig.add_trace(
                    go.Scatter(x=metrics['data']['date'], 
                              y=metrics['data']['new_covid_19_hospital_admissions'],
                              mode='lines+markers', name='Admissions',
                              line=dict(color='#6f42c1', width=3)),
                    row=1, col=1
                )
                
                # Bed occupancy
                fig.add_trace(
                    go.Scatter(x=metrics['data']['date'], 
                              y=metrics['data']['covid_19_inpatient_bed_occupancy_7_day_average'],
                              mode='lines+markers', name='Bed Occupancy',
                              line=dict(color='#e83e8c', width=3)),
                    row=1, col=2
                )
                
                # ICU occupancy
                fig.add_trace(
                    go.Scatter(x=metrics['data']['date'], 
                              y=metrics['data']['covid_19_icu_bed_occupancy_7_day_average'],
                              mode='lines+markers', name='ICU Occupancy',
                              line=dict(color='#fd7e14', width=3)),
                    row=2, col=1
                )
                
                # 7-day rolling average
                rolling_avg = metrics['data']['new_covid_19_hospital_admissions'].rolling(window=7).mean()
                fig.add_trace(
                    go.Scatter(x=metrics['data']['date'], y=rolling_avg,
                              mode='lines', name='7-day Average',
                              line=dict(color='#20c997', width=2)),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False, title_text=f"{selected_state} Health Metrics Dashboard")
                st.plotly_chart(fig, use_container_width=True)
    
    # Regional Comparison
    if dashboard_mode in ["Detailed Analysis", "Response Planning"]:
        st.header("🗺️ Regional Comparison")
        
        if not cdc_data.empty:
            comparison = create_regional_comparison(cdc_data, selected_state)
            
            if comparison is not None:
                # Comparison chart
                fig = px.scatter(
                    comparison,
                    x='bed_occupancy',
                    y='admissions',
                    size='icu_occupancy',
                    color='is_target',
                    hover_data=['state', 'trend'],
                    title="State Comparison: Bed Occupancy vs Admissions",
                    labels={'bed_occupancy': 'Bed Occupancy (%)', 'admissions': 'Daily Admissions'}
                )
                
                fig.update_traces(
                    marker=dict(line=dict(width=2, color='DarkSlateGrey')),
                    selector=dict(mode='markers')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Ranking table
                comparison_display = comparison.copy()
                comparison_display['rank'] = comparison_display['admissions'].rank(ascending=False, method='min')
                comparison_display = comparison_display.sort_values('rank')
                
                st.subheader(f"{selected_state} Regional Ranking")
                target_rank = comparison_display[comparison_display['is_target']]['rank'].iloc[0]
                st.info(f"{selected_state} ranks #{target_rank:.0f} out of {len(comparison_display)} states in daily admissions")
    
    # Public Health Actions
    if dashboard_mode == "Response Planning":
        st.header("🎯 Recommended Public Health Actions")
        
        actions = generate_public_health_actions(metrics, economic_data, mobility_data)
        
        for action in actions:
            priority_colors = {
                'Immediate': '#dc3545',
                'High': '#fd7e14',
                'Medium': '#ffc107',
                'Routine': '#20c997'
            }
            
            color = priority_colors.get(action['priority'], '#6c757d')
            
            st.markdown(f"""
            <div class="communication-box" style="border-left: 4px solid {color};">
                <strong>{action['priority']} Priority - {action['category']}</strong><br>
                <strong>Action:</strong> {action['action']}<br>
                <strong>Timeline:</strong> {action['timeline']}<br>
                <strong>Rationale:</strong> {action['rationale']}
            </div>
            """, unsafe_allow_html=True)
    
    # Communication Strategy
    if dashboard_mode in ["Executive Summary", "Response Planning"]:
        st.header("📢 Communication Strategy")
        
        strategies = create_communication_strategy(metrics, social_data)
        
        for strategy in strategies:
            st.markdown(f"""
            <div class="communication-box">
                <strong>Target Audience:</strong> {strategy['audience']}<br>
                <strong>Key Message:</strong> {strategy['message']}<br>
                <strong>Channels:</strong> {', '.join(strategy['channels'])}<br>
                <strong>Frequency:</strong> {strategy['frequency']}
            </div>
            """, unsafe_allow_html=True)
    
    # Economic Impact (if in detailed mode)
    if dashboard_mode == "Detailed Analysis" and not economic_data.empty:
        st.header("💰 Economic Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=economic_data['date'],
                y=economic_data['unemployment_claims'],
                mode='lines+markers',
                name='Unemployment Claims',
                line=dict(color='#dc3545', width=3)
            ))
            fig.update_layout(
                title="Weekly Unemployment Claims",
                xaxis_title="Date",
                yaxis_title="Claims",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=economic_data['date'],
                y=economic_data['business_closures'],
                mode='lines+markers',
                name='Business Closures',
                line=dict(color='#fd7e14', width=3)
            ))
            fig.update_layout(
                title="Business Closures",
                xaxis_title="Date",
                yaxis_title="Closures",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Economic summary
        latest_economic = economic_data.iloc[-1]
        st.markdown(f"""
        <div class="economic-impact">
            <strong>Current Economic Impact Summary</strong><br>
            Unemployment Claims: {latest_economic['unemployment_claims']:,.0f}<br>
            Business Closures: {latest_economic['business_closures']:,.0f}<br>
            Consumer Confidence: {latest_economic['consumer_confidence']:.1f}<br>
            Healthcare Jobs Posted: {latest_economic['healthcare_jobs_posted']:,.0f}
        </div>
        """, unsafe_allow_html=True)
    
    

    # Load master data
    master_data = load_master_data()
    if master_data is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    # Generate forecasts
    forecasts = get_forecasts(master_data)
    
    # Initialize recommendation agent
    agent = RecommendationAgent(api_key=os.getenv("CLAUDE_API_KEY"))
    
    # Generate recommendations if not already in session state
    if st.session_state.state_recommendations is None:
        with st.spinner("Generating recommendations..."):
            recommendations = agent.generate_recommendations(
                master_data=master_data,
                forecasts=forecasts,
                stakeholder='state',
                additional_context={
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_sources': ['CDC', 'State Health Departments', 'Local Reports'],
                    'focus_areas': ['Resource Allocation', 'Policy Implementation', 'Inter-Agency Coordination']
                }
            )
            st.session_state.state_recommendations = recommendations
    
    # Display recommendations
    st.markdown("## AI-Powered State-Level Recommendations")
    st.markdown(agent.format_recommendations_for_display(st.session_state.state_recommendations), 
                unsafe_allow_html=True)
    
    # Add refresh button
    if st.button("Refresh Recommendations"):
        with st.spinner("Generating new recommendations..."):
            recommendations = agent.generate_recommendations(
                master_data=master_data,
                forecasts=forecasts,
                stakeholder='state',
                additional_context={
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_sources': ['CDC', 'State Health Departments', 'Local Reports'],
                    'focus_areas': ['Resource Allocation', 'Policy Implementation', 'Inter-Agency Coordination']
                }
            )
            st.session_state.state_recommendations = recommendations
            st.experimental_rerun()
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Disease Early Warning System | State Government Dashboard<br>
        Last Updated: {}<br>
        <small>For official state government use - Coordinate with federal and local agencies</small>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
if __name__ == "__main__":
    main()

