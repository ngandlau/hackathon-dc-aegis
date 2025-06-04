"""
Federal Policy Makers Dashboard
==============================

Streamlit dashboard tailored for federal policy makers with:
- National-level outbreak trends and predictions
- Interstate transmission risk analysis
- Resource allocation recommendations
- Policy impact modeling
- Economic impact assessments


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

from models.predictive_models import EarlyWarningSystem, TimeSeriesForecaster
from models.advanced_analytics import MultiModalEnsemble, GeographicClusterAnalyzer
from config.config import get_config, get_stakeholder_config
from models.recommendation_agent import RecommendationAgent

# Page configuration
st.set_page_config(
    page_title="Federal Policy Dashboard - Disease Early Warning System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation header
st.markdown("""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h4 style="margin: 0; color: #2c3e50;">🏛️ Federal Policy Makers Dashboard</h4>
        <div>
            <a href="/" style="text-decoration: none; margin-right: 1rem;">🏠 Main Dashboard</a>
            <a href="/dashboards/healthcare_dashboard.py" style="text-decoration: none; margin-right: 1rem;">🏥 Healthcare</a>
            <a href="/dashboards/state_dashboard.py" style="text-decoration: none;">🏛️ State Gov</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Add navigation buttons
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown("### 🏛️ Federal Policy Makers Dashboard")
with col2:
    if st.button("🏠 Main Dashboard", use_container_width=True):
        st.switch_page("dashboards/main_dashboard.py")
with col3:
    if st.button("🏥 Healthcare", use_container_width=True):
        st.switch_page("pages/2_Healthcare_Professionals.py")
with col4:
    if st.button("🏛️ State Gov", use_container_width=True):
        st.switch_page("pages/3_State_Government.py")

# Custom CSS for federal styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e5984 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .alert-high {
        background: #fee;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-medium {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-low {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for recommendations
if 'federal_recommendations' not in st.session_state:
    st.session_state.federal_recommendations = None

@st.cache_data
def load_data():
    """Load and cache data for the dashboard"""
    # Get the current directory and project root
    current_dir = Path(__file__).parent.parent.parent
    data_dir = current_dir / "data"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    simulated_dir = data_dir / "simulated"
    simulated_dir.mkdir(exist_ok=True)
    
    data = {}
    
    # Generate sample data if it doesn't exist
    if not (simulated_dir / "cdc_simulated_2021-12-01_2022-03-31.csv").exists():
        # Create sample CDC data
        dates = pd.date_range(start='2021-12-01', end='2022-03-31', freq='D')
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
                 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa']
        
        cdc_data = []
        for date in dates:
            for state in states:
                cdc_data.append({
                    'date': date,
                    'jurisdiction': state,
                    'new_covid_19_hospital_admissions': np.random.randint(50, 500),
                    'covid_19_inpatient_bed_occupancy_7_day_average': np.random.uniform(40, 90),
                    'covid_19_icu_bed_occupancy_7_day_average': np.random.uniform(30, 80)
                })
        
        cdc_df = pd.DataFrame(cdc_data)
        cdc_df.to_csv(simulated_dir / "cdc_simulated_2021-12-01_2022-03-31.csv", index=False)
    
    # Load CDC data
    cdc_file = simulated_dir / "cdc_simulated_2021-12-01_2022-03-31.csv"
    if cdc_file.exists():
        data['cdc'] = pd.read_csv(cdc_file)
        data['cdc']['date'] = pd.to_datetime(data['cdc']['date'])
    
    # Generate and load other sample data if needed
    supplementary_files = {
        'mobility': 'mobility_2021-12-01_2022-03-31.csv',
        'economic': 'economic_2021-12-01_2022-03-31.csv',
        'trends': 'google_trends_2021-12-01_2022-03-31.csv'
    }
    
    for name, filename in supplementary_files.items():
        file_path = simulated_dir / filename
        if not file_path.exists():
            # Create sample data
            dates = pd.date_range(start='2021-12-01', end='2022-03-31', freq='D')
            sample_data = pd.DataFrame({
                'date': dates,
                'value': np.random.normal(100, 20, len(dates))
            })
            sample_data.to_csv(file_path, index=False)
        
        if file_path.exists():
            data[name] = pd.read_csv(file_path)
            data[name]['date'] = pd.to_datetime(data[name]['date'])
    
    return data

def create_national_overview(cdc_data):
    """Create national-level overview metrics"""
    if cdc_data.empty:
        return None
    
    # Calculate national daily totals
    national_daily = cdc_data.groupby('date').agg({
        'new_covid_19_hospital_admissions': 'sum',
        'covid_19_inpatient_bed_occupancy_7_day_average': 'mean',
        'covid_19_icu_bed_occupancy_7_day_average': 'mean'
    }).reset_index()
    
    # Calculate key metrics
    latest_date = national_daily['date'].max()
    latest_admissions = national_daily[national_daily['date'] == latest_date]['new_covid_19_hospital_admissions'].iloc[0]
    
    # 7-day change
    week_ago = latest_date - timedelta(days=7)
    week_ago_admissions = national_daily[national_daily['date'] == week_ago]['new_covid_19_hospital_admissions'].iloc[0] if len(national_daily[national_daily['date'] == week_ago]) > 0 else latest_admissions
    
    weekly_change = ((latest_admissions - week_ago_admissions) / week_ago_admissions * 100) if week_ago_admissions > 0 else 0
    
    # Bed occupancy
    latest_bed_occupancy = national_daily[national_daily['date'] == latest_date]['covid_19_inpatient_bed_occupancy_7_day_average'].iloc[0]
    latest_icu_occupancy = national_daily[national_daily['date'] == latest_date]['covid_19_icu_bed_occupancy_7_day_average'].iloc[0]
    
    return {
        'latest_admissions': latest_admissions,
        'weekly_change': weekly_change,
        'bed_occupancy': latest_bed_occupancy,
        'icu_occupancy': latest_icu_occupancy,
        'national_daily': national_daily
    }

def create_interstate_risk_analysis(cdc_data):
    """Analyze interstate transmission risks"""
    if cdc_data.empty:
        return None
    
    # Calculate risk scores for each state
    latest_data = cdc_data.groupby('jurisdiction').last().reset_index()
    
    # Risk factors: admission rate, bed occupancy, trend
    risk_scores = []
    
    for _, row in latest_data.iterrows():
        state = row['jurisdiction']
        
        # Get state trend (last 14 days)
        state_data = cdc_data[cdc_data['jurisdiction'] == state].tail(14)
        if len(state_data) >= 2:
            trend = (state_data['new_covid_19_hospital_admissions'].iloc[-1] - 
                    state_data['new_covid_19_hospital_admissions'].iloc[0]) / len(state_data)
        else:
            trend = 0
        
        # Calculate composite risk score
        admission_risk = min(row['new_covid_19_hospital_admissions'] / 100, 1.0)  # Normalize
        occupancy_risk = row['covid_19_inpatient_bed_occupancy_7_day_average'] / 100
        trend_risk = max(trend / 50, 0)  # Positive trend increases risk
        
        composite_risk = (admission_risk * 0.4 + occupancy_risk * 0.4 + trend_risk * 0.2)
        
        risk_scores.append({
            'state': state,
            'risk_score': composite_risk,
            'admissions': row['new_covid_19_hospital_admissions'],
            'bed_occupancy': row['covid_19_inpatient_bed_occupancy_7_day_average'],
            'trend': trend
        })
    
    return pd.DataFrame(risk_scores).sort_values('risk_score', ascending=False)

def create_resource_allocation_recommendations(cdc_data, economic_data):
    """Generate resource allocation recommendations"""
    recommendations = []
    
    if not cdc_data.empty:
        # Identify high-burden states
        latest_data = cdc_data.groupby('jurisdiction').last().reset_index()
        high_burden_states = latest_data[
            latest_data['covid_19_inpatient_bed_occupancy_7_day_average'] > 75
        ]['jurisdiction'].tolist()
        
        if high_burden_states:
            recommendations.append({
                'priority': 'High',
                'category': 'Hospital Capacity',
                'recommendation': f"Deploy additional medical personnel to {', '.join(high_burden_states[:3])}",
                'rationale': f"Bed occupancy >75% in {len(high_burden_states)} states"
            })
        
        # Identify states with rapid increases
        rapid_increase_states = []
        for state in latest_data['jurisdiction'].unique():
            state_data = cdc_data[cdc_data['jurisdiction'] == state].tail(7)
            if len(state_data) >= 7:
                recent_avg = state_data['new_covid_19_hospital_admissions'].tail(3).mean()
                earlier_avg = state_data['new_covid_19_hospital_admissions'].head(3).mean()
                if recent_avg > earlier_avg * 1.5:  # 50% increase
                    rapid_increase_states.append(state)
        
        if rapid_increase_states:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Surge Preparation',
                'recommendation': f"Pre-position surge resources in {', '.join(rapid_increase_states[:3])}",
                'rationale': f"Rapid admission increases detected in {len(rapid_increase_states)} states"
            })
    
    if not economic_data.empty:
        # Economic impact considerations
        latest_economic = economic_data.iloc[-1]
        if latest_economic['unemployment_claims'] > 300000:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Economic Support',
                'recommendation': "Consider enhanced unemployment benefits and business support",
                'rationale': f"Unemployment claims at {latest_economic['unemployment_claims']:,.0f}"
            })
    
    # Default recommendations
    if not recommendations:
        recommendations.append({
            'priority': 'Low',
            'category': 'Monitoring',
            'recommendation': "Continue routine surveillance and monitoring",
            'rationale': "No immediate high-priority actions identified"
        })
    
    return recommendations

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
    """Main dashboard function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Federal Policy Makers Dashboard</h1>
        <p>Disease Early Warning System - National Overview and Strategic Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    if not data:
        st.error("No data available. Please ensure data collection has been completed.")
        return
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Date range selector
    if 'cdc' in data and not data['cdc'].empty:
        min_date = data['cdc']['date'].min().date()
        max_date = data['cdc']['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=30), max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Alert level selector
    alert_level = st.sidebar.selectbox(
        "Alert Level Filter",
        ["All", "Critical", "High", "Medium", "Low"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    cdc_data = data.get('cdc', pd.DataFrame())
    social_data = data.get('social', pd.DataFrame())
    economic_data = data.get('economic', pd.DataFrame())
    mobility_data = data.get('mobility', pd.DataFrame())
    
    # National Overview Section
    st.header("📊 National Overview")
    
    if not cdc_data.empty:
        overview = create_national_overview(cdc_data)
        
        if overview:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Daily Admissions",
                    f"{overview['latest_admissions']:,.0f}",
                    f"{overview['weekly_change']:+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Bed Occupancy",
                    f"{overview['bed_occupancy']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "ICU Occupancy", 
                    f"{overview['icu_occupancy']:.1f}%"
                )
            
            with col4:
                # Calculate alert level based on metrics
                if overview['bed_occupancy'] > 85 or overview['weekly_change'] > 25:
                    alert_status = "🔴 High"
                elif overview['bed_occupancy'] > 70 or overview['weekly_change'] > 15:
                    alert_status = "🟡 Medium"
                else:
                    alert_status = "🟢 Low"
                
                st.metric("Alert Level", alert_status)
            
            # National trend chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=overview['national_daily']['date'],
                y=overview['national_daily']['new_covid_19_hospital_admissions'],
                mode='lines+markers',
                name='Daily Admissions',
                line=dict(color='#1f4e79', width=3)
            ))
            
            fig.update_layout(
                title="National Daily Hospital Admissions Trend",
                xaxis_title="Date",
                yaxis_title="Admissions",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Interstate Risk Analysis
    st.header("🗺️ Interstate Transmission Risk Analysis")
    
    if not cdc_data.empty:
        risk_analysis = create_interstate_risk_analysis(cdc_data)
        
        if risk_analysis is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Risk map visualization
                fig = px.bar(
                    risk_analysis.head(10),
                    x='state',
                    y='risk_score',
                    color='risk_score',
                    color_continuous_scale='Reds',
                    title="Top 10 States by Transmission Risk Score"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("High-Risk States")
                high_risk = risk_analysis[risk_analysis['risk_score'] > 0.7]
                
                if not high_risk.empty:
                    for _, state in high_risk.iterrows():
                        st.markdown(f"""
                        <div class="alert-high">
                            <strong>{state['state']}</strong><br>
                            Risk Score: {state['risk_score']:.2f}<br>
                            Admissions: {state['admissions']:.0f}<br>
                            Bed Occupancy: {state['bed_occupancy']:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("No high-risk states identified")
    
    # Resource Allocation Recommendations
    st.header("💼 Resource Allocation Recommendations")
    
    recommendations = create_resource_allocation_recommendations(cdc_data, economic_data)
    
    for rec in recommendations:
        priority_class = f"alert-{rec['priority'].lower()}"
        st.markdown(f"""
        <div class="{priority_class}">
            <strong>{rec['priority']} Priority - {rec['category']}</strong><br>
            <strong>Recommendation:</strong> {rec['recommendation']}<br>
            <strong>Rationale:</strong> {rec['rationale']}
        </div>
        """, unsafe_allow_html=True)
    
    # Economic Impact Analysis
    if not economic_data.empty:
        st.header("💰 Economic Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=economic_data['date'],
                y=economic_data['unemployment_claims'],
                mode='lines+markers',
                name='Unemployment Claims',
                line=dict(color='#dc3545')
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
                y=economic_data['consumer_confidence'],
                mode='lines+markers',
                name='Consumer Confidence',
                line=dict(color='#28a745')
            ))
            fig.update_layout(
                title="Consumer Confidence Index",
                xaxis_title="Date",
                yaxis_title="Index",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Policy Impact Modeling
    st.header("📈 Policy Impact Modeling")
    
    st.info("🚧 Advanced policy impact modeling capabilities coming soon. This section will include scenario analysis for different intervention strategies.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Disease Early Warning System | Federal Policy Dashboard<br>
        Last Updated: {}<br>
        <small>For official use only - Contains sensitive health surveillance data</small>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

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
    if st.session_state.federal_recommendations is None:
        with st.spinner("Generating recommendations..."):
            recommendations = agent.generate_recommendations(
                master_data=master_data,
                forecasts=forecasts,
                stakeholder='federal',
                additional_context={
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_sources': ['CDC', 'WHO', 'State Health Departments'],
                    'focus_areas': ['Resource Allocation', 'Policy Development', 'Interstate Coordination']
                }
            )
            st.session_state.federal_recommendations = recommendations
    
    # Display recommendations
    st.markdown("## AI-Powered Policy Recommendations")
    st.markdown(agent.format_recommendations_for_display(st.session_state.federal_recommendations), 
                unsafe_allow_html=True)
    
    # Add refresh button
    if st.button("Refresh Recommendations"):
        with st.spinner("Generating new recommendations..."):
            recommendations = agent.generate_recommendations(
                master_data=master_data,
                forecasts=forecasts,
                stakeholder='federal',
                additional_context={
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_sources': ['CDC', 'WHO', 'State Health Departments'],
                    'focus_areas': ['Resource Allocation', 'Policy Development', 'Interstate Coordination']
                }
            )
            st.session_state.federal_recommendations = recommendations
            st.experimental_rerun()

if __name__ == "__main__":
    main()

