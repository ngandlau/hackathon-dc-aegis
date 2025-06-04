"""
Healthcare Professionals Dashboard
=================================

Streamlit dashboard tailored for healthcare professionals with:
- Clinical decision support tools
- Hospital capacity monitoring
- Patient flow predictions
- Treatment protocol recommendations
- Resource management insights


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
from models.advanced_analytics import MultiModalEnsemble
from models.recommendation_agent import RecommendationAgent

# Page configuration
st.set_page_config(
    page_title="Healthcare Professionals Dashboard - Disease Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add navigation buttons
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown("### 🏥 Healthcare Professionals Dashboard")
with col2:
    if st.button("🏠 Main Dashboard", use_container_width=True):
        st.switch_page("dashboards/main_dashboard.py")
with col3:
    if st.button("🏛️ Federal", use_container_width=True):
        st.switch_page("pages/1_Federal_Policy_Makers.py")
with col4:
    if st.button("🏛️ State Gov", use_container_width=True):
        st.switch_page("pages/3_State_Government.py")

# Custom CSS for healthcare styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0d6efd 0%, #198754 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .clinical-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .capacity-critical {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .capacity-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .capacity-normal {
        background: #d1ecf1;
        border-left: 4px solid #0dcaf0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .protocol-box {
        background: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .symptom-tracker {
        background: #f0f9ff;
        border: 1px solid #7dd3fc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for recommendations
if 'healthcare_recommendations' not in st.session_state:
    st.session_state.healthcare_recommendations = None

@st.cache_data
def load_healthcare_data():
    """Load and cache data for healthcare dashboard"""
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
        cdc_df.to_csv(simulated_dir / "cdc_simulated_2021-12-01_2022-03-31.csv", index=False)
    
    # Load CDC data
    cdc_file = simulated_dir / "cdc_simulated_2021-12-01_2022-03-31.csv"
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
    
    return data

def calculate_hospital_metrics(cdc_data, jurisdiction=None):
    """Calculate key hospital metrics"""
    if cdc_data.empty:
        return None
    
    if jurisdiction:
        data = cdc_data[cdc_data['jurisdiction'] == jurisdiction]
    else:
        # National aggregation
        agg_dict = {
            'new_covid_19_hospital_admissions': 'sum',
            'covid_19_inpatient_bed_occupancy_7_day_average': 'mean',
            'covid_19_icu_bed_occupancy_7_day_average': 'mean'
        }
        
        # Only include total_hospitalized if it exists
        if 'total_hospitalized_covid_19_patients' in cdc_data.columns:
            agg_dict['total_hospitalized_covid_19_patients'] = 'sum'
        
        data = cdc_data.groupby('date').agg(agg_dict).reset_index()
    
    if data.empty:
        return None
    
    latest = data.iloc[-1]
    
    # Calculate trends
    if len(data) >= 7:
        week_ago = data.iloc[-7]
        admission_trend = ((latest['new_covid_19_hospital_admissions'] - week_ago['new_covid_19_hospital_admissions']) / 
                          week_ago['new_covid_19_hospital_admissions'] * 100) if week_ago['new_covid_19_hospital_admissions'] > 0 else 0
    else:
        admission_trend = 0
    
    # Capacity status
    bed_occupancy = latest['covid_19_inpatient_bed_occupancy_7_day_average']
    icu_occupancy = latest['covid_19_icu_bed_occupancy_7_day_average']
    
    if bed_occupancy > 85 or icu_occupancy > 90:
        capacity_status = "Critical"
        capacity_color = "red"
    elif bed_occupancy > 70 or icu_occupancy > 75:
        capacity_status = "Warning"
        capacity_color = "orange"
    else:
        capacity_status = "Normal"
        capacity_color = "blue"
    
    metrics = {
        'current_admissions': latest['new_covid_19_hospital_admissions'],
        'bed_occupancy': bed_occupancy,
        'icu_occupancy': icu_occupancy,
        'admission_trend': admission_trend,
        'capacity_status': capacity_status,
        'capacity_color': capacity_color,
        'data': data
    }
    
    # Add total hospitalized if available
    if 'total_hospitalized_covid_19_patients' in latest:
        metrics['total_hospitalized'] = latest['total_hospitalized_covid_19_patients']
    else:
        metrics['total_hospitalized'] = 0
    
    return metrics

def create_patient_flow_prediction(cdc_data, jurisdiction, days_ahead=14):
    """Create patient flow predictions"""
    try:
        forecaster = TimeSeriesForecaster()
        
        # Train on available data
        forecaster.train(cdc_data)
        
        # Make prediction
        prediction = forecaster.predict(cdc_data, jurisdiction, days_ahead)
        
        if 'error' in prediction:
            return None
        
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def analyze_symptom_patterns(social_data):
    """Analyze symptom patterns from social media"""
    if social_data.empty:
        return None
    
    # Define symptom keywords
    symptom_keywords = {
        'fever': ['fever', 'temperature', 'hot', 'chills'],
        'cough': ['cough', 'coughing', 'throat'],
        'fatigue': ['tired', 'fatigue', 'exhausted', 'weak'],
        'breathing': ['breath', 'breathing', 'shortness', 'oxygen'],
        'taste_smell': ['taste', 'smell', 'anosmia'],
        'headache': ['headache', 'head pain', 'migraine'],
        'body_aches': ['aches', 'pain', 'sore', 'muscle']
    }
    
    # Count symptom mentions by date
    symptom_data = []
    
    for _, row in social_data.iterrows():
        content = str(row.get('content', '') + ' ' + row.get('title', '')).lower()
        
        for symptom, keywords in symptom_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    symptom_data.append({
                        'date': row['date'],
                        'symptom': symptom,
                        'mentions': 1
                    })
                    break  # Count each post only once per symptom
    
    if not symptom_data:
        return None
    
    symptom_df = pd.DataFrame(symptom_data)
    
    # Aggregate by date and symptom
    daily_symptoms = symptom_df.groupby(['date', 'symptom']).agg({
        'mentions': 'sum'
    }).reset_index()
    
    # Calculate 7-day rolling averages
    symptom_trends = {}
    for symptom in daily_symptoms['symptom'].unique():
        symptom_subset = daily_symptoms[daily_symptoms['symptom'] == symptom].copy()
        symptom_subset = symptom_subset.sort_values('date')
        symptom_subset['rolling_avg'] = symptom_subset['mentions'].rolling(window=7, min_periods=1).mean()
        symptom_trends[symptom] = symptom_subset
    
    return symptom_trends

def generate_clinical_recommendations(metrics, symptom_patterns):
    """Generate clinical recommendations based on current data"""
    recommendations = []
    
    if metrics:
        # Capacity-based recommendations
        if metrics['capacity_status'] == "Critical":
            recommendations.append({
                'priority': 'Immediate',
                'category': 'Capacity Management',
                'recommendation': 'Activate surge protocols and consider patient transfers',
                'rationale': f"Bed occupancy at {metrics['bed_occupancy']:.1f}%, ICU at {metrics['icu_occupancy']:.1f}%"
            })
            
            recommendations.append({
                'priority': 'Immediate',
                'category': 'Staffing',
                'recommendation': 'Deploy additional nursing staff and consider overtime protocols',
                'rationale': 'Critical capacity levels require enhanced staffing'
            })
        
        elif metrics['capacity_status'] == "Warning":
            recommendations.append({
                'priority': 'High',
                'category': 'Preparedness',
                'recommendation': 'Prepare surge capacity and review discharge protocols',
                'rationale': f"Approaching capacity limits: {metrics['bed_occupancy']:.1f}% bed occupancy"
            })
        
        # Trend-based recommendations
        if metrics['admission_trend'] > 25:
            recommendations.append({
                'priority': 'High',
                'category': 'Resource Planning',
                'recommendation': 'Increase PPE inventory and medication stockpiles',
                'rationale': f"Admissions trending up {metrics['admission_trend']:.1f}% over past week"
            })
        
        elif metrics['admission_trend'] < -15:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Resource Optimization',
                'recommendation': 'Consider reallocating resources to other departments',
                'rationale': f"Admissions declining {abs(metrics['admission_trend']):.1f}% - opportunity for optimization"
            })
    
    # Symptom-based recommendations
    if symptom_patterns:
        # Find trending symptoms
        trending_symptoms = []
        for symptom, data in symptom_patterns.items():
            if len(data) >= 7:
                recent_avg = data['rolling_avg'].tail(3).mean()
                earlier_avg = data['rolling_avg'].head(3).mean()
                if recent_avg > earlier_avg * 1.3:  # 30% increase
                    trending_symptoms.append(symptom)
        
        if trending_symptoms:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Clinical Monitoring',
                'recommendation': f'Increase monitoring for {", ".join(trending_symptoms)} symptoms',
                'rationale': 'Social media indicates increasing reports of these symptoms'
            })
    
    # Default recommendation
    if not recommendations:
        recommendations.append({
            'priority': 'Routine',
            'category': 'Standard Care',
            'recommendation': 'Continue standard protocols and monitoring',
            'rationale': 'No immediate concerns identified'
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
    """Main healthcare dashboard function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare Professionals Dashboard</h1>
        <p>Disease Early Warning System - Clinical Decision Support and Resource Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading clinical data..."):
        data = load_healthcare_data()
    
    if not data:
        st.error("No data available. Please ensure data collection has been completed.")
        return
    
    # Sidebar controls
    st.sidebar.header("Clinical Dashboard Controls")
    
    # Jurisdiction selector
    cdc_data = data.get('cdc', pd.DataFrame())
    if not cdc_data.empty:
        jurisdictions = ['National'] + sorted(cdc_data['jurisdiction'].unique().tolist())
        selected_jurisdiction = st.sidebar.selectbox(
            "Select Region/State",
            jurisdictions
        )
        
        if selected_jurisdiction == 'National':
            jurisdiction_filter = None
        else:
            jurisdiction_filter = selected_jurisdiction
    else:
        jurisdiction_filter = None
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Analysis Time Range",
        ["Last 7 days", "Last 14 days", "Last 30 days", "Full period"]
    )
    
    # Dashboard sections selector
    sections = st.sidebar.multiselect(
        "Dashboard Sections",
        ["Hospital Metrics", "Patient Flow Predictions", "Symptom Tracking", "Clinical Recommendations"],
        default=["Hospital Metrics", "Clinical Recommendations"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    social_data = data.get('social', pd.DataFrame())
    
    # Hospital Metrics Section
    if "Hospital Metrics" in sections:
        st.header("📊 Hospital Capacity & Metrics")
        
        if not cdc_data.empty:
            metrics = calculate_hospital_metrics(cdc_data, jurisdiction_filter)
            
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
                
                # Capacity status alert
                capacity_class = f"capacity-{metrics['capacity_status'].lower()}"
                st.markdown(f"""
                <div class="{capacity_class}">
                    <strong>Capacity Status: {metrics['capacity_status']}</strong><br>
                    Current bed occupancy: {metrics['bed_occupancy']:.1f}%<br>
                    Current ICU occupancy: {metrics['icu_occupancy']:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Trend charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metrics['data']['date'],
                        y=metrics['data']['new_covid_19_hospital_admissions'],
                        mode='lines+markers',
                        name='Daily Admissions',
                        line=dict(color='#0d6efd', width=3)
                    ))
                    fig.update_layout(
                        title="Daily Hospital Admissions",
                        xaxis_title="Date",
                        yaxis_title="Admissions",
                        height=350,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metrics['data']['date'],
                        y=metrics['data']['covid_19_inpatient_bed_occupancy_7_day_average'],
                        mode='lines+markers',
                        name='Bed Occupancy',
                        line=dict(color='#198754', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=metrics['data']['date'],
                        y=metrics['data']['covid_19_icu_bed_occupancy_7_day_average'],
                        mode='lines+markers',
                        name='ICU Occupancy',
                        line=dict(color='#dc3545', width=3)
                    ))
                    fig.update_layout(
                        title="Bed Occupancy Trends",
                        xaxis_title="Date",
                        yaxis_title="Occupancy (%)",
                        height=350,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Patient Flow Predictions Section
    if "Patient Flow Predictions" in sections and jurisdiction_filter:
        st.header("🔮 Patient Flow Predictions")
        
        prediction = create_patient_flow_prediction(cdc_data, jurisdiction_filter)
        
        if prediction:
            st.success(f"14-day admission forecast: {prediction['ensemble_prediction']:.0f} daily admissions")
            
            # Forecast visualization
            forecast_data = prediction['forecast']
            dates = [item['date'] for item in forecast_data]
            predicted = [item['predicted_admissions'] for item in forecast_data]
            lower = [item['confidence_lower'] for item in forecast_data]
            upper = [item['confidence_upper'] for item in forecast_data]
            
            fig = go.Figure()
            
            # Historical data
            historical = cdc_data[cdc_data['jurisdiction'] == jurisdiction_filter].tail(30)
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['new_covid_19_hospital_admissions'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#6c757d', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=dates,
                y=predicted,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#0d6efd', width=3)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(13,110,253,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"14-Day Patient Flow Forecast - {jurisdiction_filter}",
                xaxis_title="Date",
                yaxis_title="Daily Admissions",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to generate predictions. Insufficient data or model training required.")
    
    # Symptom Tracking Section
    if "Symptom Tracking" in sections:
        st.header("🩺 Community Symptom Tracking")
        
        if not social_data.empty:
            symptom_patterns = analyze_symptom_patterns(social_data)
            
            if symptom_patterns:
                st.markdown("""
                <div class="symptom-tracker">
                    <strong>Social Media Symptom Surveillance</strong><br>
                    Real-time tracking of symptom mentions in community discussions
                </div>
                """, unsafe_allow_html=True)
                
                # Symptom trends chart
                fig = go.Figure()
                
                colors = px.colors.qualitative.Set3
                for i, (symptom, data) in enumerate(symptom_patterns.items()):
                    fig.add_trace(go.Scatter(
                        x=data['date'],
                        y=data['rolling_avg'],
                        mode='lines+markers',
                        name=symptom.replace('_', ' ').title(),
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                
                fig.update_layout(
                    title="Community Symptom Trends (7-day rolling average)",
                    xaxis_title="Date",
                    yaxis_title="Average Daily Mentions",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top symptoms table
                latest_symptoms = {}
                for symptom, data in symptom_patterns.items():
                    if not data.empty:
                        latest_symptoms[symptom] = data['rolling_avg'].iloc[-1]
                
                if latest_symptoms:
                    symptom_df = pd.DataFrame([
                        {'Symptom': k.replace('_', ' ').title(), 'Daily Mentions': v}
                        for k, v in sorted(latest_symptoms.items(), key=lambda x: x[1], reverse=True)
                    ])
                    
                    st.subheader("Current Symptom Activity")
                    st.dataframe(symptom_df, use_container_width=True)
            else:
                st.info("No symptom patterns detected in current data.")
        else:
            st.warning("No social media data available for symptom tracking.")
    
    # Clinical Recommendations Section
    if "Clinical Recommendations" in sections:
        st.header("💡 Clinical Recommendations")
        
        metrics = calculate_hospital_metrics(cdc_data, jurisdiction_filter) if not cdc_data.empty else None
        symptom_patterns = analyze_symptom_patterns(social_data) if not social_data.empty else None
        
        recommendations = generate_clinical_recommendations(metrics, symptom_patterns)
        
        for rec in recommendations:
            priority_colors = {
                'Immediate': '#dc3545',
                'High': '#fd7e14',
                'Medium': '#ffc107',
                'Routine': '#20c997'
            }
            
            color = priority_colors.get(rec['priority'], '#6c757d')
            
            st.markdown(f"""
            <div class="protocol-box" style="border-left-color: {color};">
                <strong>{rec['priority']} Priority - {rec['category']}</strong><br>
                <strong>Recommendation:</strong> {rec['recommendation']}<br>
                <strong>Clinical Rationale:</strong> {rec['rationale']}
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Disease Early Warning System | Healthcare Professionals Dashboard<br>
        Last Updated: {}<br>
        <small>For clinical decision support - Not a substitute for professional medical judgment</small>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

    # Load master data
    master_data = load_master_data()
    if master_data is None:
        return
    
    # Generate forecasts
    forecasts = get_forecasts(master_data)
    
    # Initialize recommendation agent
    agent = RecommendationAgent(api_key=os.getenv("CLAUDE_API_KEY"))
    
    # Generate recommendations if not already in session state
    if st.session_state.healthcare_recommendations is None:
        with st.spinner("Generating recommendations..."):
            recommendations = agent.generate_recommendations(
                master_data=master_data,
                forecasts=forecasts,
                stakeholder='healthcare',
                additional_context={
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_sources': ['CDC', 'WHO', 'Hospital Reports'],
                    'focus_areas': ['Clinical Protocols', 'Resource Management', 'Patient Care']
                }
            )
            st.session_state.healthcare_recommendations = recommendations
    
    # Display recommendations
    st.markdown("## AI-Powered Clinical Recommendations")
    st.markdown(agent.format_recommendations_for_display(st.session_state.healthcare_recommendations), 
                unsafe_allow_html=True)
    
    # Add refresh button
    if st.button("Refresh Recommendations"):
        with st.spinner("Generating new recommendations..."):
            recommendations = agent.generate_recommendations(
                master_data=master_data,
                forecasts=forecasts,
                stakeholder='healthcare',
                additional_context={
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_sources': ['CDC', 'WHO', 'Hospital Reports'],
                    'focus_areas': ['Clinical Protocols', 'Resource Management', 'Patient Care']
                }
            )
            st.session_state.healthcare_recommendations = recommendations
            st.experimental_rerun()

if __name__ == "__main__":
    main()

