"""
Enhanced Disease Early Warning System - Main Dashboard
=====================================================

Comprehensive 24/7 surveillance system with multi-disease monitoring,
real-time anomaly detection, and stakeholder-specific intelligence.
Date: June 3, 2025
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# from models.predictive_models import EarlyWarningSystem

# Configure page
st.set_page_config(
    page_title="AEGIS - 24/7 Shielf",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Enhanced CSS for 24/7 monitoring dashboard with dark theme
st.markdown(
    """
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .surveillance-status {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.1) 0%, rgba(16, 185, 129, 0.1) 50%, rgba(52, 211, 153, 0.1) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 32px rgba(5, 150, 105, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
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
</style>
""",
    unsafe_allow_html=True,
)


def create_predictive_forecast(df, disease, days_ahead=30):
    """Create 30-day predictive forecast for a disease"""

    # Prepare data for forecasting
    disease_data = df[disease].values
    X = np.arange(len(disease_data)).reshape(-1, 1)

    # Use polynomial features for better trend capture
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)

    # Fit model
    model = LinearRegression()
    model.fit(X_poly, disease_data)

    # Generate future dates and predictions
    future_X = np.arange(len(disease_data), len(disease_data) + days_ahead).reshape(
        -1, 1
    )
    future_X_poly = poly_features.transform(future_X)
    future_predictions = model.predict(future_X_poly)

    # Add some realistic noise and ensure non-negative values
    noise = np.random.normal(0, np.std(disease_data) * 0.1, days_ahead)
    future_predictions = np.maximum(0, future_predictions + noise)

    # Create future dates
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=days_ahead, freq="D"
    )

    return future_dates, future_predictions


def load_sample_data():
    """Load and generate sample surveillance data"""

    # Generate time series data for multiple diseases
    dates = pd.date_range(start="2024-01-01", end="2025-06-03", freq="D")

    diseases_data = {}

    # COVID-19 data with seasonal patterns
    covid_base = 1000 + 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    covid_noise = np.random.normal(0, 100, len(dates))
    covid_trend = np.cumsum(np.random.normal(0, 5, len(dates)))
    diseases_data["COVID-19"] = np.maximum(0, covid_base + covid_noise + covid_trend)

    # Influenza with strong seasonal pattern
    flu_base = 800 + 600 * np.sin(2 * np.pi * (np.arange(len(dates)) - 60) / 365.25)
    flu_noise = np.random.normal(0, 80, len(dates))
    diseases_data["Influenza"] = np.maximum(0, flu_base + flu_noise)

    # RSV with different seasonal pattern
    rsv_base = 300 + 400 * np.sin(2 * np.pi * (np.arange(len(dates)) - 30) / 365.25)
    rsv_noise = np.random.normal(0, 50, len(dates))
    diseases_data["RSV"] = np.maximum(0, rsv_base + rsv_noise)

    # Norovirus with winter peaks
    noro_base = 200 + 300 * np.sin(2 * np.pi * (np.arange(len(dates)) - 90) / 365.25)
    noro_noise = np.random.normal(0, 40, len(dates))
    diseases_data["Norovirus"] = np.maximum(0, noro_base + noro_noise)

    # Create DataFrame
    df = pd.DataFrame({"date": dates, **diseases_data})

    return df


def create_disease_tracker_card(
    disease_name, current_value, trend, alert_level, last_updated, social_chatter=None
):
    """Create a disease tracker card with social chatter markers"""

    alert_class = f"alert-{alert_level.lower()}"
    trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
    trend_color = "#ef4444" if trend > 0 else "#10b981" if trend < 0 else "#6b7280"

    # Generate social chatter markers
    chatter_html = ""
    if social_chatter:
        for topic, level in social_chatter.items():
            chatter_class = f"chatter-{level.lower()}"
            chatter_html += (
                f'<span class="social-chatter-marker {chatter_class}">{topic}</span>'
            )

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
                {f'<div style="margin-top: 0.8rem;"><small style="color: #d1d5db;">Social Chatter:</small><br>{chatter_html}</div>' if chatter_html else ""}
            </div>
            <div style="text-align: right;">
                <div style="background: {"#ef4444" if alert_level == "High" else "#f59e0b" if alert_level == "Medium" else "#10b981"}; 
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

    st.markdown("## 🎯 Stakeholder Dashboards")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🏛️ Federal Policy Makers", use_container_width=True, type="primary"
        ):
            st.switch_page("pages/1_Federal_Policy_Makers.py")
        st.markdown(
            """
        <div style="text-align: center; margin-top: 0.5rem; color: #666;">
        National oversight, resource allocation, interstate coordination
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button(
            "🏥 Healthcare Professionals", use_container_width=True, type="primary"
        ):
            st.switch_page("pages/2_Healthcare_Professionals.py")
        st.markdown(
            """
        <div style="text-align: center; margin-top: 0.5rem; color: #666;">
        Clinical decision support, capacity monitoring, patient flow
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        if st.button("🏛️ State Government", use_container_width=True, type="primary"):
            st.switch_page("pages/3_State_Government.py")
        st.markdown(
            """
        <div style="text-align: center; margin-top: 0.5rem; color: #666;">
        Regional coordination, public health response, communication
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    """Main dashboard application"""

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>AEGIS/h1>
        <h3>Adaptive Early-warning Grid for Infectious Surveillance</h3>
        <p>Protecting communities through intelligent disease monitoring and prediction</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 24/7 Surveillance Status
    st.markdown(create_surveillance_status(), unsafe_allow_html=True)

    # Navigation Section
    create_navigation_section()

    # Load sample data
    df = load_sample_data()
    current_date = df["date"].max()

    # Key Metrics Row
    st.markdown("## 📊 National Surveillance Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">6,847</div>
            <div class="metric-label">Hospitals Monitored</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">3,142</div>
            <div class="metric-label">Counties Tracked</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">24/7</div>
            <div class="metric-label">Continuous Monitoring</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Active Alerts</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">97.3%</div>
            <div class="metric-label">System Uptime</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Disease Trackers
    st.markdown("## 🦠 Disease Surveillance Trackers")

    col1, col2 = st.columns(2)

    with col1:
        # COVID-19 tracker with social chatter
        covid_current = df["COVID-19"].iloc[-1]
        covid_week_ago = df["COVID-19"].iloc[-8]
        covid_trend = ((covid_current - covid_week_ago) / covid_week_ago) * 100
        covid_alert = "Medium" if covid_trend > 10 else "Low"
        covid_chatter = {
            "symptoms": "High",
            "vaccine": "Medium",
            "testing": "Medium",
            "isolation": "Low",
        }

        st.markdown(
            create_disease_tracker_card(
                "COVID-19",
                covid_current,
                covid_trend,
                covid_alert,
                "2 min ago",
                covid_chatter,
            ),
            unsafe_allow_html=True,
        )

        # RSV tracker with social chatter
        rsv_current = df["RSV"].iloc[-1]
        rsv_week_ago = df["RSV"].iloc[-8]
        rsv_trend = ((rsv_current - rsv_week_ago) / rsv_week_ago) * 100
        rsv_alert = "High" if rsv_trend > 15 else "Medium" if rsv_trend > 5 else "Low"
        rsv_chatter = {"children": "High", "breathing": "Medium", "hospital": "Medium"}

        st.markdown(
            create_disease_tracker_card(
                "RSV (Respiratory Syncytial Virus)",
                rsv_current,
                rsv_trend,
                rsv_alert,
                "5 min ago",
                rsv_chatter,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        # Influenza tracker with social chatter
        flu_current = df["Influenza"].iloc[-1]
        flu_week_ago = df["Influenza"].iloc[-8]
        flu_trend = ((flu_current - flu_week_ago) / flu_week_ago) * 100
        flu_alert = "High" if flu_trend > 20 else "Medium" if flu_trend > 10 else "Low"
        flu_chatter = {
            "fever": "High",
            "flu shot": "Medium",
            "sick days": "High",
            "schools": "Medium",
        }

        st.markdown(
            create_disease_tracker_card(
                "Influenza", flu_current, flu_trend, flu_alert, "1 min ago", flu_chatter
            ),
            unsafe_allow_html=True,
        )

        # Norovirus tracker with social chatter
        noro_current = df["Norovirus"].iloc[-1]
        noro_week_ago = df["Norovirus"].iloc[-8]
        noro_trend = ((noro_current - noro_week_ago) / noro_week_ago) * 100
        noro_alert = "Medium" if noro_trend > 15 else "Low"
        noro_chatter = {
            "stomach bug": "High",
            "food poisoning": "Medium",
            "cruise ship": "Low",
        }

        st.markdown(
            create_disease_tracker_card(
                "Norovirus",
                noro_current,
                noro_trend,
                noro_alert,
                "3 min ago",
                noro_chatter,
            ),
            unsafe_allow_html=True,
        )

    # Time Series Visualizations with Predictive Forecasting
    st.markdown("## 📈 Multi-Disease Time Series Analysis & 30-Day Forecasting")

    # Create comprehensive time series chart with forecasting
    fig = go.Figure()

    colors = {
        "COVID-19": "#ef4444",
        "Influenza": "#3b82f6",
        "RSV": "#f59e0b",
        "Norovirus": "#8b5cf6",
    }

    # Add historical data
    for disease in ["COVID-19", "Influenza", "RSV", "Norovirus"]:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[disease],
                mode="lines",
                name=f"{disease} (Historical)",
                line=dict(color=colors[disease], width=2),
                hovertemplate=f"<b>{disease}</b><br>"
                + "Date: %{x}<br>"
                + "Cases: %{y:,.0f}<br>"
                + "<extra></extra>",
            )
        )

        # Add 30-day forecast
        future_dates, future_predictions = create_predictive_forecast(df, disease)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode="lines",
                name=f"{disease} (30-Day Forecast)",
                line=dict(color=colors[disease], width=2, dash="dash"),
                opacity=0.7,
                hovertemplate=f"<b>{disease} Forecast</b><br>"
                + "Date: %{x}<br>"
                + "Predicted Cases: %{y:,.0f}<br>"
                + "<extra></extra>",
            )
        )

    # Remove the problematic vertical line and replace with a shape
    # Add a vertical line to separate historical from forecast
    last_date = df["date"].iloc[-1]
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dot"),
    )

    # Add annotation for the forecast line
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10,
        font=dict(color="rgba(255,255,255,0.8)"),
    )

    fig.update_layout(
        title="Daily Disease Surveillance - Historical Data & 30-Day Predictive Forecasting",
        xaxis_title="Date",
        yaxis_title="Daily Cases",
        hovermode="x unified",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#f9fafb"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#f9fafb"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recent trends analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 30-Day Trend Analysis")

        # Create 30-day trend chart
        recent_df = df.tail(30)

        fig_trend = go.Figure()

        for disease in ["COVID-19", "Influenza"]:
            fig_trend.add_trace(
                go.Scatter(
                    x=recent_df["date"],
                    y=recent_df[disease],
                    mode="lines+markers",
                    name=disease,
                    line=dict(color=colors[disease], width=3),
                    marker=dict(size=6),
                )
            )

        fig_trend.update_layout(
            title="Recent 30-Day Trends (Primary Diseases)",
            xaxis_title="Date",
            yaxis_title="Daily Cases",
            height=400,
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Anomaly Detection Status")

        # Create anomaly detection visualization
        anomaly_dates = pd.date_range(
            start=current_date - timedelta(days=30), end=current_date, freq="D"
        )
        anomaly_scores = np.random.beta(2, 5, len(anomaly_dates)) * 100

        fig_anomaly = go.Figure()

        # Add anomaly score line
        fig_anomaly.add_trace(
            go.Scatter(
                x=anomaly_dates,
                y=anomaly_scores,
                mode="lines+markers",
                name="Anomaly Score",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=4),
                fill="tonexty",
            )
        )

        # Add threshold line
        fig_anomaly.add_hline(
            y=75,
            line_dash="dash",
            line_color="orange",
            annotation_text="Alert Threshold",
        )

        fig_anomaly.update_layout(
            title="Real-time Anomaly Detection Scores",
            xaxis_title="Date",
            yaxis_title="Anomaly Score (%)",
            height=400,
            yaxis=dict(range=[0, 100]),
        )

        st.plotly_chart(fig_anomaly, use_container_width=True)

    # System Status Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Disease Early Warning System</strong> | 
        Last System Check: {}</p>
        <p>🔒 Secure • 🌐 Real-time • 🎯 AI-Powered • 📊 Evidence-Based</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
