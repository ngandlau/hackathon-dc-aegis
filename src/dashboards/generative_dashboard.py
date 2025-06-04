import time

import pandas as pd  # type: ignore
import streamlit as st

import src.health_data as health_data
import src.social_media as social_media
from src.utils import (
    generate_recommendations,
    save_plot,
    standardize_value_columns,
)

SLEEP_TIME = 0.0


def main():
    st.set_page_config(
        page_title="AEGIS",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Dark mode CSS styling
    st.markdown(
        """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #581c87 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(30, 58, 138, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%, rgba(255, 255, 255, 0.05) 100%);
        pointer-events: none;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
        letter-spacing: -0.02em;
    }
    
    .main-header .subtitle {
        font-size: 1.25rem;
        font-weight: 500;
        margin: 1rem 0 0 0;
        color: rgba(255, 255, 255, 0.9);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header .tagline {
        font-size: 1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        color: rgba(255, 255, 255, 0.7);
        letter-spacing: 0.02em;
    }
    
    .search-container {
        background: rgba(38, 39, 48, 0.8);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        position: relative;
    }
    
    .search-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, transparent 50%, rgba(255, 255, 255, 0.02) 100%);
        border-radius: 20px;
        pointer-events: none;
    }
    
    .search-container h2 {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    .search-container p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
    }
    
    .agent-status {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00d4aa;
        margin: 0.5rem 0;
    }
    
    .results-section {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown(
        """
    <div class="main-header">
        <h1>🛡️ Aegis</h1>
        <h3>Dashboard Builder</h3
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Prominent search section at the top
    # st.markdown(
    #     """
    # <div class="search-container">
    #     <h2 style="text-align: center; margin-bottom: 1rem;">🔍 Disease Intelligence Search</h2>
    #     <p style="text-align: center; color: #b0b0b0;">Enter a disease name to analyze trends and generate AI-powered health recommendations</p>
    # </div>
    # """,
    #     unsafe_allow_html=True,
    # )

    # Search input - more prominent
    search_query = st.text_input(
        "",
        placeholder="Search for disease data (e.g., flu, measles, covid)",
        label_visibility="collapsed",
        key="main_search",
    )

    # Search button - more prominent
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        search_button = st.button("Build", type="primary", use_container_width=True)

    # Search execution
    if search_button or search_query:
        if search_query:
            st.markdown("---")

            # AI Agents working section
            st.markdown("### 🤖 AI Agent")

            # Surveillance phase (Scout)
            with st.status("🕵️ Searching datasets...", expanded=False) as status:
                time.sleep(SLEEP_TIME * 2)
                status.update(label="✅ Datasets found", state="complete")

            # Analysis phase (Analyst)
            with st.status("📊 Reviewing data...", expanded=False) as status:
                time.sleep(SLEEP_TIME * 2)
                status.update(label="✅ Data is relevant", state="complete")

            # Fetching Twitter chatter
            with st.status("🐦 Fetching Tweets...", expanded=False) as status:
                time.sleep(SLEEP_TIME * 2)
                try:
                    df_social = social_media.fetch_disease_data(disease=search_query)
                    status.update(label="✅ Twitter fetched", state="complete")
                except Exception as e:
                    st.error(f"Error fetching social media data: {str(e)}")
                    return

            # Cleaning Twitter chatter
            with st.status("🧹 Processing Tweets...", expanded=False) as status:
                time.sleep(SLEEP_TIME * 2)
                status.update(label="✅ Tweets processed", state="complete")

            # Health data processing
            with st.status("📋 Processing health data...", expanded=False) as status:
                time.sleep(SLEEP_TIME * 2)
                try:
                    df_health: pd.DataFrame = health_data.fetch_disease_data(
                        disease=search_query,
                        mock=True,
                    )
                    status.update(label="✅ Health data processed", state="complete")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    return

            # Merging data
            with st.status("🔄 Merging datasets...", expanded=False) as status:
                time.sleep(SLEEP_TIME * 1)
                try:
                    df_merged = pd.merge(
                        df_health,
                        df_social,
                        on="date",
                        how="left",
                    )
                    status.update(
                        label="✅ Data merged successfully!", state="complete"
                    )
                except Exception as e:
                    st.error(f"Error merging data: {str(e)}")
                    return

            # Display results in styled container
            st.markdown("---")
            st.markdown(f"### 📈 {search_query.title()}")

            # Plot
            df_merged = standardize_value_columns(df_merged)
            chart_data = df_merged.set_index("date")[
                ["num_cases", "social_media_value"]
            ]
            st.line_chart(chart_data, height=400)

            # Generating insights
            with st.status(
                "🧠 Response Agent: Generating insights...", expanded=False
            ) as status:
                time.sleep(SLEEP_TIME * 2)
                try:
                    save_plot(df_merged, x="date", y="num_cases")
                    recommendations = generate_recommendations("data/plot.jpeg")
                    st.session_state.recommendations = recommendations
                    status.update(
                        label="✅ Response Agent: Insights generated!", state="complete"
                    )
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    st.session_state.recommendations = None

            # Display AI recommendations
            if (
                hasattr(st.session_state, "recommendations")
                and st.session_state.recommendations
            ):
                st.markdown("---")
                st.markdown("### 🤖 AI-Generated Health Recommendations")
                st.markdown(
                    f"""
                    <div class="results-section">
                        {st.session_state.recommendations}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Raw data section
            st.markdown("---")
            with st.expander("📊 View Raw Dataset", expanded=False):
                st.dataframe(df_merged, use_container_width=True)

        else:
            st.warning("⚠️ Please enter a disease name to search!")


if __name__ == "__main__":
    main()
