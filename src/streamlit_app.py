import time

import pandas as pd  # type: ignore
import streamlit as st

import src.health_data as health_data
import src.social_media as social_media
from src.utils import (
    generate_recommendations,
    save_plot,
)


def main():
    st.set_page_config(page_title="Dashboard", layout="wide")

    st.title("🛡️ AEGIS: Public Health Intelligence")
    st.markdown(
        "A real-time disease detection and response dashboard powered by AI agents."
    )

    # Show all three AI agent GIFs at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            "media/scout.gif",
            caption="🕵️ Scout (Surveillance Agent)",
            use_container_width=True,
        )
    with col2:
        st.image(
            "media/aggregator.gif",
            caption="📊 Aggregator (Analysis Agent)",
            use_container_width=True,
        )
    with col3:
        st.image(
            "media/advisor.gif",
            caption="🧠 Advisor (Response Agent)",
            use_container_width=True,
        )

    st.markdown("---")

    # Search bar
    search_query = st.text_input(
        "Search for disease data:",
        placeholder="e.g., flu or measles",
    )

    # Search button
    if st.button("Search", type="primary") or search_query:
        if search_query:
            # Layout columns
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Surveillance phase (Scout)
                with st.spinner("🔍 Scout is searching datasets..."):
                    # st.image("media/scout.gif", width=150)
                    time.sleep(0.5)
                st.success("✅ Datasets found!")

                # Analysis phase (Aggregator)
                with st.spinner("📋 Aggregator is reviewing data..."):
                    # st.image("media/aggregator.gif", width=150)
                    time.sleep(0.5)
                st.success("✅ Datasets reviewed!")

                # Fetching Twitter chatter
                with st.spinner("🐦 Fetching twitter chatter..."):
                    time.sleep(2)
                    try:
                        # Fetch social media data
                        df_social = social_media.fetch_disease_data(disease=search_query)  # fmt:skip
                    except Exception as e:
                        st.error(f"Error fetching social media data: {str(e)}")
                        return
                st.success("✅ Twitter chatter fetched!")

                # Cleaning Twitter chatter
                with st.spinner("🧹 Cleaning twitter chatter..."):
                    time.sleep(2)
                st.success("✅ Twitter chatter cleaned!")

                # Cleaning phase
                with st.spinner("🧹 Cleaning data..."):
                    time.sleep(0.5)
                    try:
                        # Fetch flu data
                        df_health: pd.DataFrame = health_data.fetch_disease_data(
                            disease=search_query,
                            mock=True,
                        )
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                        return
                st.success("✅ Data cleaned and ready!")

                # Merging
                with st.spinner("🔄 Merging Twitter & CDC data..."):
                    time.sleep(1)
                    try:
                        df_merged = pd.merge(
                            df_health,
                            df_social,
                            on="date",
                            how="left",
                        )
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                        return
                st.success("✅ Data cleaned and ready!")

            # Display results
            st.markdown("---")
            st.subheader(f"📈 {search_query} Cases Over Time")

            # Plot
            chart_data = df_merged.set_index("date")[
                ["num_cases", "social_media_value"]
            ]
            st.line_chart(chart_data, height=300)

            # Generating insights
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with st.spinner("🧠 Deriving actionable insights..."):
                    time.sleep(2)  # 2 second delay as requested

                    try:
                        save_plot(df_merged, x="date", y="num_cases")
                        recommendations = generate_recommendations("data/plot.jpeg")
                        st.session_state.recommendations = recommendations

                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
                        st.session_state.recommendations = None

                st.success("✅ AI insights generated!")

            # Display AI recommendations if available
            if (
                hasattr(st.session_state, "recommendations")
                and st.session_state.recommendations
            ):
                st.markdown("---")
                st.subheader("🤖 AI-Generated Health Recommendations")
                st.markdown(st.session_state.recommendations)

            # Raw data at the very bottom in an expandable section
            with st.expander("📊 View Raw Dataset"):
                st.dataframe(df_merged, use_container_width=True)

        else:
            st.warning("Please enter a search query!")


if __name__ == "__main__":
    main()
