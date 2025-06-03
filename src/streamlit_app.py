import os
import sys
import time

import pandas as pd
import streamlit as st

# Add the current directory to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    fetch_disease_data,
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
                    st.image("media/scout.gif", width=150)
                    time.sleep(0.5)
                st.success("✅ Datasets found!")

                # Analysis phase (Aggregator)
                with st.spinner("📋 Aggregator is reviewing data..."):
                    st.image("media/aggregator.gif", width=150)
                    time.sleep(0.5)
                st.success("✅ Datasets reviewed!")

                # Cleaning phase
                with st.spinner("🧹 Cleaning data..."):
                    time.sleep(0.5)
                    try:
                        # Fetch flu data
                        df: pd.DataFrame = fetch_disease_data(search_query)
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                        return
                st.success("✅ Data cleaned and ready!")

            # Display results
            st.markdown("---")
            st.subheader(f"📈 {search_query} Cases Over Time")

            # Use Streamlit's line_chart for proper display - made smaller
            st.line_chart(df.set_index("date")["cases"], height=300)

            # Loading spinner 4: Generating insights - moved here after graph
            # Center the spinner using columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with st.spinner("🧠 Deriving actionable insights..."):
                    time.sleep(2)  # 2 second delay as requested

                    try:
                        # Save the plot using save_plot function
                        save_plot(df, f"{search_query} Cases Over Time (2025)")

                        # Generate recommendations using the saved plot
                        recommendations = generate_recommendations("data/plot.jpeg")

                        # Store recommendations in session state to display later
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
                st.dataframe(df, use_container_width=True)

        else:
            st.warning("Please enter a search query!")


if __name__ == "__main__":
    main()
