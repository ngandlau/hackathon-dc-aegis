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

    st.title("🔍 Dashboard")

    # Search bar
    search_query = st.text_input(
        "Search for disease data:",
        placeholder="e.g., flu or measles",
    )

    # Search button
    if st.button("Search", type="primary") or search_query:
        if search_query:
            # Create columns for better layout
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Loading spinner 1: Searching datasets
                with st.spinner("🔍 Searching datasets..."):
                    time.sleep(1)

                st.success("✅ Datasets found!")

                # Loading spinner 2: Reviewing datasets
                with st.spinner("📋 Reviewing datasets..."):
                    time.sleep(1)

                st.success("✅ Datasets reviewed!")

                # Loading spinner 3: Cleaning data
                with st.spinner("🧹 Cleaning data..."):
                    time.sleep(1)

                    # Actually fetch and clean the data here
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
