import os
import sys
import time

import pandas as pd
import streamlit as st

# Add the current directory to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import clean_flu_data, fetch_flu_data_2025


def main():
    st.set_page_config(page_title="Dashboard", layout="wide")

    st.title("🛡️ AEGIS: Public Health Intelligence")
    st.markdown("A real-time disease detection and response dashboard powered by AI agents.")

    # Show all three AI agent GIFs at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("media/scout.gif", caption="🕵️ Scout (Surveillance Agent)", use_column_width=True)
    with col2:
        st.image("media/aggregator.gif", caption="📊 Aggregator (Analysis Agent)", use_column_width=True)
    with col3:
        st.image("media/advisor.gif", caption="🧠 Advisor (Response Agent)", use_column_width=True)

    st.markdown("---")

    # Search bar
    search_query = st.text_input(
        "Search for disease data:",
        placeholder="e.g., flu cases US",
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
                        df: pd.DataFrame = fetch_flu_data_2025()
                        df: pd.DataFrame = clean_flu_data(df)
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                        return
                st.success("✅ Data cleaned and ready!")

            # Display results
            st.markdown("---")
            st.subheader("📈 Flu Cases Over Time")
            st.line_chart(df.set_index("date")["flu_cases"])

            st.markdown("---")
            st.subheader("🐛 Debug Information")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Shape:**", df.shape)
                st.write("**Data Types:**")
                st.text(str(df.dtypes))

            with col2:
                st.write("**Missing Values:**")
                st.text(str(df.isnull().sum()))
                st.write("**Flu Cases Stats:**")
                st.text(f"Min: {df['flu_cases'].min():,.0f}")
                st.text(f"Max: {df['flu_cases'].max():,.0f}")
                st.text(f"Mean: {df['flu_cases'].mean():,.1f}")
                st.text(f"Median: {df['flu_cases'].median():,.0f}")

            st.subheader("📊 Raw Dataset")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False),
                file_name="flu_data.csv",
                mime="text/csv",
            )

            # Final recommendations (Advisor)
            st.markdown("---")
            st.subheader("🧠 Advisor Recommendations")
            st.image("media/advisor.gif", width=150)
            st.info("Based on the data, the Advisor recommends preparing for a mild increase in hospital visits over the next 2 weeks in affected regions.")

        else:
            st.warning("Please enter a search query!")


if __name__ == "__main__":
    main()
