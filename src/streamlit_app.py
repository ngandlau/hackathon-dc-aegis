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

    st.title("🔍 Dashboard")

    # Search bar
    search_query = st.text_input(
        "Search for disease data:",
        placeholder="e.g., flu cases US",
    )

    # Search button
    if st.button("Search", type="primary") or search_query:
        if search_query:
            # Create columns for better layout
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Loading spinner 1: Searching datasets
                with st.spinner("🔍 Searching datasets..."):
                    time.sleep(0.1)

                st.success("✅ Datasets found!")

                # Loading spinner 2: Reviewing datasets
                with st.spinner("📋 Reviewing datasets..."):
                    time.sleep(0.1)

                st.success("✅ Datasets reviewed!")

                # Loading spinner 3: Cleaning data
                with st.spinner("🧹 Cleaning data..."):
                    time.sleep(0.1)

                    # Actually fetch and clean the data here
                    try:
                        # Fetch flu data
                        df: pd.DataFrame = fetch_flu_data_2025()
                        df: pd.DataFrame = clean_flu_data(df)

                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                        return

                st.success("✅ Data cleaned and ready!")

            # Display results
            st.markdown("---")
            st.subheader("📈 Flu Cases Over Time")

            # Use Streamlit's line_chart for proper display
            st.line_chart(df.set_index("date")["flu_cases"])

            # Debug information - more prominent and informative
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

            # Show raw data
            st.subheader("📊 Raw Dataset")
            st.dataframe(df, use_container_width=True)

            # Option to download the data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="flu_data.csv",
                mime="text/csv",
            )

        else:
            st.warning("Please enter a search query!")


if __name__ == "__main__":
    main()
