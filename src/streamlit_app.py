import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add the current directory to Python path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import fetch_flu_data_2025


def convert_epiweek_to_date(epiweek):
    """Convert epiweek format (YYYYWW) to a readable date"""
    try:
        year = int(str(epiweek)[:4])
        week = int(str(epiweek)[4:])
        # Simple approximation: week 1 starts around Jan 7, each week is 7 days
        day_of_year = (week - 1) * 7 + 7
        date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=day_of_year - 1)
        return date.strftime("%Y-%m-%d")
    except:
        return str(epiweek)


def clean_flu_data(df):
    """Clean the flu data to get date and flu_cases columns"""
    df_clean = df.copy()

    # Convert epiweek to date
    df_clean["date"] = df_clean["epiweek"].apply(convert_epiweek_to_date)

    # Use num_ili as flu_cases (number of influenza-like illness cases)
    df_clean["flu_cases"] = df_clean["num_ili"]

    # Select only the columns we need
    df_clean = df_clean[["date", "flu_cases"]].copy()

    # Sort by date
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean = df_clean.sort_values("date")

    # Reset index
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean


def main():
    st.set_page_config(
        page_title="Disease Data Dashboard", page_icon="📊", layout="wide"
    )

    st.title("🔍 Disease Data Dashboard")
    st.markdown("Search for disease data and visualize trends")

    # Search bar
    search_query = st.text_input(
        "Search for disease data:",
        placeholder="e.g., flu cases US",
        help="Enter your search query for disease data",
    )

    # Search button
    if st.button("Search", type="primary") or search_query:
        if search_query:
            # Create columns for better layout
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Loading spinner 1: Searching datasets
                with st.spinner("🔍 Searching datasets..."):
                    time.sleep(2)

                st.success("✅ Datasets found!")

                # Loading spinner 2: Reviewing datasets
                with st.spinner("📋 Reviewing datasets..."):
                    time.sleep(2)

                st.success("✅ Datasets reviewed!")

                # Loading spinner 3: Cleaning data
                with st.spinner("🧹 Cleaning data..."):
                    time.sleep(2)

                    # Actually fetch and clean the data here
                    try:
                        # Fetch flu data
                        raw_data = fetch_flu_data_2025()

                        # Clean the data
                        clean_data = clean_flu_data(raw_data)

                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                        return

                st.success("✅ Data cleaned and ready!")

            # Display results
            st.markdown("---")

            # Show data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(clean_data))
            with col2:
                st.metric(
                    "Date Range",
                    f"{clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}",
                )
            with col3:
                st.metric("Avg Weekly Cases", f"{clean_data['flu_cases'].mean():,.0f}")

            # Time series chart
            st.subheader("📈 Flu Cases Over Time (2025)")

            # Create the plot using pandas
            fig, ax = plt.subplots(figsize=(12, 6))
            clean_data.plot(
                x="date",
                y="flu_cases",
                ax=ax,
                kind="line",
                color="#1f77b4",
                linewidth=2,
                title="Flu Cases in the US - 2025",
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Flu Cases")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            # Show raw data (optional)
            with st.expander("View Raw Data"):
                st.dataframe(clean_data, use_container_width=True)

        else:
            st.warning("Please enter a search query!")


if __name__ == "__main__":
    main()
