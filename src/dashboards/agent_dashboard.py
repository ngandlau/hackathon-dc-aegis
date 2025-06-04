import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.llm_utils import extract_text_inside_xml_tags
from src.utils import (
    call_claude,
    check_data_preview,
    download_dataset,
    execute_code_block,
    generate_preprocessing_code,
    get_llm_digestable_dataset_preview,
    review_search_results,
    search_cdc_datasets,
    simplify_search_results,
)

# Import utility functions that will be used later

SLEEP_TIME = 0.0  # Demo sleep time for spinners


def main():
    st.set_page_config(
        page_title="AEGIS Agent",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Dark mode CSS styling (similar to generative_dashboard.py)
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
        padding: 1.5rem 1rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(30, 58, 138, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        position: relative;
        overflow: hidden;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 18px rgba(255, 255, 255, 0.22);
        letter-spacing: -0.02em;
    }
    
    .agent-status {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00d4aa;
        margin: 0.5rem 0;
    }
    
    .llm-response {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ffa500;
    }
    
    .mode-toggle {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 AEGIS Agent</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    FAKE = st.toggle(
        "Fake mode",
        value=False,
    )

    st.markdown("---")

    # User input section
    st.markdown("### 💬 User Query")
    user_query = st.text_input(
        "Enter your data request:",
        placeholder="give me a time series of covid 19 hospitalizations in 2021",
        key="user_query",
    )

    # Start workflow button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        start_workflow = st.button(
            "🚀 Start Agent Workflow", type="primary", use_container_width=True
        )

    if start_workflow and user_query:
        st.markdown("---")
        st.markdown("### 🔄 Agent Workflow in Progress")

        # Step 1: Search CDC Datasets
        with st.status(
            "🔍 Step 1: Searching publicly available datasets...", expanded=False
        ) as status:
            time.sleep(SLEEP_TIME)
            search_results: list[dict[dict[str, Any]]] = search_cdc_datasets(
                user_query, limit=10
            )
            simplified_search_results: list[dict] = simplify_search_results(
                search_results
            )

        # Show search results in expandable section
        with st.expander("🔍 Search Results", expanded=False):
            st.markdown(
                f"""
                <div class="llm-response">
                    <strong>Search Results:</strong><br>
                    {simplified_search_results}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Step 2: LLM Review of Search Results
        with st.status(
            "🧠 Step 2: LLM reviewing search results...", expanded=False
        ) as status:
            time.sleep(1.5)
            llm_review, relevant_ids = review_search_results(
                simplified_search_results, user_query
            )
            status.update(
                label=f"✅ Step 2: LLM found {len(relevant_ids)} relevant datasets",
                state="complete",
            )

        # Show LLM response in expandable section
        with st.expander("🤖 LLM Analysis of Search Results", expanded=False):
            st.markdown(
                f"""
                <div class="llm-response">
                    <strong>LLM Response:</strong><br>
                    {llm_review}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"**Relevant Dataset IDs:** {relevant_ids}")

        # Step 3: Download Datasets
        with st.status(
            "📥 Step 3: Downloading relevant datasets...", expanded=False
        ) as status:
            # TODO: for simplicity, only download first dataset
            example_id = relevant_ids[0]
            # example_id = "cf5u-bm9w" # NOTE: hardcoded for demo-purposes
            df = download_dataset(example_id)
            status.update(label="✅ Step 3: Downloaded dataset", state="complete")

        # Step 4: LLM Check Data Preview
        with st.status(
            "👀 Step 4: LLM checks dataset snapshot...", expanded=False
        ) as status:
            time.sleep(SLEEP_TIME)
            dataset_preview = get_llm_digestable_dataset_preview(df)
            llm_answer_long, llm_answer_short = check_data_preview(
                dataset_preview, user_query
            )
            if FAKE:
                llm_answer_short = "yes"
            status.update(
                label=f"✅ Step 4: LLM confirmed data relevance: {llm_answer_short}",
                state="complete",
            )

        # Show LLM preview check in expandable section
        with st.expander("🤖 LLM Data Preview Analysis", expanded=False):
            st.markdown(
                f"""
                <div class="llm-response">
                    <strong>Dataset Preview:</strong><br>
                    <pre>{dataset_preview}</pre>
                    <br>
                    <strong>LLM Assessment:</strong><br>
                    {llm_answer_long}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Step 5: LLM Data Preprocessing
        with st.status(
            "🔧 Step 5: LLM generating preprocessing code...", expanded=False
        ) as status:
            time.sleep(SLEEP_TIME)
            preprocessing_code = generate_preprocessing_code(
                dataset_preview,
                user_query,
            )

            if FAKE:
                # generate dummy time series data to plot with x="date", y="value"
                df_clean = pd.DataFrame(
                    {
                        "date": pd.date_range(start="2020-01-01", periods=100),
                        "value": np.random.randn(100),
                    }
                )
            else:
                df_clean = execute_code_block(preprocessing_code, df)

            status.update(label="✅ Step 5: Data preprocessed", state="complete")

        # Show LLM preprocessing code in expandable section
        with st.expander("🤖 LLM Generated Preprocessing Code", expanded=False):
            st.markdown("**Generated Code:**")
            st.code(preprocessing_code, language="python")

        # Step 6: Plot Time Series
        with st.status(
            "📊 Step 6: Generating visualization...", expanded=False
        ) as status:
            time.sleep(SLEEP_TIME)
            status.update(label="✅ Step 6: Visualization created", state="complete")

        # Final Results Section
        st.markdown("---")
        st.markdown(f"### 📈 {user_query}")

        if FAKE:
            st.line_chart(df_clean, x="date", y="value")
        else:
            cols = df_clean.columns.tolist()
            llm_answer = call_claude(
                f"i want to plot a simple time series. what are columns x and y I should plot? here are available columns: {cols}. answer with the actual colnames in XML tags like this: <x>colname</x> and <y>colname</y>"
            )
            x_col = extract_text_inside_xml_tags(
                llm_answer,
                "x",
                return_only_first_match=True,
            )
            y_col = extract_text_inside_xml_tags(
                llm_answer,
                "y",
                return_only_first_match=True,
            )

            with st.expander("Columns to plot", expanded=False):
                st.markdown(f"**X:** {x_col} (date)")
                st.markdown(f"**Y:** {y_col} (value)")

            st.line_chart(df_clean, x=x_col, y=y_col)

        # actual dataframe
        with st.expander("📊 View Processed Dataset", expanded=False):
            st.dataframe(df_clean, use_container_width=True)

    elif start_workflow and not user_query:
        st.warning("⚠️ Please enter a data request to start the workflow!")


if __name__ == "__main__":
    main()
