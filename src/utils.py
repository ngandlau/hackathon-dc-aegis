import base64
from pathlib import Path
from typing import Literal

import anthropic
import outlines
import pandas as pd
import requests
from delphi_epidata import Epidata
from dotenv import load_dotenv
from matplotlib import pyplot as plt

load_dotenv()


class Claude:
    claude_4 = "claude-sonnet-4-20250514"
    claude_3_7 = "claude-3-7-sonnet-latest"
    claude_3_5 = "claude-3-5-sonnet-latest"


def fetch_disease_data(disease: Literal["flu", "measles"]) -> pd.DataFrame:
    if disease == "flu":
        df = fetch_flu_data_2025()
        df = clean_flu_data(df)  # cols: date, cases
        return df
    elif disease == "measles":
        return pd.read_csv("data/measles.csv")  # cols: date, cases
    else:
        raise ValueError(f"Invalid disease: {disease}")


def fetch_flu_data_2025() -> pd.DataFrame:
    regions = ["nat"]
    epiweeks = Epidata.range(202501, 202524)  # Year 2023 weeks 1 to 24
    response = Epidata.fluview(regions, epiweeks)
    if response["result"] != 1:
        raise Exception(f"API request failed: {response['message']}")
    df = pd.DataFrame(response["epidata"])
    return df


def _convert_epiweek_to_date(epiweek):
    """Convert epiweek format (YYYYWW) to a readable date"""
    try:
        year = int(str(epiweek)[:4])
        week = int(str(epiweek)[4:])
        # Simple approximation: week 1 starts around Jan 7, each week is 7 days
        day_of_year = (week - 1) * 7 + 7
        date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=day_of_year - 1)
        return date  # Return datetime object directly
    except:
        return pd.NaT


def clean_flu_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["date"] = df_clean["epiweek"].apply(_convert_epiweek_to_date)
    df_clean["cases"] = df_clean["num_ili"]
    df_clean = df_clean[["date", "cases"]].copy()
    df_clean = df_clean.sort_values("date")
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean[["date", "cases"]]


def search_cdc_datasets(term: str, limit: int = 20, offset: int = 0) -> list[dict]:
    """Full‑text search across CDC's open‑data catalog."""
    DATA_CATALOG = "https://api.us.socrata.com/api/catalog/v1"
    params = {
        "q": term,  # your search keywords
        "domains": "data.cdc.gov",  # constrain results to CDC
        "only": "datasets",  # skip charts & filtered views
        "limit": limit,
        "offset": offset,
    }
    r = requests.get(DATA_CATALOG, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["results"]


def simplify_search_results(search_results: list[dict]) -> list[dict]:
    """Returns only a subset of the metadata of each search result"""
    COLS = ["id", "description", "type", "updatedAt", "createdAt"]
    COLS += ["columns_name", "columns_datatype"]
    results = [
        {key: result["resource"][key] for key in COLS if key in result["resource"]}
        for result in search_results
    ]
    # concatenate columns into a single string
    for result in results:
        colnames = result["columns_name"]
        coldtypes = result["columns_datatype"]
        result["columns_name"] = "; ".join(
            f"{colname}[{dtype}]" for colname, dtype in zip(colnames, coldtypes)
        )
    return results


@outlines.prompt
def prompt_review_search_results(
    simplified_search_results: list[dict],
    user_query: str,
) -> str:
    """
    ## Datasets available

    {% for result in simplified_search_results %}
    ### Num {{ loop.index }} (dataset id = {{ result.id }})
    * **Description**: {{ result.description }}
    * **Type**: {{ result.type }}
    * **Updated At**: {{ result.updatedAt[:10] }}
    * **Created At**: {{ result.createdAt[:10] }}
    * **Columns**: {{ result.columns_name }}

    {% endfor %}

    ## Question

    Which of the above datasets can be used to answer the following user question:
    {{ user_query }}

    Format your answer like this:

    * **dataset <id>dataset_id</id>**: reasoning why the dataset is relevant in bullet points
    * **dataset <id>dataset_id</id>**: reasoning why the dataset is relevant in bullet points

    Make sure the dataset id is inside XML tags.
    If there are multiple relevant datasets, put the most relevant datasets first.
    """
    return ""  # outlines.prompt functions need a return statement


def get_table_llm_context(table: pd.DataFrame, nrows: int = 5) -> str:
    return table.head(nrows).to_markdown(tablefmt="github", index=False)


def call_claude(user_message: str) -> str:
    """
    Convenience wrapper to call the Claude.
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=Claude.claude_3_7,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1024,
    )
    return response.content[0].text


def download_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Download a dataset from the CDC API in CSV format.
    """
    url = f"https://data.cdc.gov/resource/{dataset_id}.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(response.text))


def get_llm_digestable_dataset_preview(df: pd.DataFrame) -> str:
    """
    Get markdown-formatted preview of the dataset that can be digested by the LLM.
    """
    df_head: str = df.head(5).to_markdown(tablefmt="github", index=False)
    colranges = df.describe().to_markdown(tablefmt="github", index=False)
    return f"{df_head}\n\n{colranges}"


@outlines.prompt
def prompt_template_check_data_preview(dataset_preview: str, user_query: str) -> str:
    """
    Here is a preview of the first few rows of the dataset. It is not sorted in a particular order.
    It should give an overview of what columns and data are available.

    {{ dataset_preview }}

    Will there likely be data that can be used to answer the user query?

    {{ user_query }}

    In your answer, include an XML tag <answer>[yes|no]</answer>. Include reasoning.
    """
    return ""


@outlines.prompt
def prompt_template_prep_data(dataset_preview: str, user_query: str) -> str:
    """
    Here is a preview of the dataset:

    {{ dataset_preview}}

    I want to preprocess the dataset such that I can answer the user inquiry: "{{ user_query }}"

    As an output, I want a simple pandas dataframe with 2 columns:
    - date: the date of the data point
    - [column_name]: the value of the column the user is interested in (e.g. number of covid 19 cases). rename this column to make it easy for outsiders to understand.

    A pandas DataFrame named `df` is already in memory.
    🚫  DO NOT read files, fetch URLs, or re‑import data.
    ✅  Only transform `df` and store the result in `df_clean`.
    Return **only** executable Python code (no markdown fences, no comments).
    Write the code to preprocess the dataset.
    Write it in a single code block.
    Format the code block like this:

    ```python
    # all executable code goes here
    ```
    """
    return ""


def strip_code_block(text: str) -> str:
    """
    Remove ```python fences and return only the code.
    """
    body = text.split("```")[1]  # between the two fences
    if body.startswith("python"):
        body = body[len("python") :].lstrip()
    return body


def execute_code_block(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run model‑generated code with pd and df pre‑bound.
    Prefers `df_clean` if the snippet creates it.
    """
    ns = {"pd": pd, "df": df}
    exec(code, ns)
    return ns.get("df_clean", ns["df"])


def plot_time_series(df: pd.DataFrame, column_name: str) -> None:
    """
    Plot a time series of the column.
    """
    import matplotlib.pyplot as plt

    columns = df.columns.tolist()
    x_col = "date"
    y_col = [col for col in columns if col != x_col][0]
    ax = df.plot(x=x_col, y=y_col, kind="line")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def save_plot(df: pd.DataFrame, title: str) -> None:
    """
    Save a plot to project_folder/data directory.
    The JPEG can be fed to an LLM.
    """
    DATA_DIR = Path("data")

    ax = df.plot("date", "cases", figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.title("Cases Over Time")
    plt.tight_layout()
    plt.savefig(
        Path(DATA_DIR, "plot.jpeg"), format="jpeg", dpi=300, bbox_inches="tight"
    )


def generate_recommendations(image_path: str) -> str:
    prompt = """\
You are a public health analyst in 2025. Based on the extracted health data (from an image), provide 3 practical and relevant recommendations for:
	•	Public Health Officials
	•	Government Policy Makers

Each recommendation should:
	•	Reflect current health trends and risks
	•	Include a potential financial cost (in USD)

Present your response in a table format for quick readability.
Include a column with emoji indicating degree of urgency (red (urgent), yellow (moderate), green (low)).
Include a column with cost with dollar emojis (1 to 3 emojis).
Create two tables, one for each stakeholder.
don't use text dollar signs like '$' in your response.
"""

    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    return message.content[0].text
