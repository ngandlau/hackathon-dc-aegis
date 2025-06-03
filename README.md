# 🛡️ AEGIS

<div align="center">
Hackathon Project

**AI Agents for Defense & Government Hackathon**  
*June 2025 • Washington, D.C.*

*Sponsored by*  
**National Drone Association** • **OpenAI** • **Anthropic** • **Meta** • **agihouse**

</div>


## What is AEGIS?

**AEGIS** automatically builds dashboards for any disease using heterogenous data sources, empowering health experts to make fast & data-driven decisions.

We leverage 3 specialized AI agents:

* **DataAgent**: Searches for data sources relevant to the disease, public datasets, Twitter and Reddit data.
* **AnalysisAgent**: Preprocesses & combines data sources
* **Response Agent**: Extracts actionable insights

| Agent | Key Tasks |
|-------|-----------|
| **DataAgent** | Searches for data on official health portals, Twitter/X, Reddit |
| **AnalysisAgent** | Cleans and unifies heterogeneous datasets |
| **ResponseAgent** | Summarizes key trends & anomalies |


## 🚀 Quick Start

Setup your local environment. Requires `poetry` for dependency management (see https://python-poetry.org/docs/ or run `pipx install poetry` to install it).

```bash
poetry config virtualenvs.in-project true
poetry shell
poetry install
```

Launch the dashboard:

```bash
poetry run streamlit run src/streamlit_app.py
```