# 🛡️ AEGIS

<div align="center">
Hackathon Project

**AI Agents for Defense & Government Hackathon**  
*June 2025 • Washington, D.C.*

*Sponsored by*  
**National Drone Association** • **OpenAI** • **Anthropic** • **Meta** • **agihouse**

</div>


## What is AEGIS?

[AEGIS](.) automatically builds dashboards for any disease, empowering health experts to make fast & data-driven decisions.

We use a 3-agent-architecture:

* **DataAgent**: Searches for data sources relevant to the disease, public datasets, Twitter and Reddit data.
* **AnalysisAgent**: Preprocesses & combines data sources
* **Response Agent**: Extracts actionable insights


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