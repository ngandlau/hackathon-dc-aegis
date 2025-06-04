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

We leverage specialized AI agents:

* **Data Engineer Agent**: Deep Research but for datasets
* **Analysis Agent**: Preprocesses & combines data sources
* **Response Agent**: Comes up with actionable items from graphs & data

| Agent | Key Tasks |
|-------|-----------|
| **DataAgent** | Searches for data on official health portals, Twitter/X, Reddit |
| **AnalysisAgent** | Cleans and unifies heterogeneous datasets |
| **ResponseAgent** | Summarizes key trends & anomalies |

## Demo 

Data Engineer Agent: https://www.loom.com/share/273d0a2ef679444c89adb1bbef3454b8?sid=9042f992-972a-4042-b2b3-955246254372

## 🚀 Quick Start

Setup your local environment. Requires `poetry` for dependency management (see https://python-poetry.org/docs/ or run `brew install pipx` and `pipx install poetry` to install it).

Make sure to use Python >3.10.

```bash
poetry config virtualenvs.in-project true
poetry shell
poetry install
```

Launch the dashboard:

```bash
poetry run streamlit run src/dashboards/dashboard.py
poetry run streamlit run src/dashboards/generative_dashboard.py
poetry run streamlit run src/dashboards/agent_dashboard.py
```