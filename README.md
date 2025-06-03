# AI+Expo Hackathon Project, Washington D.C. 2025

## Setup & running the project

Setup your local (virtual) environment:

```
poetry config --list
poetry config virtualenvs.in-project true
poetry shell
poetry install
```

To run the streamlit app:

```
poetry run streamlit run src/streamlit_app.py
```