# hackathon-dc-aegis

## How to run the project

Setup your local (virtual) environment:

```
poetry config --list
poetry config virtualenvs.in-project true
poetry shell
poetry install
```

To run the streamlit app:

```
poetry run streamlit src/streamlit_app.py
```