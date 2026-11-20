# Langchain RAG Tutorial

## Install dependencies

Use Python 3.12. From the project folder, create a virtual environment and install the dependencies.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and run the commands through `.venv\Scripts\python.exe`.

Create a `.env` file in the project folder with your OpenAI API key.

```dotenv
OPENAI_API_KEY=replace_with_your_own_key
```

## Create database

Create the Chroma DB.

```sh
python create_database.py
```

## Query the database

Query the Chroma DB.

```sh
python query_data.py "How does Alice meet the Mad Hatter?"
```

The output shows the prompt with up to five retrieved chunks, followed by the answer and source paths. Add `--show-context` to print the chunks separately as well.

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
