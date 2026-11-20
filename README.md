# LangChain RAG Chatbot

A Python command-line tutorial in retrieval-augmented generation (RAG). Ask a question about *Alice’s Adventures in Wonderland* and see which local document the chatbot used. Each command answers one question; there is no web interface or chat history.

## 1. Get the project and install dependencies

Use Python 3.12 and an OpenAI API key with access to the API. API calls may incur charges. Clone the repository or use **Code → Download ZIP** on GitHub, then open a terminal in the extracted folder.

```sh
git clone https://github.com/Hunter-Yan/Langchain-RAG-Chatbot-.git
cd Langchain-RAG-Chatbot-
```

On macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell), no activation is required:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` in place of `python` in the remaining commands.

## 2. Add your API key

Create a file named `.env` next to `requirements.txt`. Make sure it isn't saved as `.env.txt`.

```dotenv
OPENAI_API_KEY=replace_with_your_own_key
```

Use your own key from the [OpenAI API dashboard](https://platform.openai.com/api-keys). Never share or commit this file. The repository ignores `.env` files, virtual environments, and the local vector database.

## 3. Build the index and ask a question

```sh
python create_database.py
python query_data.py "How does Alice meet the Mad Hatter?"
```

The first command builds a local index in `chroma/`. The second retrieves passages, sends them with your question to the chat model, and prints an answer and source paths. The model is instructed to say when the passages don't contain enough information. To see the retrieved text, add `--show-context` to the query command.

To try your own documents, add UTF-8 `.md` files directly to `data/books/` and rerun `python create_database.py`. This replaces the generated index. Only use documents you're allowed to send to OpenAI: indexing sends document chunks to the embedding API, and querying sends your question and retrieved excerpts.

## What to read next

| File | What it shows |
| --- | --- |
| `create_database.py` | Read Markdown, split it into overlapping chunks, embed the chunks, and store them in Chroma. |
| `query_data.py` | Retrieve nearby chunks, assemble a prompt, and generate an answer. |
| `compare_embeddings.py` | Run `python compare_embeddings.py` to compare the vectors for `apple` and `iphone` using cosine similarity. |

The scripts use `text-embedding-3-small` for embeddings and `gpt-5.4-mini` for answers. If your API project uses another compatible chat model, set `OPENAI_CHAT_MODEL` in `.env`. If authentication fails or you hit a quota error, check your API key, project model access, and API billing.

Based on [Pixegami’s RAG and LangChain tutorial](https://www.youtube.com/watch?v=tcqEUSNCn8I). The sample book is [Project Gutenberg ebook #11](https://www.gutenberg.org/ebooks/11); its license notice is included in `data/books/alice_in_wonderland.md`.
