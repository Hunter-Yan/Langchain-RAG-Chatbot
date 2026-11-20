import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_config import CHROMA_PATH, DATA_PATH, PROJECT_ROOT, get_embeddings


def load_documents():
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(PROJECT_ROOT).as_posix()},
        )
        for path in sorted(DATA_PATH.glob("*.md"))
        if path.is_file()
    ]


def main():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise SystemExit("No text to index. Add a nonempty .md file to data/books/.")

    embeddings = get_embeddings()
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    Chroma.from_documents(chunks, embeddings, persist_directory=str(CHROMA_PATH))
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents in chroma/.")


if __name__ == "__main__":
    main()
