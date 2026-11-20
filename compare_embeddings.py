from math import hypot

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

def main():
    # Get embedding for a word.
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    words = ("apple", "iphone")
    other_vector = embedding_function.embed_query(words[1])
    dot_product = sum(left * right for left, right in zip(vector, other_vector, strict=True))
    distance = 1 - dot_product / (hypot(*vector) * hypot(*other_vector))
    print(f"Cosine distance ({words[0]}, {words[1]}): {distance:.4f}")


if __name__ == "__main__":
    main()
