from math import hypot

from rag_config import get_embeddings


def main():
    words = ["apple", "iphone"]
    apple, iphone = get_embeddings().embed_documents(words)
    similarity = sum(left * right for left, right in zip(apple, iphone, strict=True)) / (
        hypot(*apple) * hypot(*iphone)
    )

    print(f"First five values for 'apple': {apple[:5]}")
    print(f"Vector length: {len(apple)}")
    print(f"Cosine similarity (apple, iphone): {similarity:.4f}")


if __name__ == "__main__":
    main()
