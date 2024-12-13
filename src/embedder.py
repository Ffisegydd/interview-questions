from typing import List
from haystack import Document
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

document_embedder = OllamaDocumentEmbedder(
    model="mxbai-embed-large",
    progress_bar=False,
)


def embed_documents(documents: List[str] | List[Document]) -> List[Document]:
    if all(isinstance(doc, str) for doc in documents):
        documents = [Document(content=doc) for doc in documents]

    return document_embedder.run(documents)


if __name__ == "__main__":
    doc = Document(
        content="What is the difference between supervised and unsupervised learning?"
    )

    result = document_embedder.run([doc])
    print(result)
    print(doc.content)
    print(doc.embedding[:10])
