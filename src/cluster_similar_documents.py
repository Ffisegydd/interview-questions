from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from consts import Question
from embedder import embed_documents
import uuid


# Function to calculate similarity and decide on adding a document
def add_document_if_not_similar(
    doc: Question, existing_docs: dict[str, List[Question]], similarity_threshold=0.85
) -> dict[str, List[Question]]:
    # Embed the new document
    doc_embedding = embed_documents([doc.question])["documents"][0].embedding
    doc.question_embedding = doc_embedding

    # Retrieve all existing document embeddings
    for cluster_key, cluster in existing_docs.items():
        cluster_embeddings = np.array([d.question_embedding for d in cluster])

        # Check similarity with existing documents
        similarities = cosine_similarity([doc_embedding], cluster_embeddings)[0]
        max_similarity = max(similarities)

        if max_similarity >= similarity_threshold:
            existing_docs[cluster_key].append(doc)
            return existing_docs

    # Add the document if not similar
    existing_docs[str(uuid.uuid4())] = [doc]
    return existing_docs


def cluster_documents(documents: List[Question], similarity_threshold=0.9):
    existing_docs: dict[str, List[Question]] = {}

    for doc in documents:
        add_document_if_not_similar(doc, existing_docs, similarity_threshold)
    return existing_docs


if __name__ == "__main__":
    questions = [
        "What is the difference between supervised and unsupervised learning?",
        "What is the difference between unsupervised and supervised learning?",
        "How does a neural network work?",
        "Could you explain how neural nets work for me please?",
        "What are the common metrics used to evaluate a machine learning model?",
        "Can you explain the concept of overfitting and how to prevent it?",
        "What is the role of a loss function in machine learning?",
        "How do you handle missing data in a dataset?",
        "Is missing data a problem in a dataset?",
        "What are the advantages and disadvantages of using decision trees?",
        "How does the k-means clustering algorithm work?",
        "What is the purpose of cross-validation in machine learning?",
        "Can you explain the bias-variance tradeoff?",
        "What is the difference between bagging and boosting?",
        "How do convolutional neural networks (CNNs) work?",
        "What is the purpose of regularization in machine learning?",
        "Can you explain the concept of gradient descent?",
        "What are support vector machines (SVMs) and how do they work?",
        "How do you evaluate the performance of a clustering algorithm?",
        "What is the difference between precision and recall?",
        "How does the random forest algorithm work?",
        "What is the purpose of the activation function in a neural network?",
        "Can you explain the concept of feature scaling?",
        "What is the difference between L1 and L2 regularization?",
        "How does the AdaBoost algorithm work?",
        "What are the different types of neural networks?",
        "Can you explain the concept of transfer learning?",
        "What is the purpose of dropout in neural networks?",
        "How do you choose the number of clusters in k-means?",
        "What is the difference between a generative and a discriminative model?",
        "How does the principal component analysis (PCA) algorithm work?",
        "What are the common activation functions used in neural networks?",
        "Can you explain the concept of reinforcement learning?",
        "What is the purpose of the softmax function in neural networks?",
        "How do you handle imbalanced datasets?",
        "What is the difference between a convolutional layer and a pooling layer?",
        "How does the gradient boosting algorithm work?",
        "What are the different types of clustering algorithms?",
        "Can you explain the concept of a confusion matrix?",
        "What is the purpose of the learning rate in gradient descent?",
        "How do you evaluate the performance of a regression model?",
        "What is the difference between a parametric and a non-parametric model?",
        "How does the t-SNE algorithm work?",
    ]

    documents = [
        Question(question=q, topic="machine learning", sub_topic="ml engineering")
        for q in questions
    ]

    clustered_documents = cluster_documents(documents, similarity_threshold=0.85)

    for key, value in list(
        sorted(clustered_documents.items(), key=lambda x: -len(x[1]))
    )[:5]:
        print(f"Cluster: {key}")
        for doc in value:
            print(doc.question)
        print()
