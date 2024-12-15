import logging
from typing import Dict, List
from haystack import Document, component
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from consts import EmbeddedQuestion, Question
import uuid


@component
class ClusterQuestions:
    def __init__(self, ollama_model_name: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initialising instance")
        self.embedder = OllamaDocumentEmbedder(
            model=ollama_model_name,  # type: ignore
            progress_bar=False,  # type: ignore
        )  # type: ignore

        self.clusters: Dict[str, List[EmbeddedQuestion]] = {}

    @component.output_types(clusters=List[List[EmbeddedQuestion]])
    def run(self, questions: List[Question], similarity_threshold: float = 0.9):
        self.logger.info("Beginning run")
        embedded_questions = self._embed_questions(questions)

        clustered_questions = self._cluster_questions(
            embedded_questions, similarity_threshold
        )

        clustered_questions = [
            self._clear_embeddings(cluster) for cluster in clustered_questions.values()
        ]

        self.logger.info(f"# of clusters generated: {len(clustered_questions)}")

        return {"clusters": list(clustered_questions)}

    def _embed_questions(self, questions: List[Question]) -> List[EmbeddedQuestion]:
        documents = [Document(content=q.question) for q in questions]
        embeddings = self.embedder.run(documents)["documents"]

        # embeddings = []
        # for question in questions:
        #     embedding = self.embedder.run(question.question)["embedding"]
        #     embeddings.append(embedding)

        embedded_questions: List[EmbeddedQuestion] = []
        for question, embedding in zip(questions, embeddings):
            embedded_questions.append(
                EmbeddedQuestion(
                    question_embedding=embedding.embedding,
                    question=question.question,
                    topic=question.topic,
                    sub_topic=question.sub_topic,
                )
            )

        return embedded_questions

    def _cluster_questions(
        self, questions: List[EmbeddedQuestion], similarity_threshold
    ) -> Dict[str, List[EmbeddedQuestion]]:
        for question in questions:
            self._add_to_cluster(question, similarity_threshold)

        return self.clusters

    def _add_to_cluster(self, question: EmbeddedQuestion, similarity_threshold):
        for cluster_key, cluster in self.clusters.items():
            cluster_embeddings = np.array([d.question_embedding for d in cluster])

            # Check similarity with existing documents
            similarities = cosine_similarity(
                np.array([question.question_embedding]), cluster_embeddings
            )[0]
            max_similarity = max(similarities)

            # Add document if sufficiently similar to an existing cluster
            if max_similarity >= similarity_threshold:
                self.clusters[cluster_key].append(question)
                return

        self.clusters[str(uuid.uuid4())] = [question]
        return

    def _clear_embeddings(self, questions: List[EmbeddedQuestion]) -> List[Question]:
        return [
            Question(
                question=question.question,
                topic=question.topic,
                sub_topic=question.sub_topic,
            )
            for question in questions
        ]


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
