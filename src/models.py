import os

from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.utils import Secret

from enum import Enum


class ModelType(Enum):
    LOCAL_SLOW = "llama3.3"
    LOCAL_FAST = "wizard-vicuna"
    REMOTE = "openai"


def create_llm_model(model_type: ModelType):
    """
    Create a new instance of OllamaGenerator with the specified model type.

    :param model_type: The type of the model to use for the OllamaGenerator.
    :return: An instance of OllamaGenerator.
    """
    model_name = model_type.value

    if model_type == ModelType.REMOTE:
        openai_api_key = os.getenv("OPENAI_API_KEY", None)
        assert (
            openai_api_key is not None
        ), "Please set the OPENAI_API_KEY environment variable."
        return OpenAIGenerator(
            model="gpt-4o-mini",  # type: ignore
            api_key=Secret.from_token(openai_api_key),  # type: ignore
        )  # type: ignore
    else:
        return OllamaGenerator(
            model=model_name,  # type: ignore
            url="http://localhost:11434",  # type: ignore
            generation_kwargs={  # type: ignore
                "num_predict": 1000,
                "temperature": 0.9,
            },
            keep_alive=60 * 20,  # 20 minutes # type: ignore
        )  # type: ignore
