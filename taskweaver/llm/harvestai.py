import os
import time

import jsdata
from jsdata.llm import HarvestAIClient

from typing import Any, Generator, List, Optional
from injector import inject

from taskweaver.llm.base import CompletionService, EmbeddingService, LLMServiceConfig
from taskweaver.llm.util import ChatMessageType, format_chat_message, ChatMessageRoleType


class HarvestServiceConfig(LLMServiceConfig):
    def _configure(self) -> None:
        self._set_name("harvest")
        # shared common config

        shared_api_key = self.llm_module_config.api_key
        self.api_key = self._get_str(
            "api_key",
            shared_api_key,
        )

        shared_model = self.llm_module_config.model
        self.model = self._get_str(
            "model",
            shared_model if shared_model is not None else "gpt-4",
        )

        shared_embedding_model = self.llm_module_config.embedding_model
        self.embedding_model = self._get_str(
            "embedding_model",
            shared_embedding_model if shared_embedding_model is not None else "BGE",
        )

        self.response_format = self.llm_module_config.response_format

        self.temperature = self._get_float("temperature", 0)
        self.max_tokens = self._get_int("max_tokens", 1024)
        self.top_p = self._get_float("top_p", 0)
        self.top_k = self._get_int("top_k", 0)


class HarvestService(CompletionService, EmbeddingService):

    @inject
    def __init__(self, config: HarvestServiceConfig):
        self.config = config
        jsdata.set_token(self.config.api_key)
        self.api = jsdata.get_api()
        self.client = HarvestAIClient()

    def chat_completion(
            self,
            messages: List[ChatMessageType],
            stream: bool = False,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            logprobs: bool = True,
            **kwargs: Any,
    ) -> Generator[ChatMessageType, None, None]:
        engine = self.config.model

        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        top_p = top_p if top_p is not None else self.config.top_p

        try:
            tools_kwargs = {}
            if "tools" in kwargs and "tool_choice" in kwargs:
                tools_kwargs["tools"] = kwargs["tools"]
                tools_kwargs["tool_choice"] = kwargs["tool_choice"]
            if "response_format" in kwargs:
                response_format = kwargs["response_format"]
            elif self.config.response_format == "json_object":
                response_format = {"type": "json_object"}
            else:
                response_format = None

            if stream:
                response = self.client.chat_stream(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                role = "assistant"
                for stream_res in response:
                    yield format_chat_message(role, stream_res)
                    time.sleep(0.01)
            else:
                response = self.client.chat_query(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                role = "assistant"
                yield format_chat_message(role, response)

        except Exception as e:
            print(e)

    def get_embeddings(self, strings: List[str]) -> List[List[float]]:
        # TODO
        embedding_model = self.config.embedding_model
        if embedding_model == "BGE":
            embedding_results = self.api.llm_bge_embedding(
                input=strings,
                limit=10
            ).data
        elif embedding_model == "M3E":
            embedding_results = self.api.llm_m3e_embedding(
                input=strings,
                limit=10
            ).data
        else:
            raise Exception(f"Invalid embedding model: {embedding_model}")
        return [r.embedding_results[0][0]["embedding"] for r in embedding_results]
