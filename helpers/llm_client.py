import os

import requests
from deepeval.models import DeepEvalBaseLLM


class LocalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model: str = "meta-llama-3.1-8b-instruct",
    ):
        self.base_url = os.getenv("LLM_BASE_URL", base_url).rstrip("/")
        self.model = os.getenv("LLM_MODEL", model)
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "120"))

    def get_model_name(self) -> str:
        return self.model

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant used for automated LLM testing. "
                    "Always answer the user's request directly in natural language. "
                    "Never call tools. Never produce function calls. "
                    "Never return tool-call JSON unless a JSON schema is explicitly provided."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        # DeepEval uses JSON schema when it needs structured output.
        if schema is not None:
            if hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            elif hasattr(schema, "schema") and callable(schema.schema):
                schema = schema.schema()

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "deepeval_schema",
                    "schema": schema,
                },
            }

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        content = message.get("content")

        if not content:
            raise RuntimeError(
                "LM Studio returned no message content. "
                f"Full message: {message}"
            )

        return content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)