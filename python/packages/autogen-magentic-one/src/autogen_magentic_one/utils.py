import json
import logging
import os
import base64
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal

from autogen_core.application.logging.events import LLMCallEvent
from autogen_core.components import Image
from autogen_core.components.models import (
    ChatCompletionClient,
    ModelCapabilities,
    AssistantMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from .messages import (
    AgentEvent,
    AssistantContent,
    FunctionExecutionContent,
    OrchestrationEvent,
    SystemContent,
    UserContent,
    WebSurferEvent,
)

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2-vision"
    temperature: float = 0.7
    top_p: float = 0.95

class OllamaChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        config: OllamaConfig = OllamaConfig(),
        **kwargs: Any
    ):
        self.config = config
        self.kwargs = kwargs
        
        self.model_capabilities = ModelCapabilities(
            vision=True,
            function_calling=True,
            json_output=True,
        )

    def extract_role_and_content(self, msg) -> (str, Union[str, List[Union[str, Image]]]):
        """Helper function to extract role and content from various message types."""
        if isinstance(msg, SystemMessage):
            return 'system', msg.content
        elif isinstance(msg, UserMessage):
            return 'user', msg.content
        elif isinstance(msg, AssistantMessage):
            return 'assistant', msg.content
        elif hasattr(msg, 'role') and hasattr(msg, 'content'):
            return msg.role, msg.content
        else:
            return 'user', str(msg)

    def process_message_content(self, content):
        text_parts = []
        images = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, Image):
                    image_data = self.encode_image(item)
                    if image_data:
                        images.append(image_data)
                else:
                    text_parts.append(str(item))
        else:
            text_parts.append(str(content))
        return '\n'.join(text_parts), images

    def encode_image(self, image: Image) -> Optional[str]:
        """Encodes an Image object to a base64 string."""
        try:
            if hasattr(image, 'path'):
                with open(image.path, 'rb') as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            elif hasattr(image, 'data'):
                return base64.b64encode(image.data).decode('utf-8')
            else:
                return None
        except Exception as e:
            # Log or handle the error as needed
            return None

    async def create(
        self,
        messages: List[LLMMessage],
        *,
        response_format: Optional[Dict[str, str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> AssistantMessage:
        """Create a completion using the Ollama API."""
        import aiohttp

        chat_messages = []
        for msg in messages:
            role, content = self.extract_role_and_content(msg)
            text, images = self.process_message_content(content)
            chat_message = {
                "role": role,
                "content": text
            }
            if images:
                chat_message["images"] = images
            chat_messages.append(chat_message)

        request_data = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/chat",
                    json=request_data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return AssistantMessage(
                            content=f"Error: Ollama API error: {error_text}",
                            source='assistant'
                        )

                    result = await response.json()
                    return AssistantMessage(
                        content=result.get("message", {}).get("content", "Error: No response content"),
                        source='assistant'
                    )
        except Exception as e:
            return AssistantMessage(
                content=f"Error: Failed to get response from Ollama server: {str(e)}",
                source='assistant'
            )

def message_content_to_str(
    message_content: Union[UserContent, AssistantContent, SystemContent, FunctionExecutionContent, AssistantMessage],
) -> str:
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, AssistantMessage):
        return message_content.content
    elif isinstance(message_content, List):
        converted: List[str] = []
        for item in message_content:
            if isinstance(item, str):
                converted.append(item.rstrip())
            elif isinstance(item, Image):
                converted.append("<Image>")
            elif isinstance(item, AssistantMessage):
                converted.append(item.content.rstrip())
            else:
                converted.append(str(item).rstrip())
        return "\n".join(converted)
    else:
        raise AssertionError("Unexpected response type.")

def create_completion_client_from_env(env: Optional[Dict[str, str]] = None, **kwargs: Any) -> ChatCompletionClient:
    """Create a model client based on environment variables."""
    if env is None:
        env = dict()
        env.update(os.environ)

    _kwargs = json.loads(env.get("MODEL_CONFIG", "{}"))
    _kwargs.update(kwargs)

    return OllamaChatCompletionClient(
        config=OllamaConfig(
            base_url=_kwargs.pop("base_url", OllamaConfig.base_url),
            model=_kwargs.pop("model", OllamaConfig.model),
            temperature=_kwargs.pop("temperature", OllamaConfig.temperature),
            top_p=_kwargs.pop("top_p", OllamaConfig.top_p),
        ),
        **_kwargs
    )

class LogHandler(logging.FileHandler):
    """MagenticOne log event handler."""
    def __init__(self, filename: str = "log.jsonl") -> None:
        super().__init__(filename)
        self.logs_list: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).isoformat()
            if isinstance(record.msg, OrchestrationEvent):
                console_message = (
                    f"\n{'-'*75} \n" f"\033[91m[{ts}], {record.msg.source}:\033[0m\n" f"\n{record.msg.message}"
                )
                print(console_message, flush=True)
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": record.msg.message,
                        "type": "OrchestrationEvent",
                    }
                )
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            elif isinstance(record.msg, AgentEvent):
                console_message = (
                    f"\n{'-'*75} \n" f"\033[91m[{ts}], {record.msg.source}:\033[0m\n" f"\n{record.msg.message}"
                )
                print(console_message, flush=True)
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": record.msg.message,
                        "type": "AgentEvent",
                    }
                )
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            elif isinstance(record.msg, WebSurferEvent):
                console_message = f"\033[96m[{ts}], {record.msg.source}: {record.msg.message}\033[0m"
                print(console_message, flush=True)
                payload: Dict[str, Any] = {
                    "timestamp": ts,
                    "type": "WebSurferEvent",
                }
                payload.update(asdict(record.msg))
                record.msg = json.dumps(payload)
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            elif isinstance(record.msg, LLMCallEvent):
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "prompt_tokens": record.msg.prompt_tokens,
                        "completion_tokens": record.msg.completion_tokens,
                        "type": "LLMCallEvent",
                    }
                )
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
        except Exception:
            self.handleError(record)

class SentinelMeta(type):
    def __repr__(cls) -> str:
        return f"<{cls.__name__}>"

    def __bool__(cls) -> Literal[False]:
        return False
