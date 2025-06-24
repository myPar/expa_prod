from pydantic import BaseModel
import pydantic_core
from enum import Enum


class FewShotMode(str, Enum):
    NO_PREPROCESS = "NO_PREPROCESS"
    PREPROCESS = "PREPROCESS"
    DROP = "DROP"


class ModelSettings(BaseModel):
    temperature: float
    stop: list
    top_p: float
    max_completion_tokens: int


class ServerSettings(BaseModel):
    vllm_server_url: str
    openai_api_key: str
    proper_chat_format: bool # if the chat messages order/format is invalid and this field is enabled an error response will be returned
    client_timeout: int # timeout in seconds for the http client


class AppSettings(BaseModel):
    server_settings: ServerSettings
    default_model_settings: ModelSettings


with open('settings.json', encoding='utf-8') as f:
    json_data = f.read()


app_settings = AppSettings.model_validate(pydantic_core.from_json(json_data))
