from typing import Union, Literal
from pydantic import BaseModel, Field, model_validator
from settings import app_settings

model_config = app_settings.default_model_settings


class R1Settings(BaseModel):
    # custom r1 fields:
    math_mode: bool | None = False    # pre and post process
    return_think_data: bool | None = False


# /v1/completions model
class CompletionRequest(BaseModel):
    prompt: Union[str, list[str]]
    temperature: float | None = Field(default=model_config.temperature, ge=0.0, le=2.0)
    max_completion_tokens: int | None = Field(default=model_config.max_completion_tokens, ge=1)
    top_p: float | None = Field(default=model_config.top_p, gt=0.0, le=1.0)
    stop: Union[str, list[str]] | None = None
    n: int | None = Field(default=1, ge=1)
    stream: bool | None = False
    logprobs: int | None = None
    echo: bool | None = False
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: int | None = None
    logit_bias: dict | None = None
    user: str | None = None
    r1_settings: R1Settings | None = R1Settings()


# /v1/chat/completions model
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class R1SettingsChat(BaseModel):
    # custom r1 fields:
    math_mode: bool | None = False    # pre and post process
    return_think_data: bool | None = True
    few_shot_mode: Literal["DROP", "PREPROCESS", "NO_PREPROCESS"] = "NO_PREPROCESS"


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = Field(default=model_config.temperature, ge=0.0, le=2.0)
    max_completion_tokens: int | None = Field(default=model_config.max_completion_tokens, ge=1)
    top_p: float | None = Field(default=model_config.top_p, gt=0.0, le=1.0)
    stop: Union[str, list[str]] | None = []
    n: int | None = Field(default=1, ge=1)
    stream: bool | None = False
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict | None = None
    user: str | None = None
    tools: list[dict] | None = None  # You can expand this if your API supports tools
    # custom r1 fields:
    r1_settings: R1SettingsChat | None = R1SettingsChat()


    @model_validator(mode='after')
    def check_messages(self):
        if not self.messages or not isinstance(self.messages, list) or len(self.messages) < 1:
            raise ValueError("messages field must be a non-empty list")
        if self.tools is not None:
            raise ValueError("tools calling are not supported in vllm backend for r1 model series")
        return self
