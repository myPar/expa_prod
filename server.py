from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from settings import app_settings
from contextlib import asynccontextmanager
from exceptions import FatalServerException, FormatServerException
from tools import *
from pydantic import ValidationError
import sys
from dtos import CompletionRequest, ChatCompletionRequest

# global http client reference for reuse
client = None  

# authorization headers for model vllm server:
vllm_auth_headers = {
        "Authorization": f"Bearer {app_settings.server_settings.openai_api_key}",
        "Content-Type": "application/json",
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=app_settings.server_settings.client_timeout)
    yield
    await client.aclose()

app = FastAPI(lifespan=lifespan)


# legacy api, <think> token can't be added manually, only model itself will do it or not
@app.post("/v1/completions")
async def proxy_completions(request: Request) -> JSONResponse:
    body = await request.json()
    try:
        request_dto = CompletionRequest.model_validate(body)
    except ValidationError as e:
        return FormatServerException(str(e)).get_json()
    # math preprocessing:
    request_dto.prompt = preprocess_math(prompt=request_dto.prompt, math_mode=request_dto.r1_settings.math_mode)
    try:    # request to vllm server:
        response = await client.post(f"{app_settings.server_settings.vllm_server_url}/v1/completions", 
                                    json=request_dto.model_dump(exclude={'r1_settings'}), headers=vllm_auth_headers)
    except httpx.HTTPError as vllm_ex:
        return FatalServerException(str(vllm_ex)).get_json()
            
    result = response.json()
    print(f'result={result}')
    # Postprocess output
    if "choices" in result:
        for choice in result["choices"]:
            choice["text"] = postprocess_output(choice["text"], 
                                                request_dto.r1_settings.math_mode,
                                                think_data=request_dto.r1_settings.return_think_data
                                                )
    else:
        return FatalServerException(f"no messages found in the vllm server 'completions' response: {str(result)}").get_json()
    return JSONResponse(content=result, status_code=response.status_code)


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> JSONResponse:
    # validate request:
    body = await request.json()
    try:
        request_dto = ChatCompletionRequest.model_validate(body)
    except ValidationError as e:
        return FormatServerException(str(e)).get_json()
    try:
        check_chat_format(request_dto)
        # first - preprocess math
        request_dto.messages = preprocess_few_shot(messages=request_dto.messages, few_shot_mode=request_dto.r1_settings.few_shot_mode)
        request_dto.messages = preprocess_math_chat(messages=request_dto.messages, math_mode=request_dto.r1_settings.math_mode)
    except FormatServerException as e:
        # format is invalid - no preprocessing
        if app_settings.server_settings.proper_chat_format:
            return e.get_json()
    try:    # request to vllm server:
        response = await client.post(f"{app_settings.server_settings.vllm_server_url}/v1/chat/completions", 
                                    json=request_dto.model_dump(exclude={'r1_settings'}),
                                    headers=vllm_auth_headers
                                    )
    except httpx.HTTPError as vllm_ex:
        return FatalServerException(str(vllm_ex)).get_json()
    result = response.json()
    print(f'result={result}')
    # Postprocess output
    if "choices" in result:
        for choice in result["choices"]:
            if "message" in choice:
                if "reasoning_content" in choice["message"] and not request_dto.r1_settings.return_think_data: # remove think data field from the response
                    choice["message"].pop("reasoning_content")
                choice["message"]["content"] = postprocess_output(choice["message"]["content"], 
                                                                  request_dto.r1_settings.math_mode
                                                                  )
    else:
        return FatalServerException(f"no messages found in the vllm server 'chat/completions' response: {str(result)}").get_json()
    return JSONResponse(content=result, status_code=response.status_code)


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"message": f"pong"}
