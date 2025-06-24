import re
from exceptions import FormatServerException
from settings import FewShotMode
from dtos import ChatCompletionRequest, CompletionRequest, ChatMessage


def drop_think_data(text: str) -> str:
    assert text is not None
    st_pos = text.find('<think>')
    end_pos = text.find('</think>')

    if end_pos != -1:   # </think> ...
        st_pos = 0 if st_pos == -1 else st_pos  # ...<think>...</think>...
        return text[:st_pos] + text[end_pos+len('</think>'):]
    elif st_pos != -1:
        return text[:st_pos]    # ...<think>
    return text


def extract_boxed_content(text) -> str:
    match = re.search(r'\\boxed\{(.*)}', text, re.DOTALL)

    if match:
        _, end_pos = match.span(1)
        st_pos = text.find('\\boxed{')
        answer = match.group(1)
        return text[:st_pos] + answer + text[end_pos+1:]
    return text


def postprocess_output(text: str, math_mode:bool, think_data:bool=True) -> str:
    if text is None:
        return ""
    if not think_data:   # drop think data if necessary (only for completion endpoint)
        text = drop_think_data(text)
    if math_mode:
        text = extract_boxed_content(text)
        return text.strip()

    return text


def check_chat_format(request_dto: ChatCompletionRequest) -> None:
    messages = request_dto.messages

    if not isinstance(messages, list):
        raise FormatServerException("Input must be a list of chat messages.")
    if not messages:
        raise FormatServerException("Chat messages list is empty.")

    expected_roles = {'system', 'user', 'assistant'}
    system_prompt_seen = False
    expected_next = 'user'

    for i, message in enumerate(messages):
        role = message.role
        # check that role is in the expected roles set
        if role not in expected_roles:
            raise FormatServerException(f"Invalid role '{role}' at item {i}, only 'user', 'assistant', 'system' allowed.")

        if role == 'system':
            if system_prompt_seen:
                raise FormatServerException(f"Multiple system prompts found; second one at item {i}.")
            if i != 0:
                raise FormatServerException(f"System prompt must be the first message, found at item {i}.")
            if len(messages) == 1:
                raise FormatServerException("Only system prompt exists, no further dialog.")
            system_prompt_seen = True
        elif role != expected_next:
            raise FormatServerException(
                f"Expected '{expected_next}' role at item {i}, but got '{role}'."
            )
        expected_next = 'assistant' if role == 'user' else 'user'

    if messages[-1].role != 'user':
        raise FormatServerException(f"Last message must be from 'user', got '{messages[-1].role}'.")


def drop_few_shot(messages: list) -> list:
    result = []

    if len(messages) <= 1:
        return messages
    system_item = None if messages[0].role != 'system' else messages[0]
    assert messages[-1].role == 'user'

    if system_item is not None:
        result.append(system_item)
    result.append(messages[-1])

    return result


# returns one user message instead of chat dialog:
def join_few_shot(messages: list) -> list:
    if len(messages) < 2:
        return messages  # Нечего объединять

    system_prompt = messages[0].content if messages[0].role == 'system' else None
    start_idx = 1 if system_prompt else 0

    # Обработка случая из двух сообщений: system + user
    if system_prompt and len(messages) == 2:
        assert messages[1].role == 'user'
        content = (system_prompt + "\n" if system_prompt.strip() else "") + messages[1].content
        return [ChatMessage.model_validate({'role': 'user', 'content': content})]

    # Собираем диалог в один промпт
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    prompt_parts.append("Ниже приведён пример ожидаемого от тебя диалога, ответь на последний запрос пользователя, с учётом контекста:")

    expected_role = 'user'
    for msg in messages[start_idx:]:
        assert msg.role == expected_role
        if msg.role == 'user':
            prompt_parts.append(f"Пользователь: {msg.content}")
            expected_role = 'assistant'
        else:
            prompt_parts.append(f"Твой ответ: {msg.content}")
            expected_role = 'user'

    assert messages[-1].role == 'user'
    full_prompt = "\n".join(prompt_parts)
    return [ChatMessage.model_validate({'role': 'user', 'content': full_prompt})]


def preprocess_math(prompt: str, math_mode: bool) -> str:
    if math_mode:
        return "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n" + prompt
    return prompt
    

def preprocess_math_chat(messages:list, math_mode: bool) -> list:
    if not math_mode:
        return messages # no preprocessing
    result_messages = []

    for i in range(len(messages)):
        chat_item = messages[i]

        if chat_item.role == 'user':
            chat_item.content = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n" + chat_item.content
        result_messages.append(chat_item)

    return result_messages


def preprocess_few_shot(messages:list, few_shot_mode: FewShotMode) -> list:
    if few_shot_mode == FewShotMode.NO_PREPROCESS:
        return messages
    elif few_shot_mode == FewShotMode.DROP:
        messages = drop_few_shot(messages)
        return join_few_shot(messages)
    elif few_shot_mode == FewShotMode.PREPROCESS:
        return join_few_shot(messages)
    else:
        assert False
