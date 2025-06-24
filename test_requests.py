from openai import OpenAI
import openai
import httpx
import sys

# Modify OpenAI's API key and API base to use vLLM's API server.
default_openai_api_key = "token-abc123"
default_openai_api_base = "http://localhost:8000/v1"
default_model = "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16"


def openai_chat_request(test_name, messages, client, model):
    print(f"\n===== [CHAT] {test_name} =====\n")
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", "<no reasoning_content>")
        content = message.content
        print("reasoning_content:", reasoning_content)
        print("content:", content)
    except openai.BadRequestError as e:
        print("BadRequestError:", e.response.status_code)
        print(e.response.json())

def openai_chat_json_request(test_name, payload, api_key, api_base):    
    print(f"\n===== [CHAT] {test_name} =====\n")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }    
        response = httpx.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        message = response.json()['choices'][0]['message']
        if 'reasoning_content' in message:
            print("reasoning_content:", message['reasoning_content'])
        print("content:", message['content'])
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {str(exc)}; resp={str(response)}")
    except httpx.HTTPStatusError as exc:
        print(f'{str(exc)}; resp={str(response)}')


def openai_json_request(test_name, payload, api_key, api_base):    
    print(f"\n===== [COMPLETION] {test_name} =====\n")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }    
        response = httpx.post(
            f"{api_base}/completions",
            headers=headers,
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        message = response.json()['choices'][0]
        print("text:", message['text'])
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {str(exc)}; resp={str(response)}")
    except httpx.HTTPStatusError as exc:
        print(f'{str(exc)}; resp={str(response)}')


def openai_completion_request(test_name, prompt, client, model):
    print(f"\n===== [COMPLETION] {test_name} =====\n")
    try:
        response = client.completions.create(model=model, prompt=prompt, max_tokens=100)
        print("completion:", response.choices[0].text)
    except openai.BadRequestError as e:
        print("BadRequestError:", e.response.status_code)
        print(e.response.json())


def main():
    if len(sys.argv) <= 1:
        model = default_model
    else:
        model = sys.argv[1]
    if len(sys.argv) <= 2:
        openai_api_key = default_openai_api_key
    else:
        openai_api_key = sys.argv[2]
    if len(sys.argv) <= 3:
        openai_api_base = default_openai_api_base
    else:
        openai_api_base = sys.argv[3]

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # invalid fields and/or values:
    valid_messages = [{"role": "user", "content": "Hello who are you?"}]

    invalid_payloads = [
        # 1. Missing messages
        {"messages":[], "model": model},

        # 2. temperature too high
        {"messages": valid_messages, "model": model, "temperature": 2.5},

        # 3. top_p too low
        {"messages": valid_messages, "model": model, "top_p": 0.0},

        # 4. max_completion_tokens too small
        {"messages": valid_messages, "model": model, "max_completion_tokens": 0},

        # 5. n = 0
        {"messages": valid_messages, "model": model, "n": 0},

        # 6. presence_penalty too low
        {"messages": valid_messages, "model": model, "presence_penalty": -3.0},

        # 7. frequency_penalty too high
        {"messages": valid_messages, "model": model, "frequency_penalty": 3.5},

        # 8. logit_bias is not a dict
        {"messages": valid_messages, "model": model, "logit_bias": [50256, 50257]},

        # 9. stop is an integer
        {"messages": valid_messages, "model": model, "stop": 12345},

        # 10. tools contains a non-dict
        {"messages": valid_messages, "model": model, "tools": ["not-a-dict"]},
    ]

    for i, payload in enumerate(invalid_payloads):
        openai_chat_json_request(f'invalid payload {i}', payload, openai_api_key, openai_api_base)

    # invalid chat format:
    openai_chat_request("Error: Duplicate 'system' messages", [
        { "role": "system", "content": "ты ассистент для интеллектуальной викторины." },
        { "role": "system", "content": "кто ты?" }
    ], client, model)

    openai_chat_request("Error: Empty messages", [], client, model)

    openai_chat_request("Error: Starts with assistant", [
        { "role": "assistant", "content": "Привет" },
        { "role": "user", "content": "Кто ты?" }
    ], client, model)

    openai_chat_request("Error: No system message", [
        { "role": "user", "content": "Кто ты?" },
        { "role": "user", "content": "Сколько будет 1+1?" }
    ], client, model)

    # Normal cases
    openai_chat_request("Normal: Simple query", [
        { "role": "user", "content": "Сколько ног у лошади?"}
    ], client, model)

    openai_chat_request("Normal: Multi-turn history", [
        { "role": "system", "content": "ты ассистент для интеллектуальной викторины." },
        { "role": "user", "content": "Сколько будет 1+100?" },
        { "role": "assistant", "content": "101" },
        { "role": "user", "content": "Сколько ног у лошади?"}
    ], client, model)

    openai_chat_request("Normal: Medicine", [
        {"role": "system", "content":"Ты медицинский эксперт. Пользователь будет спрашивать тебя как решить определённые проблемы со здоровьем."},
        { "role": "user", "content": "Подскажи как справится с бессонницей, если много времени проводишь за гаджетами по работе и во время отдыха?" }
    ], client, model)

    openai_chat_request("Normal: Algorithms", [
        { "role": "user", "content": "Проверьте, сбалансирована ли входная последовательность скобок:\n} } ) [ } ] ) { [ { { ] ( ( ] ) ( ) [ {\nВыведите 1, если да и 0 в противном случае." }
    ], client, model)

    openai_chat_request("Normal: Logic MCQ", [
        { "role": "system", "content": "ты ассистент для интеллектуальной викторины." },
        { "role": "user", "content": "Выбери один правильный вариант ответа. Напиши только букву ответа. Кто использует кровеносную систему?\nA. лошадь\nB. дерево\nC. машина\nD. скала\nОтвет:" }
    ], client, model)

    openai_chat_request("Normal: Code", [
        { "role": "user", "content": "напиши алгоритм сортировки слиянием на C++" }
    ], client, model)

    # test math:
    openai_chat_json_request("Normal: Math (arithmetic):", {"messages":[
        { "role": "system", "content": "ты ассистент для решения задач по математике." },
        { "role": "user", "content": "сколько будет 13243 * 5 - 6" }], 'r1_settings':{'math_mode': True}, "model": model
    }, 
    openai_api_key, 
    openai_api_base
    )

    openai_chat_json_request("Normal: Math (geometry):", {"messages":[
        { "role": "system", "content": "ты ассистент для решения задач по математике." },
        { "role": "user", "content": "длина 1 катета равна 10, длина второго 12, чему будет равна длина гипотенузы?" }], 
        "model": model,
        'r1_settings':{'math_mode': True}
    },
    openai_api_key, 
    openai_api_base
    )

    openai_chat_json_request("Normal: Math (bool algebra):", {"messages":[
        { "role": "system", "content": "ты ассистент для решения задач по математике." },
        { "role": "user", "content": "a=true, b=false, c=true, чему будет равно !a | b ^ c" }], 
        "model": model, 
        'r1_settings':{'math_mode': True}
    },
    openai_api_key, 
    openai_api_base
    )

    openai_chat_json_request("Normal: Multi-turn history (join few-shot)", {"messages":[
        { "role": "system", "content": "ты ассистент для интеллектуальной викторины." },
        { "role": "user", "content": "Сколько будет 1+100?" },
        { "role": "assistant", "content": "101" },
        { "role": "user", "content": "Сколько ног у лошади?"}], 
        'r1_settings':{'math_mode': True, 'few_shot_mode':"PREPROCESS"}
    },
    openai_api_key, 
    openai_api_base
    )

    # ───── Completion Endpoint Tests ─────
    openai_json_request("Simple math", {"prompt":"Сколько будет 1458 + 293?\nОтвет:", 'r1_settings':{'math_mode': True}}, openai_api_key, openai_api_base)
    openai_completion_request("Trivial question", "Кто написал роман «Война и мир»? Сколько в нём страниц\nОтвет:", client, model)


if __name__ == '__main__':
    main()
