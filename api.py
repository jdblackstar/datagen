import asyncio
import json
import os

import aiohttp
from dotenv import load_dotenv


async def call_openai(data):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=json.dumps(data), headers=headers) as resp:
            response = await resp.text()
            print(response)


def main():
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a council of experts. Two of you are experts in the field, and one of you is a novice who asks questions.",
            },
            {"role": "user", "content": "What is the theory of relativity?"},
        ],
    }

    loop = asyncio.get_event_loop()
    loop.run_until_complete(call_openai(data))


if __name__ == "__main__":
    main()
