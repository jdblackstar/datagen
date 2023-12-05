import asyncio
import json
import os
import time

import aiohttp
import numpy as np
from dotenv import load_dotenv


class OpenAI_API:
    RETRY_ATTEMPTS = 5
    RETRY_DELAY = 2
    SAMPLES = 10

    def __init__(self):
        self.session = None
        self.throughput = np.zeros(self.SAMPLES)
        self.throughput_index = 0

    async def send_prompt(self, prompt, index, results_queue, inputs_queue):
        base_url = os.getenv("OPENAI_API_BASE")
        url = f"{base_url}/chat/completions"
        headers = self.get_headers()
        data = self.get_data(prompt)
        for retry in range(self.RETRY_ATTEMPTS):
            try:
                start_time = time.time()
                result, response_text = await self.post_request(url, headers, data)
                end_time = time.time()
                self.update_throughput(end_time - start_time)
                if self.is_special_response(response_text):
                    continue
                else:
                    await self.is_normal_response(
                        index, result, prompt, results_queue, inputs_queue
                    )
                break
            except Exception as e:
                print(f"Error sending prompt {index} (retry {retry + 1}):", e)
                if retry == self.RETRY_ATTEMPTS - 1:
                    print(
                        f"Failed to process prompt {index} after {self.RETRY_ATTEMPTS} retries."
                    )
                    return (index, None)
                await asyncio.sleep(self.RETRY_DELAY ** (retry + 1))

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }

    def get_data(self, prompt):
        return {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt["prompt"]}],
        }

    async def post_request(self, url, headers, data):
        async with self.session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result, json.dumps(result)

    def is_special_response(self, response_text):
        return any(
            phrase in response_text
            for phrase in [
                "%assistant%",
                "%I'm sorry%",
                "%I cannot%",
                "%language model%",
            ]
        )

    async def is_normal_response(
        self, index, result, prompt, results_queue, inputs_queue
    ):
        print(f"Model output for prompt {index}:", json.dumps(result))
        await results_queue.put((index, result))
        await inputs_queue.put((index, prompt))
        return (index, result)

    def update_throughput(self, elapsed_time):
        self.throughput[self.throughput_index] = 1 / elapsed_time
        self.throughput_index = (self.throughput_index + 1) % self.SAMPLES

    async def process_prompts(self, prompt_generator):
        results_queue = asyncio.Queue()
        inputs_queue = asyncio.Queue()

        asyncio.create_task(
            self.write_to_file(results_queue, "finished/complete.jsonl")
        )
        asyncio.create_task(
            self.write_to_file(inputs_queue, "finished/indexed_inputs.jsonl")
        )

        async with aiohttp.ClientSession() as self.session:
            batch = []
            for index, prompt in enumerate(prompt_generator):
                batch.append((index, prompt))
                if len(batch) >= self.SAMPLES:
                    await self.process_batch(batch, results_queue, inputs_queue)
                    batch = []
            if batch:
                await self.process_batch(batch, results_queue, inputs_queue)

        await results_queue.put(None)
        await inputs_queue.put(None)

    async def write_to_file(self, queue, file_name):
        with open(file_name, "w") as f:
            while True:
                item = await queue.get()
                if item is None:
                    break
                f.write(json.dumps(item) + "\n")  # new line because jsonl

    async def process_batch(self, batch, results_queue, inputs_queue):
        tasks = [
            self.send_prompt(prompt, index, results_queue, inputs_queue)
            for index, prompt in batch
        ]
        return await asyncio.gather(*tasks)


def ask_for_generations():
    while True:
        try:
            generations = int(input("How many generations would you like to perform? "))
            return generations
        except ValueError as e:
            print("Invalid input, please enter an integer.")


def prompt_generator(generations):
    with open("processed/preapi.jsonl", "r") as f:
        prompts = [json.loads(line) for line in f][:generations]
        for prompt in prompts:
            yield prompt


def main():
    api = OpenAI_API()
    generations = ask_for_generations()
    try:
        asyncio.run(api.process_prompts(prompt_generator(generations)))
    except Exception as e:
        print("An error occurred: ", e)


if __name__ == "__main__":
    main()
