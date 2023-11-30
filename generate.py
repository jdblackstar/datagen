import json
import os
from itertools import cycle
from pathlib import Path

TEMPLATES_FOLDER = "templates"


def load_supporting_data(input_file):
    supporting_data = {}
    with open(input_file, "r") as file:
        for line in file:
            item = json.loads(line)
            for key, value in item.items():
                if key not in supporting_data:
                    supporting_data[key] = [value]
                else:
                    supporting_data[key].append(value)
    for key, values in supporting_data.items():
        supporting_data[key] = cycle(values)
    return supporting_data


def load_templates(folder=TEMPLATES_FOLDER):
    templates = []
    template_folder = Path(folder)
    for file_path in template_folder.glob("*.txt"):
        with open(file_path, "r") as file:
            templates.append(file.read().strip())
    return templates


def generate_prompts_with_supporting_data(
    input_file, supporting_data_file, output_file="prompts.jsonl"
):
    supporting_data = load_supporting_data(supporting_data_file)
    templates = cycle(load_templates())

    with open(input_file, "r") as input_file, open(output_file, "w") as output_file:
        data = json.load(input_file)
        for item in data:
            print(f"Processing item: {item}")
            template = next(templates)
            prompt = template
            if not isinstance(item, dict):
                print(f"Unexpected item: {item}")
            else:
                main_data_unwrapped = {k: str(v) for k, v in item.items()}
                combined_data = {
                    **main_data_unwrapped,
                    **{k: str(next(v)) for k, v in supporting_data.items()},
                }
                for key, value in combined_data.items():
                    prompt = prompt.replace(f"{{{key}}}", value)
                prompt_object = {"prompt": prompt}
                output_file.write(json.dumps(prompt_object) + "\n")


def main():
    # directories and files stuff
    data_folder_path = "data"
    preapi_folder_path = "preapi"
    input_jsonl_path = os.path.join(data_folder_path, "data.json")
    supporting_data_jsonl_path = None
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(preapi_folder_path, exist_ok=True)
    output_jsonl_path = os.path.join(preapi_folder_path, "preapi.jsonl")

    for file_name in os.listdir(data_folder_path):
        if file_name.endswith(".jsonl") and file_name != "data.jsonl":
            supporting_data_jsonl_path = os.path.join(data_folder_path, file_name)
            break
    if supporting_data_jsonl_path is None:
        print("No supporting data file found in the 'data' folder.")
        return

    generate_prompts_with_supporting_data(
        input_jsonl_path, supporting_data_jsonl_path, output_jsonl_path
    )


if __name__ == "__main__":
    main()
