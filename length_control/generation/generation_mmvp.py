import os
import json
import argparse
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def load_model_and_processor(model_path, device='cuda:0'):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    model.eval()
    return model, processor


def extract_thinking_tags(text, start_tag="<think>", end_tag="</think>"):
    match = re.search(f"({re.escape(start_tag)}.*?{re.escape(end_tag)})", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        return thinking
    return ""


def generate_response(model, processor, image_path, question, device='cuda:0'):
    image = processor.image_processor(images=image_path, return_tensors="pt")["pixel_values"].to(device)

    prompt = f"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags.\n{question}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def process_dataset(model, processor, csv_file, output_json, image_dir, num_samples=100):
    df = pd.read_csv(csv_file)[:num_samples]

    results = []
    thinking_lengths = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_id = str(row['Index'])
        image_path = os.path.join(image_dir, f"{image_id}.jpg")

        question = f"{row['Question']} {row['Options']}"
        answer = row['Correct Answer']

        response = generate_response(model, processor, image_path, question)
        thinking = extract_thinking_tags(response)
        thinking_token_len = len(processor.tokenizer(thinking, return_tensors="pt")["input_ids"][0]) if thinking else 0

        result = {
            "index": image_id,
            "image": image_path,
            "question": row['Question'],
            "options": row['Options'],
            "correct_answer": answer,
            "model_response": response,
            "thinking": thinking,
            "thinking_length": thinking_token_len
        }
        thinking_lengths.append(thinking_token_len)
        results.append(result)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.hist(thinking_lengths, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Thinking Length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Thinking Lengths")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(output_json), "thinking_length_distribution.png"))

    print(f"Saved {len(results)} responses to {output_json}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="/data/MMVP/Questions.csv")
    parser.add_argument('--image_dir', type=str, default="/data/MMVP/Images")
    parser.add_argument('--output', type=str, default="results/openvl_output.json")
    parser.add_argument('--model_path', type=str, default="/path/to/any-VLM-model")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_path, args.device)
    process_dataset(model, processor, args.csv, args.output, args.image_dir, args.num_samples)