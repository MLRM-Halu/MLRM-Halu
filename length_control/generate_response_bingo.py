
import gc
import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import time

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)


def load_model(model_id):
    """Load the model and processor."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:2").eval()
    return model, processor


def extract_thinking(response):
    """Extracts the thinking part from response text, including the <think> tags."""
    match = re.search(r"(<think>.*?</think>)", response, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        return thinking_text, len(processor.tokenizer(thinking_text, return_tensors='pt')['input_ids'][0])
    return "", -1


def get_response(model, processor, image_path, question):
    """Process a single image and question pair to get model response."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",
                 "text": f"{question}You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."},

            ],
        }
    ]


    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def process_jsonl(model, processor, input_jsonl, output_jsonl, num_samples=None):
    """Process Bingo dataset (jsonl format) with thinking analysis"""
    image_root = os.path.dirname(input_jsonl)
    thinking_lengths = []
    responses_data = []

    os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)

    with open(input_jsonl, 'r') as infile, open(output_jsonl, 'w') as outfile:
        for idx, line in enumerate(infile):
            if num_samples is not None and idx >= num_samples:
                break

            try:
                entry = json.loads(line.strip())
                image_path = os.path.join(image_root, entry['path'].lstrip('/'))
                question = entry['question']
                ground_truth = entry.get('ground truth', "")

                response = get_response(model, processor, image_path, question)

                thinking_part, thinking_length = extract_thinking(response)
                thinking_lengths.append(thinking_length)


                result = {
                    "path": entry['path'],
                    "question": question,
                    "model_answer": response,
                    "thinking": thinking_part,
                    "thinking_length": thinking_length,
                    "ground_truth": ground_truth
                }

                # Write to output jsonl
                outfile.write(json.dumps(result) + "\n")
                print(f"Processed {idx + 1}: {response[:50]}...")

                responses_data.append({
                    "image": os.path.basename(image_path),
                    "question": question,
                    "response": response,
                    "thinking": thinking_part,
                    "thinking_length": thinking_length,
                    "ground_truth": ground_truth
                })

            except Exception as e:
                print(f"Error processing line {idx + 1}: {str(e)}")
                continue


    os.makedirs("responses", exist_ok=True)
    model_name = args.model_id.replace('/', '_')
    with open(f"responses/ThinkLite-VL-7B.json", 'w') as f:
        json.dump(responses_data, f, indent=4)

    if thinking_lengths:
        plt.figure(figsize=(10, 6))
        plt.hist(thinking_lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Thinking Length (tokens)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Thinking Length in Vision Model Responses")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"responses/ThinkLite-VL-7B.png")

    print(f"Results saved to {output_jsonl}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/path/to/annotation.jsonl')
    parser.add_argument('--output', type=str, default='results/bingo_output.jsonl')
    parser.add_argument('--model_id', type=str, default='/path/to/your-model')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print("Loading model...")
    start = time.time()
    model, processor = load_model(args.model_id, args.device)
    print(f"Model loaded in {time.time() - start:.2f}s")

    print("Processing Bingo dataset...")
    process_jsonl(model, processor, args.input, args.output, args.num_samples, args.device)
    print("Done.")
