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
    ).to("cuda:1").eval()
    return model, processor


def extract_thinking(response):
    """Extracts the thinking part from response text, including the <think> tags."""
    match = re.search(r"(<think>.*?</think>)", response, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        return thinking_text, len(processor.tokenizer(thinking_text, return_tensors='pt')['input_ids'][0])
    return "", -1



def get_response(model, processor, image_path, question):
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

    # Prepare inputs
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
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def process_json(model, processor, input_json, output_json, num_samples=96):
    json_data = json.load(open(input_json, 'r'))
    json_data = json_data[:num_samples]
    total_samples = len(json_data)

    if not os.path.exists(output_json):
        with open(output_json, 'w') as f:
            json.dump([], f)

    thinking_lengths = []
    responses_data = []

    # 遍历并处理每个样本
    for idx, line in enumerate(json_data):
        image_path = line['image_src']
        image_path = os.path.basename(image_path)
        image_path = os.path.join('/data/MMhalu/images/', image_path)

        question = line['question']

        response = get_response(model, processor, image_path, question)
        thinking_part, thinking_length = extract_thinking(response)
        thinking_lengths.append(thinking_length)

        line['model_answer'] = response
        line['thinking'] = thinking_part
        line['thinking_length'] = thinking_length

        print(f"Processed sample {idx + 1}/{total_samples}: {response[:50]}...")

        with open(output_json, 'r') as f:
            current_data = json.load(f)

        current_data.append(line)

        with open(output_json, 'w') as f:
            json.dump(current_data, f, indent=2)

        responses_data.append({
            "image": os.path.basename(image_path),
            "question": question,
            "response": response,
            "thinking": thinking_part,
            "thinking_length": thinking_length
        })

    os.makedirs("responses", exist_ok=True)
    with open(f"responses/MM-Eureka-Qwen-7B_mmhalu.json", 'w') as f:
        json.dump(responses_data, f, indent=4)

    plt.figure(figsize=(10, 6))
    plt.hist(thinking_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Thinking Length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Thinking Length in Vision Model Responses")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"responses/thinking_length_distribution_MM-Eureka-Qwen-7B_mmhalu.png")

    print(f"All results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/data/response_template.json',
                        help='Template file containing images and questions')
    parser.add_argument('--output', type=str,
                        default='responses/MM-Eureka-Qwen-7B_mmhalu.json',
                        help='Output file to store model responses')
    parser.add_argument('--model_id', type=str, default="/model/MM-Eureka-Qwen-7B",
                        help='Path to the model')
    parser.add_argument('--num_samples', type=int, default=96,
                        help='Number of samples to process (default: 50)')
    args = parser.parse_args()

    model, processor = load_model(args.model_id)

    process_json(model, processor, args.input, args.output, args.num_samples)

