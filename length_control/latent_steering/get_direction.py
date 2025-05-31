import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Reproducibility
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/model/Ocean_R1_7B_Instruct/")
parser.add_argument("--json_path", type=str, default="response/Ocean_R1_7B_Instruct.jsonl")
parser.add_argument("--output_path", type=str, default="directions/thinking_length_direction.pt")
parser.add_argument("--mode", type=str, choices=["text", "vision"], default="text",
                    help="Choose 'text' for text-only or 'vision' for image+text processing.")
args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()


residual_outputs = []

def capture_residual_hook():
    def hook_fn(module, input, output):
        residual_outputs.append(input[0].detach())
    return hook_fn

for layer in model.model.layers:
    layer.post_attention_layernorm.register_forward_hook(capture_residual_hook())

# Load JSON/JSONL
def load_json_or_jsonl(file_path):
    if file_path.endswith(".jsonl"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

responses_data = load_json_or_jsonl(args.json_path)
valid_responses = [ex for ex in responses_data if ex.get('thinking_length', -1) != -1]
# Select different sizes based on various datasets, which can be determined by the distribution of think tokens obtained through generation.
long_examples = [ex for ex in valid_responses if ex['thinking_length'] > 350]
short_examples = [ex for ex in valid_responses if ex['thinking_length'] < 150]

print(f"Number of long-thinking examples: {len(long_examples)}")
print(f"Number of short-thinking examples: {len(short_examples)}")

# === TEXT-ONLY MODE ===
def extract_text_embedding(example):
    residual_outputs.clear()

    messages = [
        {"role": "user", "content": [{"type": "text", "text": example['question']}]},
        {"role": "assistant", "content": [{"type": "text", "text": example['thinking']}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    question_toks = processor(text=[example['question']], return_tensors="pt").input_ids
    start = question_toks.shape[1]
    full_toks = processor(text=[example['question'] + example['thinking']], return_tensors="pt").input_ids
    end = full_toks.shape[1]

    with torch.no_grad():
        _ = model(**inputs)

    embedding = torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu()
    residual_outputs.clear()
    return embedding

# === VISION-LANGUAGE MODE ===
def extract_vision_embedding(example):
    residual_outputs.clear()

    image = Image.open(example['image_path']).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": example['image_path']}},
                {"type": "text", "text": example['question']}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['thinking']}]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

    input_ids = inputs['input_ids'][0].tolist()
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    try:
        vision_start = input_ids.index(vision_start_id)
        vision_end = input_ids.index(vision_end_id)
    except ValueError:
        print(f"[Warning] Vision tokens missing in: {example['image_path']}")
        return None

    vision_token_len = vision_end - vision_start + 1
    question_len = processor(text=[example['question']], return_tensors="pt").input_ids.shape[1]
    start = vision_token_len + question_len
    full_len = processor(text=[example['question'] + example['thinking']], return_tensors="pt").input_ids.shape[1]
    end = vision_token_len + full_len

    with torch.no_grad():
        _ = model(**inputs)

    if end > start:
        emb = torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu()
        residual_outputs.clear()
        return emb
    else:
        residual_outputs.clear()
        return None

if args.mode == "text":
    extract_fn = extract_text_embedding
elif args.mode == "vision":
    extract_fn = extract_vision_embedding
else:
    raise ValueError("Unsupported mode. Use 'text' or 'vision'.")

long_embeddings, short_embeddings = [], []

print(f"Extracting long-thinking embeddings using mode: {args.mode} ...")
for ex in tqdm(long_examples):
    emb = extract_fn(ex)
    if emb is not None:
        long_embeddings.append(emb)

print(f"Extracting short-thinking embeddings using mode: {args.mode} ...")
for ex in tqdm(short_examples):
    emb = extract_fn(ex)
    if emb is not None:
        short_embeddings.append(emb)

mean_long = torch.stack(long_embeddings, dim=0).mean(dim=0)
mean_short = torch.stack(short_embeddings, dim=0).mean(dim=0)
direction = mean_long - mean_short

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
torch.save(direction, args.output_path)
print(f"Thinking length direction vector saved to: {args.output_path}")
