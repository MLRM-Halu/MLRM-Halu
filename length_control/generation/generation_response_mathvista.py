import os
import re
import json
import time
import torch
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

def timestamp() -> str:
    return time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))

def save_jsonl(data: list, path: str, mode='w', add_timestamp=False) -> None:
    file_name = f"{path.replace('.jsonl', '')}{timestamp()}.jsonl" if add_timestamp else path
    with open(file_name, mode, encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_model_and_processor(model_id, device="cuda:0", torch_dtype=torch.bfloat16, flash_attn=True):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if flash_attn else "eager",
    ).to(device).eval()
    return model, processor

def build_multimodal_prompt(image_path, question, prompt_template='default'):
    if prompt_template == 'default':
        prompt = (
            f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process MUST BE enclosed within <think> </think> tags. "
            "The final answer MUST BE in <answer> </answer> tags."
        )
    else:
        raise ValueError(f"Unknown prompt template: {prompt_template}")

    return [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt}
    ]}]

def generate_response(model, processor, messages, max_new_tokens=2048):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

def parse_response(text, processor, parse_thinking=True, parse_answer=True):
    thinking, answer = "", ""
    if parse_thinking:
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thinking = m.group(1).strip() if m else ""
    if parse_answer:
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        answer = m.group(1).strip() if m else ""
    think_len = len(processor.tokenizer(thinking, return_tensors="pt")['input_ids'][0]) if thinking else -1
    return thinking, answer, think_len

def run_benchmark(model, processor, data_path, image_root, save_path, prompt_template='default',
                  num_samples=None, resume=True, save_interval=10):
    data = load_jsonl(data_path)
    if num_samples:
        data = data[:num_samples]

    results = []
    if resume and os.path.exists(save_path):
        results = load_jsonl(save_path)
    start = len(results)
    print(f"Resuming at {start} / {len(data)}")

    temp_results = []
    for idx in tqdm(range(start, len(data))):
        entry = data[idx]
        image_file = os.path.join(image_root, entry['image'])
        question = entry.get('query', entry.get('question', ''))

        try:
            messages = build_multimodal_prompt(image_file, question, prompt_template)
            response = generate_response(model, processor, messages)
            thinking, answer, thinking_len = parse_response(response, processor)

            result = {
                "id": entry.get("pid", idx),
                "question": question,
                "ground_truth": entry.get("answer", ""),
                "question_type": entry.get("question_type", ""),
                "model_answer": response,
                "thinking": thinking,
                "thinking_length": thinking_len,
                "answer": answer,
                "image": entry['image']
            }
            temp_results.append(result)
            results.append(result)

            if (idx + 1) % save_interval == 0:
                save_jsonl(temp_results, save_path, mode='a')
                temp_results = []

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            traceback.print_exc()
            time.sleep(5)

    save_jsonl(results, save_path, mode='w')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/MathVista/output.json', help='Input JSONL path')
    parser.add_argument('--output', type=str, default='/data/responses/math_vista/ThinkLite-VL.json', help='Output JSONL path')
    parser.add_argument('--image_root', type=str, default='/data/MathVista/images/', help='Image root directory')
    parser.add_argument('--model_id', type=str, default='/data/model/ThinkLite-VL-7B/', help='Model path or hub name')
    parser.add_argument('--prompt_template', type=str, default='default')
    parser.add_argument('--num_samples', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load_model_and_processor(args.model_id, device=args.device)

    print("Running benchmark...")
    run_benchmark(
        model=model,
        processor=processor,
        data_path=args.input,
        image_root=args.image_root,
        save_path=args.output,
        prompt_template=args.prompt_template,
        num_samples=args.num_samples
    )
    print("Benchmark completed.")
