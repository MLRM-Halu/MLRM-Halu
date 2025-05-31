import os
import json
import re
import time
import argparse
import torch
import logging
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda").eval()
    return model, processor


def clean_think_output(output_text):
    match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    think = match.group(1).strip() if match else output_text.strip()
    return re.split(r"\*\*Answer:\*\*", think, maxsplit=1)[0].strip()


def get_response(model, processor, image_path, question,
                 max_tokens_thinking=5000, final_answer_tokens=1024, max_think_length=1024):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",
             "text": f"{question}You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."}
        ]}
    ]
    prompt_text = processor.apply_chat_templatemessages, tokenize = False, add_generation_prompt = True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
    text = [prompt_text], images = image_inputs, videos = video_inputs,
    padding = True, return_tensors = "pt"

).to(model.device)

with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=max_tokens_thinking, do_sample=True)
trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
output = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

think = clean_think_output(output)
think = think[:max_think_length] if len(think) > max_think_length else think
final_think_prompt = f"<think>\n{think}\n</think>\n\n"

final_messages = [
{"role": "user", "content": [
    {"type": "image", "image": image_path},
    {"type": "text", "text": f"{question}\n{final_think_prompt} So the Final Answer is:"}
]}
]
final_text = processor.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
final_inputs = processor(
text = [final_text], images = image_inputs, videos = video_inputs,
padding = True, return_tensors = "pt"
).to(model.device)

with torch.no_grad():
    final_ids = model.generate(**final_inputs, max_new_tokens=final_answer_tokens, do_sample=True)
final_trimmed = [out[len(inp):] for inp, out in zip(final_inputs.input_ids, final_ids)]
final_output = processor.batch_decode(final_trimmed, skip_special_tokens=True)[0].strip()

return final_output


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_json(model, processor, input_json, output_json, image_root, max_think_length, resume=True):
    data = load_dataset(input_json)

    if resume and os.path.exists(output_json):
        processed = load_dataset(output_json)
        start_idx = len(processed)
        logger.info(f"Resuming from index {start_idx}")
    else:
        processed = []
        start_idx = 0

    for idx in tqdm(range(start_idx, len(data)), desc="Processing"):
        entry = data[idx]
        image_name = os.path.basename(entry["image_src"])
        image_path = os.path.join(image_root, image_name)
        question = entry["question"]

        try:
            answer = get_response(model, processor, image_path, question, max_think_length=max_think_length)
            entry["model_answer"] = answer
            processed.append(entry)

            save_dataset(processed, output_json)
            logger.info(f"[{idx + 1}] Saved result: {answer[:50]}...")

        except Exception as e:
            logger.error(f"Error at index {idx}: {e}")
            continue

    logger.info(f"Finished processing {len(processed)} samples. Results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage VLM reasoning and answering script")
    parser.add_argument("--input", type=str, default="input.json", help="Input JSON file path (default: input.json)")
    parser.add_argument("--output", type=str, default="output.json",
                        help="Output JSON file path (default: output.json)")
    parser.add_argument("--model_id", type=str, default="/model/R1-OneVision/", help="Path to model")
    parser.add_argument("--image_root", type=str, default="/data/MMhalu/images/", help="Directory with images")
    parser.add_argument("--max_think_length", type=int, default=100,
                        help="Maximum length of thinking process (default: 1024)")
    args = parser.parse_args()

    model, processor = load_model(args.model_id)
    process_json(model, processor, args.input, args.output, args.image_root, args.max_think_length)