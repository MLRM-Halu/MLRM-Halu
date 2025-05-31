import gc
import os
import argparse
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)


def setup_device(device_id: str = "cuda:0") -> torch.device:
    if not torch.cuda.is_available() and "cuda" in device_id:
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_id)


def adjust_residual_hook(direction: torch.Tensor, layer_idx: int, weight: float):
    def hook_fn(module, input, output):
        direction_layer = direction[layer_idx].to(output[0].device)
        return (output[0] + weight * direction_layer,) + output[1:]
    return hook_fn


def load_model(model_id: str, direction_path: Optional[str] = None, direction_weight: float = 0.0) -> Tuple[Any, Any]:

    logger.info(f"Loading model: {model_id}")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    if direction_weight != 0.0 and direction_path:
        logger.info(f"Applying attention steering with weight {direction_weight}")
        try:
            direction = torch.load(direction_path, map_location=device).to(device)
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.register_forward_hook(adjust_residual_hook(direction, i, direction_weight))
        except Exception as e:
            logger.error(f"Failed to apply steering: {e}")
            raise

    return model, processor


def extract_thinking(response: str, processor: Any) -> Tuple[str, int]:

    import re
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        thinking_text = "".join(thinking_text.split())
        token_length = len(processor.tokenizer(thinking_text, return_tensors="pt")["input_ids"][0])
        return thinking_text, token_length
    return "", -1


def generate_response(model: Any, processor: Any, image_path: str, question: str) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (
                    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
                    "The reasoning process MUST BE enclosed within <think> </think> tags. "
                    "The final answer MUST BE in <answer> </answer> tags.\n" + question
                )},
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
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def load_dataset(file_path: str) -> List[Dict[str, Any]]:

    logger.info(f"Loading dataset from: {file_path}")
    try:
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def save_results(data: List[Dict[str, Any]], file_path: str, mode: str = "w", add_timestamp: bool = False) -> None:
    if add_timestamp:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(file_path)
        file_path = f"{base}_{timestamp}{ext}"

    try:
        with open(file_path, mode, encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(data)} items to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {file_path}: {e}")
        raise


def process_benchmark(
    model: Any,
    processor: Any,
    dataset_path: str,
    output_path: str,
    image_root: str,
    num_samples: Optional[int] = None,
    checkpoint_interval: int = 10
) -> None:

    dataset = load_dataset(dataset_path)
    if num_samples:
        dataset = dataset[:num_samples]

    try:
        existing_results = load_dataset(output_path)
    except FileNotFoundError:
        existing_results = []

    start_idx = len(existing_results)
    if start_idx >= len(dataset):
        logger.info(f"Already processed all {len(dataset)} samples")
        return

    logger.info(f"Processing samples {start_idx} to {len(dataset)}")
    results = existing_results[:]
    batch_results = []

    retry_count = 0
    max_retries = 5
    last_start_idx = start_idx

    while start_idx < len(dataset):
        if start_idx == last_start_idx:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) reached at index {start_idx}")
                break
        else:
            retry_count = 0
            last_start_idx = start_idx

        try:
            for idx in tqdm(range(start_idx, len(dataset)), desc="Processing"):
                entry = dataset[idx]
                image_path = os.path.join(image_root, os.path.basename(entry.get("image", entry.get("image_src", ""))))
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue

                question = entry["question"]
                ground_truth = entry.get("ground_truth", entry.get("answer", ""))
                question_type = entry.get("question_type", "")

                response = generate_response(model, processor, image_path, question)
                thinking_part, thinking_length = extract_thinking(response, processor)

                result = {
                    "id": entry.get("id", idx),
                    "question": question,
                    "model_answer": response,
                    "thinking": thinking_part,
                    "thinking_length": thinking_length,
                    "question_type": question_type,
                    "ground_truth": ground_truth,
                    "image_path": image_path
                }

                batch_results.append(result)
                results.append(result)

                if (idx + 1) % checkpoint_interval == 0 or idx + 1 == len(dataset):
                    save_results(batch_results, output_path, mode="a")
                    batch_results = []

            save_results(results, output_path, mode="w")
            break

        except Exception as e:
            logger.error(f"Error at index {idx}: {e}", exc_info=True)
            save_results(results, output_path, mode="w")
            if batch_results:
                save_results(batch_results, output_path, mode="a")
            time.sleep(5)
            start_idx = len(results)


def main():
    """Main function to run the benchmark processing."""
    parser = argparse.ArgumentParser(description="Process vision-language model benchmarks with attention-based steering.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSON/JSONL dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save output JSONL results")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier or path")
    parser.add_argument("--image_root", type=str, default="", help="Root directory for images")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--direction_path", type=str, default=None, help="Path to steering direction tensor")
    parser.add_argument("--direction_weight", type=float, default=0.0, help="Weight for attention steering")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for computation (e.g., cuda:0, cpu)")

    args = parser.parse_args()
    global device
    device = setup_device(args.device)

    model, processor = load_model(
        args.model_id,
        direction_path=args.direction_path,
        direction_weight=args.direction_weight
    )

    process_benchmark(
        model=model,
        processor=processor,
        dataset_path=args.dataset,
        output_path=args.output,
        image_root=args.image_root,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
