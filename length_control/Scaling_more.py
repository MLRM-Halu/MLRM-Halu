import os
import re
import torch
import argparse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0"
    ).to("cuda").eval()
    return model, processor

def clean_think_output(output_text):
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
    else:
        think_content = output_text.strip()
    think_content = re.split(r"\*\*Answer:\*\*", think_content, maxsplit=1)[0].strip()
    return think_content

def get_response(model, processor, image_path, question,
                 max_tokens_thinking=5000,
                 enforce_num=2,
                 ignore_str="Wait, but I'm not sure if I'm missing anything.",
                 final_answer_tokens=1024):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_tokens_thinking, do_sample=True)
    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    output_full = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    think_content = "<think>\n" + clean_think_output(output_full) + "\n"
    remaining_tokens = max_tokens_thinking - len(trimmed_ids[0])

    for _ in range(enforce_num):
        if remaining_tokens <= 0:
            break
        think_content += f"\n{ignore_str}\n"

        think_messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"{question}\n\n{think_content}"}
        ]}]

        think_text = processor.apply_chat_template(think_messages, tokenize=False, add_generation_prompt=True)
        think_inputs = processor(text=[think_text], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            think_gen_ids = model.generate(**think_inputs, max_new_tokens=remaining_tokens, do_sample=True)
        think_trimmed_ids = [out[len(inp):] for inp, out in zip(think_inputs.input_ids, think_gen_ids)]
        additional_think = processor.batch_decode(think_trimmed_ids, skip_special_tokens=True)[0].strip()
        think_content += "\n" + clean_think_output(additional_think)
        remaining_tokens -= len(think_trimmed_ids[0])

    think_content += "\n</think>\n\n"
    final_prompt = f"{question}\n{think_content}"

    final_messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"{final_prompt} Hint: Only Provide the Answer."}
    ]}]

    final_text = processor.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
    final_inputs = processor(text=[final_text], images=image_inputs, videos=video_inputs,
                             padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        final_gen_ids = model.generate(**final_inputs, max_new_tokens=final_answer_tokens, do_sample=True)
    final_trimmed_ids = [out[len(inp):] for inp, out in zip(final_inputs.input_ids, final_gen_ids)]
    final_answer = processor.batch_decode(final_trimmed_ids, skip_special_tokens=True)[0].strip()

    print(final_answer)
    torch.cuda.empty_cache()
    return final_answer

def process_single_sample(model, processor, image_path, question, enforce_num):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return get_response(model, processor, image_path, question, enforce_num=enforce_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/data/MMVP/images/')
    parser.add_argument('--question', type=str, default='/data/question.json')
    parser.add_argument('--model_id', type=str, default='/data/R1-Onevision-7B/')
    parser.add_argument('--enforce_num', type=int, default=2)
    args = parser.parse_args()

    model, processor = load_model(args.model_id)
    process_single_sample(model, processor, args.image_path, args.question, args.enforce_num)