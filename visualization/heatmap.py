import math
import torch
import argparse
import os
import re
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import numpy as np
import seaborn as sns

model_path = '/data/ThinkLite-VL-7B/'

def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding_side='left', use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0"
    ).to("cuda:0").eval()
    return model, processor

def calculate_plt_size(attention_layer_num):
    cols = math.ceil(math.sqrt(attention_layer_num))
    rows = math.ceil(attention_layer_num / cols)
    return rows, cols

def get_response(model, processor, image_path, question, final_answer_tokens=1024):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags, Let's think more."}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    input_ids = inputs['input_ids'][0].tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = input_ids.index(vision_start_token_id) + 1
    pos_end = input_ids.index(vision_end_token_id)
    image_indices = list(range(pos, pos_end))

    image_inputs_aux = processor.image_processor(images=image_inputs)
    output_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:] / 2
    output_shape = output_shape.astype(int)

    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=final_answer_tokens,
            do_sample=True,
            output_attentions=True,
            return_dict_in_generate=True
        )
        gen_ids = gen_output.sequences
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()
        all_attentions = gen_output.attentions

    num_layers = len(all_attentions[0])
    num_generated_tokens = len(all_attentions)
    rows, cols = calculate_plt_size(num_layers)
    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))

    for i, ax in enumerate(axes.flatten()):
        if i < num_layers:
            layer_att = []
            for t in range(num_generated_tokens):
                att_t = all_attentions[t][i][0, :, -1, pos:pos_end].mean(dim=0)
                layer_att.append(att_t)
            att = torch.stack(layer_att).mean(dim=0)
            att = att.to(torch.float32).detach().cpu().numpy()
            att_reshaped = att.reshape(output_shape)
            ax.imshow(att_reshaped, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Layer {i+1}", fontsize=10, pad=5)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated Answer:")
    print(output_text)
    return output_text

def process_single_sample(model, processor, image_path, question):
    answer = get_response(model, processor, image_path, question)
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--model_id', type=str, default='/data/ThinkLite-VL-7B/')
    args = parser.parse_args()

    model, processor = load_model(args.model_id)
    process_single_sample(model, processor, args.image_path, args.question)
