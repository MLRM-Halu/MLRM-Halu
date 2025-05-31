
<p align="center">
  <img src="figures/logo1.png" alt="Project Logo" width="230"/>
</p>

# More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models


<a href='https://arxiv.org/abs/2505.21523'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://mlrm-halu.github.io/ '><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/LCZZZZ/RH-Bench/tree/main'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>


![Teaser figure](figures/intro.png)


## News
- \[**May 2025**\]  [Paper](https://arxiv.org/abs/2505.21523) is now available. 📢


## TO DO 
- [X] Release visualization tools and reasoning length control  strategies.
- [X] Release small-scale RH-Bench benchmark.
- [ ]  Expand and refine RH-Bench to support more multimodal reasoning model. Coming soon!



## ❗Abstract
Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, we observe that this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more on language priors. Attention analysis reveals that longer reasoning chains reduce focus on visual inputs, contributing to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model’s perception accuracy changes with reasoning length, enabling evaluation of whether the model preserves visual grounding while reasoning. We also release RH-Bench, a diagnostic benchmark covering diverse multimodal tasks, designed to jointly assess the balance of reasoning ability and hallucination. We find that (i) larger models generally exhibit a better balance between reasoning and perception; and (ii) this balance depends more on the types and domains of the training data than its volume. Our findings highlight the need for evaluation frameworks that account for both reasoning quality and perceptual reliability.




## 🎯 Visualization 

```
python heatmap.py \
--image_path /data/image.jpg  \
--question "Describe this image in detail."
```

```
python layer_analysis.py \
  --model-path "R1-OneVision/" \
  --image-folder "images/" \
  --question-file "question.jsonl" \
  --answers-file "./results.pt" \
  --plot-path "./attention_distribution.png"
```
  
![Teaser figure](figures/heatmap.png)

## 🕹️ Reasoning Length Contorl

### Budget Forcing & Test Time Scaling
 Refer to budget_forcing.py and Scaling_more.py in the length_control directory.

### Latent State Steering

**Step 1**   Collect responses from multimodal reasoning models on various tasks to extract hidden states of internal attention later.
```
python generate_response_your_data.py \
  --input "/your/dataset/path/annotation.jsonl" \
  --output "/your/output/path/results.jsonl" \
  --model_id "/your/model/path/ModelDirectory/" \
  --num_samples 100 \
  --device "cuda:2"
```
**Step 2**    Extract per-layer directional vectors from the residual inputs of the self-attention mechanism in multimodal reasoning models.  It supports two modes: text mode, which processes only the question and thinking tokens, and vision mode, which processes the image along with the question and thinking tokens.
```
python get_direction.py \
  --model /path/to/your/model/ \
  --json_path /path/to/your/response.jsonl \
  --output_path /path/to/save/steering_direction.pt \
  --mode text
```

**Step 3**    Control the reasoning length of the multimodal model and obtain responses under each steering state. The current range is [-0.1,0.1], which can be adjusted according to different datasets and tasks. Note: Extremely large or small parameter values may degrade the model's performance. Support both automated sweeping of parameters via range input and manual specification of individual values.
```
python steering_mlrm.py \
  --dataset /path/to/dataset.jsonl \
  --output results/output.jsonl \
  --model_id /path/to/model \
  --image_root /path/to/images \
  --direction_path /path/to/direction_vector.pt \
  --direction_weights_range -0.1 0.1 0.01 \ #  (start, end, and step )
  --num_samples 100 \
  --device cuda:0
```

![Teaser figure](figures/length.png)


## 🧐 Evaluation 

### Model
| Model                          | Link                              |
|--------------------------------|-------------------------------------------|
|R1-Onevision          | 🤗 [R1-Onevision](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B-RL)|
|ThinkLite-VL          | 🤗 [ThinkLite-VL ](https://huggingface.co/russwang/ThinkLite-VL-7B)     |
|MM-Eureka-Qwen        | 🤗 [MM-Eureka-Qwen ](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B)   |
|Vision-R1        | 🤗 [Vision-R1](https://huggingface.co/JefferyZhan/Qwen2.5-VL-7B-Instruct-Vision-R1)   |
|Ocean-R1        | 🤗 [Ocean-R1 ](https://huggingface.co/minglingfeng/Ocean_R1_7B_Instruct)   |
|MM-R1       | 🤗 [MM-R1 ](https://huggingface.co/MMR1/MMR1-Math-v0-7B)   |
|Curr-ReFT       | 🤗 [MM-R1 ](https://huggingface.co/ZTE-AIM/3B-Curr-ReFT)   |
|LLM- R1      | 🤗 [LLM-R1 ](https://huggingface.co/VLM-Reasoner/LMM-R1-MGT-PerceReason)   |
|Skywork-R1V      | 🤗 [Skywork-R1V](https://huggingface.co/Skywork/Skywork-R1V-38B)   |

```
# Reason
python evaluation_rhbench_reason.py \
--input_dir "/data/steering_reason/" \
--output_dir "/data/steering_reason/score" \
--summary_file "/data/steering_reason/evaluation_summary.txt"

# Hallucination
python evaluation_rhbench_perception.py \
--input_dir "/data/steering_hallu/" \
--output_dir "/data/steering_hallu/score" \
--summary_file "/data/steering_hallu/evaluation_summary.txt"
```


### Citation
If you find the code is valuable, please use this citation.
```
 @misc{liu2025thinkingseeingassessingamplified,
      title={More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models}, 
      author={Chengzhi Liu and Zhongxing Xu and Qingyue Wei and Juncheng Wu and James Zou and Xin Eric Wang and Yuyin Zhou and Sheng Liu},
      year={2025},
      eprint={2505.21523},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21523}, 
```



