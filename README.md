# More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models


<a href='https://arxiv.org/abs/2410.06172'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://mssbench.github.io/ '><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://huggingface.co/datasets/kzhou35/mssbench/tree/main'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>


![Teaser figure](figures/intro.png)


## News
- \[**May 2025**\]  [Paper](https://arxiv.org/abs/2505.21523) is now available. 📢


## TO DO 
- [√] Release visualization tools and reasoning length control  strategies.
- [√] Release small-scale RH-Bench benchmark.
-  [ ]  Expand and refine RH-Bench to support more multimodal reasoning model. Coming soon!



## ❗Abstract
Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, we observe that this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more on language priors. Attention analysis reveals that longer reasoning chains reduce focus on visual inputs, contributing to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model’s perception accuracy changes with reasoning length, enabling evaluation of whether the model preserves visual grounding while reasoning. We also release RH-Bench, a diagnostic benchmark covering diverse multimodal tasks, designed to jointly assess the balance of reasoning ability and hallucination. We find that (i) larger models generally exhibit a better balance between reasoning and perception; and (ii) this balance depends more on the types and domains of the training data than its volume. Our findings highlight the need for evaluation frameworks that account for both reasoning quality and perceptual reliability.



## 🎯 Visualization 

```
>> conda create -n myenv python=3.9
>> conda activate myenv
>> pip install -r requirements.txt
```


## 🕹️ Reasoning Length Contorl

*Step 1** 1111
```

```
*Step 2**    1111

```

```


## 🧐 Evaluation 


```

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



