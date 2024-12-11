# Announcing Bamba

<p align="center">
  <img src="https://github.com/foundation-model-stack/bamba/blob/main/bamba.jpeg" alt="Bamba" width="400" height="400">
</p>

Transformer models are increasingly being adopted in production settings, where memory bandwidth bottlenecks have become a significant challenge. These bottlenecks arise during the per-token decoding step, which is computationally lightweight but constrained by memory and compute interactions. For shorter sequences, the issue stems from transferring weights between memory and compute, while for longer sequences, it is dominated by moving KV-cache data. 

As the demand for longer-sequence models grows, such as [Meta’s Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B) (128K tokens), [IBM’s Granite Code v2](https://huggingface.co/ibm-granite/granite-3b-code-base-128k) (128K tokens), and [Mistral Large](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411) (32K tokens)—the KV-cache bottleneck becomes more pronounced. This challenge is driven by the linear growth of KV-cache with sequence length due to full-attention layers. 

While techniques like lower precision, layer pruning, and compression can reduce KV-cache size, they do not address the root cause. To tackle this, a new class of architectures that maintain a constant KV-cache size has emerged, including Mamba2, [DeltaNet](https://arxiv.org/abs/2406.06484), and [Griffin](https://arxiv.org/abs/2402.19427). Among these, the Mamba architecture has received widespread community attention, with several implementations demonstrating its potential over the past year, such as NVIDIA Mamba2, Codestral Mamba, Jamba, and Samba.

We introduce Bamba, a 9B Mamba2 hybrid model that adds to the proof point of these emerging architectures and closes the gap further with SoTA transformer models. Inspired by AllenAI, in this collaboration between IBM, Princeton, and UIUC, we provide the entire lineage of data for training, multiple checkpoints, and the code for pretraining. We also enable this model in key OSS communities - Hugging Face `transformers`, `TRL`, and `vLLM` to allow developers to use the model from the get go. We will share the details of our learnings when training this model and welcome the community to help further close the gap with SoTA open source models and bring Mamba architecture to mainstream models and alleviate the KV-cache bottleneck.

## Evaluations

We break our evaluations into three parts:
1. Comparison with Mamba architecture based language models
2. Comparison with transformers trained to similar tokens
3. Comparison with SoTA transformer models of similar size

We use a local copy of the [Open LLM leaderboard v1 and v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) for our benchmarks, whereas for other models, we use the output from the Live leaderboard (we could not run this for NVIDIA Hybird Mamba model as the model weights are not in Hugging Face transformers compatible format, hence we report the numbers from the paper). For the v2 leaderboard results, we perform [normalization](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization) and report the normalized results.

TL;DR
We find that Bamba9B adds another proof point to similar models such as NVIDIA Mamba2, Zamba, and Falcon Mamba, while providing the entire data lineage. This will allow the community to surgically improve the model further. We also compare to similar sized transformer models and observe that our model outperforms since we use newer training techniques and better quality data. Compared to SoTA transformer models, Bamba 9B rivals Meta Llama3.1 8B based on averages of OpenLLM v1 and v2 leaderboards. However, we acknowledge that there are gaps in certain key tasks such as MMLU and GSM8K. We have done experiments that indicate that this gap is due to lack of quality training data targeting these tasks. We plan to continue training the current versions with newer datasets like [Olmo2 mix](https://huggingface.co/datasets/allenai/olmo-mix-1124) and SFT datasets such as [Tuluv3](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture), [agent instruct](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1), and [Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater).

<p align="center">
<img src="https://github.com/user-attachments/assets/71fe4c98-612d-476b-bff1-7dff7f8df98d" alt="Bamba-perf-avg" width="618" height="332">
</p>


### Comparison with Hybrid Architectures
Several Mamba based architecture models have started coming up in the last 6months (e.g., NVIDIA Hybrid Mamba2, Codestral Mamba, Falcon Mamba, Zamba7Bv1) furthering the performance of these architectures and demonstrating their inference performance as well as closing the gap with quality. We compare 8 key benchmarks across Bamba, NVIDIA Hybrid model, Zamba, and Falcon Mamba. Falcon Mamba is a pure Mamba model, Zamba has shared attention layer for every 6 Mamba layers, and Bamba and NVIDIA are both Hybrid models with full attention layers interspersed with Mamba layer. While Falcon Mamba performs the best overall and has been trained to 5.5T tokens, there are open questions on how well copying tasks work on such pure Mamba models. Zamba was trained on fewer tokens (1T), but with a different Hybrid architecture. Bamba and NVIDIA Mamba are quite similar to each other (details on differences are summarized in the model architecture section), but Bamba is trained to 2.2T and NVIDIA Hybrid Mamba is trained to 3.5T tokens. The key point is that even with all these architectural variations and different number of tokens, Mamba based models are demonstrating competitive results. We are continuing to train the Bamba model with latest datasets and plan to release future checkpoints as the model gets better.
  
| Benchmark score   | Bamba 9B   | NVIDIA Mamba2 Hybrid 8B | Zamba 7B   | Falcon Mamba 7B   |
|-------------------|------------|-------------------------|------------|-------------------|
| MMLU (5-shot)     | 60.77      | 53.6                   | 57.85      | **63.19**         |
| Hellaswag         | 81.8       | 77.69                  | **82.27**  | 80.82             |
| Winogrande        | 76.87      | 71.27                  | **79.32**  | 78.14             |
| Piqa              | 82.26      | 79.65                  | 82.21      | **83.62**         |
| OpenbookQA        | 47.6       | 42.8                   | 46.8       | **47.8**          |
| ARC-C             | 63.23      | 47.7                   | 55.38      | **63.4**          |
| TruthfulQA        | 49.21      | 38.72                  | 49.69      | **53.46**         |
| **Average**       | 65.96      | 58.78                  | 64.79      | **67.2**          |

## Comparison with transformers with similar token budget
We pick a few promiment models: Olmo 7B trained on identical data (2024), Meta Llama2 7B (2023), and IBM Granite 7B (2023), which have been trained to 2T tokens. While Olmo 7B outperforms Meta Llama2 and IBM Granite models across these 8 benchmarks, we note that with the same dataset, Bamba outperforms Olmo 7B. The main takeaway is that the Bamba model does well on the same dataset and similar token budget transformer models.

| Benchmark score   | Bamba 9B   | Olmo1.5 7B   | Meta Llama2 7B   | IBM Granite 7B   |
|-------------------|------------|--------------|------------------|------------------|
| MMLU (5-shot)     | **60.77**  | 53.39        | 46.87            | 49.02            |
| Hellaswag         | **81.8**   | 78.65        | 78.59            | 77.0             |
| Winogrande        | **76.87**  | 72.77        | 74.03            | 70.17            |
| Piqa              | **82.26**  | 78.4         | 79.0             | 80.14            |
| OpenbookQA        | 47.6       | **50.2**     | 44.0             | 40.8             |
| ARC-C             | **63.23**  | 48.5         | 53.07            | 49.91            |
| TruthfulQA        | **49.21**  | 36.0         | 38.76            | 38.7             |
| **Average**       | **65.96**  | 59.7         | 59.19            | 57.96            |

### Comparison with SoTA transformer models

Finally, we compare with SoTA transformer models (Meta Llama 3.1 8B, IBM Granite v3 8B, and Olmo2 7B). For the missing results in HF leaderboard, we rerun these (marked with a *). We observe that there are obvious benchmark gaps, but note that the Bamba and Falcon Mamba models are closing these benchmark gaps. We believe that with the right data (note that the latest transformer models have been trained on data from the last six months, before we trained our model), these gaps can be closed. For example, we had one small scale run that added `metamath` and improved our `GSM8k` score from `36.77` to `60.0`. We are continuing to train this model with latest datasets such as the one released by the AllenAI team - [Olmo2 mix](https://huggingface.co/datasets/allenai/olmo-mix-1124), [Dolmino mix](https://huggingface.co/datasets/allenai/dolmino-mix-1124), and [TuluV3](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture).


| Benchmark score | Bamba 9B | Falcon Mamba 7B | Meta Llama 3.1 8B | IBM Granite v3 8B | Olmo2 7B* |
|-----------------|----------|------------------|-------------------|-------------------|-----------|
| MMLU           | 60.77    | 63.19           | **66.70**         | 65.54             | 63.96     |
| MMLU PRO       | 17.53    | 14.33           | 25.46             | **25.83**         | 22.79     |
| BBH            | 17.40    | 19.88           | 25.16             | **28.02**         | 21.69     |
| Hellaswag      | 81.80    | 80.82           | 81.98             | **83.61**         | 81.93     |
| Winogrande     | 76.87    | 78.14           | 77.51             | **80.90**         | 77.03     |
| Piqa           | 82.26    | **83.62**       | 82.54             | 82.32             | 81.39     |
| OpenbookQA     | 47.60    | 47.80           | 46.80             | 46.80             | **49.20** |
| ARC-C          | 63.23    | 63.40           | 57.85             | 63.40             | **64.51** |
| TruthfulQA     | 49.21    | **53.46**       | 45.16             | 52.89             | 43.32     |
| MuSR           | 9.59     | 9.88            | 8.72              | 9.32              | **10.02** |
| GSM8K          | 36.77    | 52.08           | 49.96             | 62.55             | **68.01** |
| GPQA           | 4.14     | 8.17            | 8.61              | **9.06**          | 4.92      |
| **Average**    | 47.92    | 50.09           | 48.61             | **56.77**         | 54.89     |


We invite the community to help improve the model further and identify any fundamental limitations in this inference efficient model.

## Inference efficiency
The **KV-cache bottleneck** is the biggest challenge for Large language models and the community has furiously pursued addressing this issue through different approaches - quantization, pruning for standard transformers and of course novel model architectures such as Mamba2, Linear transformers, and Retnets. Realizing inference gains at production time is non-trivial even for changes to standard transformers - typically needing custom kernels that scale with the problem size. One key reason to pursue Mamba2 architecture is to build on the momentum of availability of kernels in the community. This effort furthers the progress by fixing issues in the core Mamba2 kernels via the popular vLLM model serving framework.

Our current progression of integration into vLLM can be tracked via [this PR](https://github.com/vllm-project/vllm/pull/10909). We use this PR to benchmark the inference latencies against a typical transformer architecture; we pick Meta Llama 3.1 8B because of its popularity and that it will be highly optimized. We use an NVIDIA H100 80GB GPU for obtaining our measurements and use our measurement framework to obtain throughput in tokens/second. We pick input size as 1K tokens and generate varying outputs (from 2K to 64K) and at varying batch sizes. Even though the model is 20% bigger than Llama 3.1 8B, as batch size and sequence lengths increase, Bamba outperforms a similar sized transformer model by a margin of 2x. 

<p align="center">
<img src="https://github.com/user-attachments/assets/ed9901d5-7721-4158-8fc7-3106ae50097f" alt="Bamba-ratios" width="600" height="300">
</p>

Our analysis indicates that on a H100 NVIDIA GPU, we expect 4-5x speedup when inference shifts to a memory bottleneck (which typically happens in production settings). However, we have not realized that in vLLM because of two primary reasons: (i) chunked pre-fill is not supported for Bamba and any Mamba2 based architectures, (ii) memory allocation is assuming standard transformer KV-cache, and (iii) kernels are not optimized for inference. These issues are being tracked [here](https://github.com/foundation-model-stack/bamba/issues/3).

## Model architecture
We base our model architecture on the NVIDIA Hybrid Mamba2 with the following changes.
| Parameter | Bamba 9B | NVIDIA Hybrid Mamba2 8B |
|---------|-------|-------|
| Num layers | 32 | 29 |
| Num attention layers | 3 | 4 |
| Num Mamba2 layers | 29 | 25 |
| MLP expansion factor | 3.5 | 4 |
| Vocab size | 128k | 256k |
| Non-embedding parameters | 8.8B | 8.6B |
| RoPE | yes | no |
| Gated linear units | yes | no |

We have a total of 8B parameters in the Mamba2 layers, 800M in full attention layers, and 1B in embeddings. The hidden state is 4K, GQA for full attention with 8 KV-heads and 32 heads, Mamba2 layer head dimension is 64, and convolution filter size is 4.

## Data
Open source data has come a long way since the inception of Pile. When we started training this model, the best open source data was Dolma v1.7 which was shown to be quite performant through [Olmo models](https://allenai.org/olmo) and [ablations](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) by the Hugging Face data team. Since then, several other higher quality open source datasets have been released such as [DCLM](https://www.datacomp.ai/dclm/), [FineWeb-2](https://huggingface.co/collections/HuggingFaceFW/fineweb2-6755657a481dae41e8fbba4d), and Olmo mix.

We use Dolma v1.7 for the first phase of training and the data mixes chosen are illustrated below. For the second phase of training, we use [Fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) datasets. These datasets are downloaded in their raw form and we tokenize them using the [Ray framework](https://github.com/ray-project/ray) running on an internal large scale [Red Hat Open Shift](https://www.redhat.com/en/technologies/cloud-computing/openshift) cluster. We plan to release the tokenized and formatted parquet data soon for reproducibility.

<p align="center">
<img src="https://github.com/user-attachments/assets/0bc03608-fc3d-4886-b746-9839c52261d5" alt="Datamix" width="600" height="400">
</p>
<p align="center"><strong>Data mix for pretraining phase one</strong></p>


## Pre-Training
Pre-training Bamba was done in a phased manner, we performed ablation experiments at 1.8B model size and a few 100B tokens to determine the right learning rates and built on the previous community efforts. Based on the promising results from this study, we scaled the model to 3B and 2T tokens using Dolma mix. We also trained a 3B transformer model following Meta Llama architecture with the same data mix and observed similar or better performance from the Bamba model. Finally, we scaled the model to 9B size and leveraged PyTorch FSDP to train the model. 

We used a cosine learning rate schedule, with a peak learning rate of `3e−4`, a quadratic warmup over 2000 steps, decay factor of 0.033, and an ending learning rate of `1e−5` over 2T tokens. We use the AdamW optimizer with `β1` of 0.9 and `β2` of 0.95. We use a weight decay of 0.1, sequence length of 4096, and a global batch size of 1.5M tokens/batch. We used 192 A100 GPUs from the [IBM Cloud Vela](https://research.ibm.com/blog/AI-supercomputer-Vela-GPU-cluster) production cluster to train this model over a period of 2 months. This cluster runs on Red Hat Open Shift platform and is built on IBM's full stack (e.g., PyTorch, vLLM). We experienced 3 job interruptions, which were attributed to incorrect deployment of jobs and hardware failures. The hardware related job failures were detected automatically using [autopilot](https://github.com/IBM/autopilot).

We also performed a second phase training with high quality data from Hugging Face FineWeb-edu and Cosmopedia for an additional 200B tokens. We use a learning rate of 2e−5 and a cosine schedule to anneal the model, which helps improve our scores. We are currently experimenting with additional high quality data and will release any future checkpoints as part of our commitment to open source.

### Data loader
There are several aspects to train a high quality language model, one of them being a data loader. We have been working over the past 18 months on creating a dataloader that satisfies the demands of a large scale distributed training and trading that off with model quality. We open source this data loader to enable others to use it in conjunction with their framework of choice, we have used it in the Bamba model training as well as integrated with Torch Titan. To date, we believe this is the only open source dataloader that provides a rich set of features.

The data loader provides the following key features:
1. **Stateful and checkpointable** to ensure seamless resumption mid-epoch
2. **Auto-rescales** to changing workload and GPU allocations
3. Data **streaming** with **zero-overhead** for **shuffling**
4. **Asynchronous distributed** operation with no peer-to-peer communication
5. Allows for **dynamic data mixing** and on-the-fly tokenization
6. PyTorch native, **modular**, and **extensible**

We have battle tested this data loader over hundreds of training jobs and optimized it over months of continuous operation. While theoretical random shuffle would require global visibility, we empirically demonstrate that using a distributed stateless shuffle is sufficient (this lets us scale to arbitrary sizes without imposing any communications overhead at time of training). We use the [Linear Congruential Generator (LCG)](https://en.wikipedia.org/wiki/Linear_congruential_generator) to generate random walks on the input data, which eliminates state maintenance while providing us the property of visiting a token exactly once during training time (in a single epoch). Data mixes are spread uniformly across the GPUs to ensure that each GPU "sees" an exact representation of the mix chosen at training launch. We compare an ideal random shuffle over 1.5K data points with that generated by the stateless shuffle using LCG in the below figure and observe that there are no significant differences.


| ![Figure 1](https://github.com/user-attachments/assets/33c1302a-1669-4d59-9072-98b058c7914d "Figure 1") | ![Figure 2](https://github.com/user-attachments/assets/d3c35613-c877-4db8-ab56-060e32243bd1 "Figure 2") |
|:--:|:--:|
| **Figure 1:** Ideal random shuffle. | **Figure 2:** Distributed stateless shuffle. |

We have recently added support to consume datasets directly from Hugging Face as long as they are indexed. The primary code base is located in our repo [here](https://github.com/foundation-model-stack/fms-fsdp/blob/main/fms_fsdp/utils/dataset_utils.py) and we have also worked with Torch Titan team to make it available [here](https://github.com/pytorch/torchtitan/pull/376). We are working with the Meta PyTorch team to contribute this data loader into core PyTorch.

## Quantization


### Future work
There are several directions that we intend to explore and further these inference efficient architectures:
1. Faster 

## Artifacts

## Contributors

* **Data collection and curation**: We acknowledge and thank AllenAI team for making a high quality open source dataset Dolma as well as Hugging Face data team for making FineWeb-edu and Cosmopedia available. These are tremendous contributions and enable us to create the model today.
* **Data preprocessing**: We thank IBM's internal data preprocessing team for helping tokenize the data at scale. The code for tokenization is available [here](https://github.com/IBM/data-prep-kit)
* **Model architecture**: 

