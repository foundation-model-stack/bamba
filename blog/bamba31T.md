# Bamba9B - Fast and powerful!

During Christmas of 2024, IBM, Princeton, CMU, and UIUC [released](https://huggingface.co/blog/bamba) a performant Mamba2 based pretrained model with full data lineage trained to 2T tokens. Since then, we have been busy cooking an update with new datasets. Today, we are excited to release Bamba v2 trained for an additional 1T tokens that significantly improves on the previous checkpoint. The L1 and L2 leaderboard scores outperform Llama 3.1 8B, which was trained with nearly 5x the amount of data. All of this with the inference speedup that we get from Mamba2 based architecture, which with the latest vLLM is 2-2.5x faster than similar sized transformer models.

## Artifacts 📦

## Fast and Powerful ⚡🏎️ 
We rerun all the benchmarks following the setup and scripts [here]() similar to our previous release. We run the benchmarks for various popular 

While the HF leaderboards themselves are [not active](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/1135) anymore since models evolve, comparing the key benchmarks is an important measure of model capabilities. We provide these comparisons below for various benchmarks. We observe that compared to other SoTA models that are trained to at least 10T+ tokens (and in many cases 15T+), Bamba 9B v2 outperforms the popular Llama 3.1 8B base model on both L1 and L2 averages.

HF OpenLLM v1 benchmarks \+ OpenbookQA, Boolq, and PIQA:

| Model | Average | MMLU | ARC-C | GSM8K | Hellaswag | OpenbookQA | Piqa | TruthfulQA | Winogrande | Boolq |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| [Bamba 9B v2](https://huggingface.co/ibm-fms/Bamba-9B) | 62.39 | 67.89 | 63.65 | 40.11 | **83.7** | **50.2** | 83.13 | 51.67 | 79.48 | 82.81 |
| [Nemotron-H 8B]() | XX | XX | XX | XX | XX | XX | XX | XX | XX | XX | XX |
| [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 60.79 | 66.26 | 57.85 | 49.96 | 81.98 | 46.8 | 82.54 | 45.16 | 77.51 | 82.66 |
| [Olmo2 7B](https://huggingface.co/allenai/OLMo-2-1124-7B) | 63.99 | 63.96 | 64.51 | 68.01 | 81.93 | 49.2 | 81.39 | 43.32 | 770.3 | 84.77 |
| [IBM Granite v3 8B](https://huggingface.co/ibm-granite/granite-3.0-8b-base) | 64.25 | 64.13 | 63.74 | 60.2 | 83.34 | 47.2 | 83.08 | 51.35 | 79.79 | 87.22 |
| [Gemma2 9B](https://huggingface.co/google/gemma-2-9b) | 66.26 | 72.29 | **68.26** | 67.4 | 82.56 | 47.8 | **83.24** | 45.39 | **80.11** | 86.45 |
| [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B) | **69.05** | **75.41** | 63.82 | **83.24** | 80.23 | 48.4 | 81.28 | **56.34** | 75.93 | **87.74** |

HF OpenLLM v2 benchmarks:

| Model | Average | MMLU-PRO | BBH | GPQA | IFEval | MATH Lvl 5 | MuSR |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| [Bamba 9B](https://huggingface.co/ibm-fms/Bamba-9B) | 14.86 | 25.07 | 23.51 | 8.17 | 18.76 | 6.57 | 7.09 |
| [Nemotron-H 8B]() | 14.86 | 25.07 | 23.51 | 8.17 | 18.76 | 6.57 | 7.09 |
| [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 14.45 | 25.46 | 25.16 | 8.61 | 12.55 | 6.19 | 8.72 |
| [Olmo2 7B](https://huggingface.co/allenai/OLMo-2-1124-7B) | 13.4 | 22.79 | 21.69 | 4.92 | 16.35 | 4.61 | 10.02 |
| [IBM Granite v3 8B](https://huggingface.co/ibm-granite/granite-3.0-8b-base) | 19.89 | 24.8 | 25.78 | 9.06 | **41.97** | 9.44 | 8.26 |
| [Gemma2 9B](https://huggingface.co/google/gemma-2-9b) | 21.79 | 34.84 | 34.81 | **11.07** | 21.28 | 13.44 | **15.3** |
| [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B) | **25.83** | **37.62** | **35.62** | 9.96 | 34.77 | **22.43** | 14.6 |

## Training recipe 📖🍳 
Given the limited GPU budget (192 A100s), we did not have the option of training the model for 10T+ tokens. Instead, we decided to explore infusing the existing model with newer data as well as experimenting with techniques like model merging, which was highlighted in the [Olmo2 model training](https://allenai.org/blog/olmo2). Our training recipe is outlined in the below diagram.

We took the 2T base checkpoint (aka Bamba 9b v1) and extended it by adding [Olmo Mix](), released by AllenAI as part of Olmo2 training recipe. We use a constant learning rate schedule to go from 2T to 2.5T tokens, specifically 2e-5. The precise data mix is in the below table.

We then used a mix of synthetic data from Nemotron-CC and Hugging Face datasets to continue training on 500B additional tokens, putting us at 3T tokens. During this phase, we launch two jobs, one with constant learning rate at 2e-5 and another with cosine learning rate going from 2e-5 ending at 2e-6. In our experiments, we observe that using these learning rate schedules improves different benchmarks. Our general observation is that cosine improves memorization benchmarks and constant improves knowledge.

Finally, we anneal both these models using very high quality data for 100B additional tokens and merge the final annealed models using [MergeKit](https://github.com/arcee-ai/mergekit). Specifically, we observe that simple weighted averaging works best for us. The resulting model is Bamba 9B v2!

A note on instruction tuning: We have experimented with [Tuluv3 data]() for creating an instruction following model leveraging [Open Instruct](). We observe that the 

## vLLM integration 📥🧠📤 
We are deeply engaged with the vLLM community on adding support for Mamba2 attention in a generic manner (th)

## Call for Action 📢👉🚀
We are committed to keeping open datasets with complete reproduction of our results. We call on the community to help improve the model on multiple fronts:
1. Test time scaling and GRPO on the model
2. Improve inference performance in vLLM (we expect 4-5x better than corresponding transformer models)
3. Help us improve MATH skills!