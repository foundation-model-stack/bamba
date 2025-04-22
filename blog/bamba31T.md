# Bamba9B - Fast and powerful!

During Christmas of 2024, IBM, Princeton, CMU, and UIUC [released](https://huggingface.co/blog/bamba) a performant Mamba2 based pretrained model with full data lineage trained to 2T tokens. Since then, we have been busy cooking an update with new datasets. Today, we are excited to release a 3.1T checkpoint that significantly improves the 2T checkpoint with L1 and L2 leaderboard scores outperforming Llama 3.1 8B, with nearly 5x the amount of training data. All of this with the inference speedup that we get from Mamba2 based architecture, which with vLLM is 2-2.5x faster than similar sized transformer models. In this article, we will share our updated benchmarks as well as how we got to a performant model.

## Artifacts 

## 
In the last 4 months, we have been cooking to identify better datasets and how to improve the model with purely open datasets.

Diagram of the 

Training recipe
