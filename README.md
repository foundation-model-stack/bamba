# Bamba

<p align="center">
  <img src="/bamba.jpeg" width="400"/>
</p>


<p align="center">
        🤗 <a href="https://huggingface.co/collections/ibm-fms/bamba-674f1388b9bbc98b413c7bab"> Bamba on Hugging Face</a>&nbsp | <a href="https://github.com/foundation-model-stack/bamba/blob/main/blog/bamba-9b-release.md"> Bamba Blog</a>&nbsp | <a href="https://huggingface.co/blog/bamba">
<be>


<!--Bamba is a repository for training and using [Bamba](https://huggingface.co/ibm-fms/Avengers-Mamba2-9B) models, which are derived from [Mamba](https://github.com/state-spaces/mamba) models.--> 

Bamba-9B is a decoder-only language model based on the [Mamba-2](https://github.com/state-spaces/mamba) architecture and is designed to handle a wide range of text generation tasks. It is trained from scratch using a two-stage training approach. In the first stage, the model is trained on 2 trillion tokens from the Dolma v1.7 dataset. In the second stage, it undergoes additional training on 200 billion tokens, leveraging a carefully curated blend of high-quality data to further refine its performance and enhance output quality.

## Installation

Besides [PyTorch](https://pytorch.org/), you would need a few [extra dependencies](https://github.com/state-spaces/mamba?tab=readme-ov-file#installation) for
Mamba models.

We found some of these dependencies picky on PyTorch versions when doing pip install, so 
the best way is to build from source for all Mamba dependencies if you hit dependency 
issue with your env:

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && pip install . && cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install . && cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && pip install . && cd ..
```

For users using our HF versions of the model, you would need to install the latest transformers which includes our newly merged implementation for our Bamba models:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

## Models

| Model | Params     | # Layers | Hidden Dim. | Attention Heads | GQA  | KV Heads | Context Length | Tied Embeddings |
| ----- | ---------- | -------- | ----------- | --------------- | ---- | -------- | -------------- | --------------- |
| Bamba | 9B (9.78B) | 32       | 4096        | 32              | Yes  | 8        | 4096           | False           |

### Checkpoints
You can find links to our model checkpoints here: [Bamba Models](https://huggingface.co/collections/ibm-fms/bamba-674f1388b9bbc98b413c7bab)

## Inference

You can use the following command to perform text generation using one of our checkpoints provided above:

```python
python text_generation.py --model_path ibm-fms/Bamba-9B --tokenizer_path ibm-fms/Bamba-9B --prompt "The largest living mammal on Earth is " --max_new_tokens 128
```

## Training

Details on training can be found [here](./training/training.md).

<!---
For exact reproduction of Bamba 9.8B using the same training data, access is available TODO:[here](Add link to dataloader readme). All fields listed there can be added as optional arguments to the training command (e.g. `--eos_token=128000`).
--->

## Benchmark scores

### Base pretrained models

<table>
<tr>
<td><strong>Category</strong>
</td>
<td><strong>Benchmark</strong>
</td>
<td><strong>Bamba 9B (2.2T)</strong>
</td>
</tr>
<tr>
<td rowspan="8" >General
</td>
<td>MMLU (5-shot)
</td>
<td>60.77
</td>
</tr>
<tr>
<td>ARC-C (25-shot)
</td>
<td>63.23
</td>
</tr>
<tr>
<td>GSM8K (5-shot)
</td>
<td>36.77
</td>
</tr>
<tr>
<td>Hellaswag (10-shot)
</td>
<td>81.8
</td>
</tr>
<tr>
<td>OpenbookQA (5-shot)
</td>
<td>47.6
</td>
</tr>
<tr>
<td>Piqa (5-shot)
</td>
<td>82.26
</td>
</tr>
<tr>
<td>TruthfulQA (0-shot)
</td>
<td>49.21
</td>
</tr>
<tr>
<td>Winogrande (5-shot)
</td>
<td>76.87
</td>
</tr>
<tr>
<td rowspan="6" >HF OpenLLM- V2*
</td>
<td>MMLU-PRO (5-shot)
</td>
<td>17.53
</td>
</tr>
<tr>
<td>BBH (3-shot)
</td>
<td>17.4
</td>
</tr>
<tr>
<td>GPQA (0-shot)
</td>
<td>4.14
</td>
</tr>
<tr>
<td>IFEval (0-shot)
</td>
<td>15.16
</td>
</tr>
<tr>
<td>MATH Lvl 5 (4-shot)
</td>
<td>1.66
</td>
</tr>
<tr>
<td>MuSR (0-shot)
</td>
<td>9.59
</td>
</tr>
<tr>
<td rowspan="4" >Safety Tasks
</td>
<td>PopQA (5-shot)
</td>
<td>20.5
</td>
</tr>
<tr>
<td>Toxigen (5-shot)
</td>
<td>57.4
</td>
</tr>
<tr>
<td>BBQ (5-shot)
</td>
<td>44.2
</td>
</tr>
<tr>
<td>Crows-pairs english (5-shot)
</td>
<td>70.78
</td>
</tr>
</table>

*For the v2 leaderboard results, we perform [normalization](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization) and report the normalized results.

Further details on our evaluation and normalization detailes along with run and analysis scripts can be found [here](https://github.com/foundation-model-stack/bamba/blob/main/evaluation/README.md).

## Fine-tuning

This [example](./tuning/Fine-tuning.md) shows how to fine tune the bamba model for a specific task using [SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer#supervised-fine-tuning-trainer).


## Quantization

We can create a (FP8) quantized model using [`fms-model-optimizer`](https://github.com/foundation-model-stack/fms-model-optimizer/), which will make the storage and inference even more efficient.

```python
python -m fms_mo.run_quant \
    --model_name_or_path <"path_to_original_model"> \
    --quant_method fp8 \
    --torch_dtype bfloat16 \
    --output_dir <"path_to_save_new_model">
```

Model size comparison before and after FP8:

|                     |                 original |                                                    quantized |
| :-----------------: | -----------------------: | -----------------------------------------------------------: |
|   memory (total)    |                 39.12 GB |                                                     10.83 GB |
| memory (break-down) | `torch.float32` 39.12 GB | `torch.bfloat16` 2.10 GB<br>`torch.float8_e4m3fn`    8.73 GB |

More details about `fms-model-optimizer` can be found [here](https://github.com/foundation-model-stack/fms-model-optimizer/tree/main/examples/FP8_QUANT#quickstart).


## Llama.cpp

There is preliminary work to enable running Bamba architecture models using [llama.cpp](https://github.com/ggerganov/llama.cpp). This is work-in-progress, so should only be used as a guide for the adventurous!

### Known Limitations

* Currently, inference is only supported on CPUs
* Models quantized with `llama-quantize` exhibit bad performance

### Setup

To enable Bamba support, you'll need to build from source using [Gabe's fork](https://github.com/gabe-l-hart/llama.cpp/tree/BambaArchitecture).

```sh
git clone --branch BambaArchitecture git@github.com:gabe-l-hart/llama.cpp.git
cd llama.cpp
mkdir build
cd build
# NOTE: To build with debug symbols and extra logging, use CMAKE_BUILD_TYPE=Debug
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### Conversion to GGUF

You can use a pre-converted GGUF file from Huggingface (e.g. [bamba-9b.gguf](https://huggingface.co/ibm-fms/Bamba-9B/blob/main/bamba-9b.gguf)). If one doesn't exist, you can use the [convert_hf_to_gguf.py](https://github.com/gabe-l-hart/llama.cpp/blob/BambaArchitecture/convert_hf_to_gguf.py) script from Gabe's fork to perform the conversion manually.

```sh
# Install the python dependencies
cd /path/to/llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Perform the conversion
./convert_hf_to_gguf.py /path/to/bamba-model --outfile /path/to/bamba-model/bamba-model.gguf
```

### Run with llama-cli

```sh
# Run the model with no layers on the GPU (CPU-only)
cd /path/to/llama.cpp
./bin/llama-cli  -ngl 0 -m /path/to/bamba-model/bamba-model.gguf -p "Tell me a story about a developer and their dog"
```

### Quantization with llama-quantize

You can (optionally) quantize the GGUF model using `llama.cpp`'s built in quantizaiton tool `llama-quantize`.

```sh
# Run the quantization (see llama-quantize --help for all quant types)
cd /path/to/llama.cpp
./build/bin/llama-quantize /path/to/bamba-model/bamba-model.gguf Q4_K_M
```

## Contributors

* **Data collection and curation**: We acknowledge and thank AllenAI team for making a high quality open source dataset Dolma as well as Hugging Face data team for making FineWeb-edu and Cosmopedia available. These are tremendous contributions which enabled us to create the model.
* **Data preprocessing**: We thank IBM's internal data preprocessing team, specifically Tuan Hoang Trong, Syed Zawad, Jay Gala, and Ryan Gordon for helping tokenize the data at scale. The code for tokenization is available [here](https://github.com/IBM/data-prep-kit).  
* **Model architecture**: The model architecture design was jointly done by Princeton, CMU, IBM, and UIUC and involved the following folks: Tri Dao (Princeton), Albert Gu (CMU), Linsong Chu (IBM), Davis Wertheimer (IBM), Minjia Zhang (UIUC), Mudhakar Srivatsa (IBM), and Raghu Ganti (IBM).  
* **Model training**: Model training was performed primarily by the IBM team using the Mamba2 kernels and layer implementation from Tri Dao and Albert Gu. The following folks from IBM were primarily involved: Linsong Chu, Divya Kumari, Davis Wertheimer, Raghu Ganti, and Dakshi Agrawal.  
* **Model tuning**: Tuning of the model was enabled and verified in [TRL](https://github.com/huggingface/trl) by the IBM team, involving Sukriti Sharma and Anh Uong.  
* **Model inference**: Model inference in `transformers`, `vLLM`, and `llama.cpp` builds on the kernels written by Princeton and CMU. The IBM team is working with the community to enable it in various ecosystems. The team includes Fabian Lim, Antoni viros i Martin, Adnan Hoque, Jamie Yang, Nelson Nimura Gonzalez, Joshua Rosenkranz, Nick Hill, and Gabe Goodhart.  
* **Quantization**: Quantization is led by the IBM team \- Naigang Wang and Charlie Liu.  
* **Evaluations**: Evaluations are led by a team in IBM with long context evaluations being performed by UIUC, involving the following folks: Yotam Perlitz, Ofir Arviv, Michal Shmueli-Scheuer (IBM), Haoechen Shen, and Minjia Zhang (UIUC).

Finally, we would like to thank our leadership for their support in this effort \- Priya Nagpurkar, David Cox, Sriram Raghavan, Aya Soffer, Ruchir Puri, and Mukesh Khare.

We would also like to thank the community, in particular Pablo Montalvo-Leroux, Aritra Roy Gosthipaty, and Vaibhav Srivastav from Hugging Face and Stas Bekman from Contextual AI who provided valuable feedback to this blog and the PRs into transformers. Further, we would like to thank Tyler Michael Smith from Neural Magic, who is shepherding the integration with vLLM.

A huge shoutout to Meta PyTorch, AllenAI, and Hugging Face teams for their contributions to the open initative, PyTorch FSDP allowed us to smoothly train this model and the data from Dolma and Fineweb/Cosmopedia made this model today! 
