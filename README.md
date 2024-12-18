# Bamba

<p align="center">
  <img src="/bamba.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/ibm-fms/bamba-674f1388b9bbc98b413c7bab"> Bamba on Hugging Face</a>&nbsp | <a href="https://github.com/foundation-model-stack/bamba/blob/main/blog/bamba-9b-release.md"> Bamba Blog</a>&nbsp
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

## Models

| Model            | Params       | # Layers | Hidden Dim. | Attention Heads | GQA | KV Heads | Context Length |  Tied Embeddings |
|-------------------|--------------|----------|-------------|-----------------|-----|----------|----------------|------------------|
| Bamba  | 9B (9.78B)   | 32       | 4096        | 32              | Yes | 8        | 4096           | False |

### Checkpoints
We have published our model checkpoints here: [Bamba Models](https://huggingface.co/collections/ibm-fms/bamba-674f1388b9bbc98b413c7bab)


## Inference
You can utilize our newly contributed HF integration to run inference on our Bamba models:
```python
python text_generation.py --model_path ibm-fms/Bamba-9B --tokenizer_path ibm-fms/Bamba-9B --prompt "The largest living mammal on Earth is " --max_new_tokens 128
```

## Training

Details on training can be found [here](https://github.com/foundation-model-stack/bamba/blob/8eaf524806020a6740fcbd107d610a613d3a2955/training/training.md).

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
   <td>MMLU
   </td>
   <td>60.77
   </td>
  </tr>
  <tr>
   <td>ARC-C
   </td>
   <td>63.23
   </td>
  </tr>
  <tr>
   <td>GSM8K
   </td>
   <td>36.77
   </td>
  </tr>
  <tr>
   <td>Hellaswag
   </td>
   <td>81.8
   </td>
  </tr>
  <tr>
   <td>OpenbookQA
   </td>
   <td>47.6
   </td>
  </tr>
  <tr>
   <td>Piqa
   </td>
   <td>82.26
   </td>
  </tr>
  <tr>
   <td>TruthfulQA
   </td>
   <td>49.21
   </td>
  </tr>
  <tr>
   <td>Winogrande
   </td>
   <td>76.87
   </td>
  </tr>
  <tr>
   <td rowspan="6" >HF LLM- V2
   </td>
   <td>MMLU-PRO	
   </td>
   <td>17.53
   </td>
  </tr>
  <tr>
   <td>BBH
   </td>
   <td>17.4
   </td>
  </tr>
  <tr>
   <td>GPQA
   </td>
   <td>4.14
   </td>
  </tr>
  <tr>
   <td>IFEval
   </td>
   <td>15.16
   </td>
  </tr>
  <tr>
   <td>MATH Lvl 5	
   </td>
   <td>1.66
   </td>
  </tr>
  <tr>
   <td>MuSR
   </td>
   <td>9.59
   </td>
  </tr>
  <tr>
   <td rowspan="3" >Safety Tasks
   </td>
   <td>PopQA (5-shot, generation)	
   </td>
   <td>20.5
   </td>
  </tr>
  <tr>
   <td>Toxigen (5-shot, logits)	
   </td>
   <td>57.4
   </td>
  </tr>
  <tr>
   <td>BBQ (5-shot, generation)
   </td>
   <td>44.2
   </td>
  </tr>
</table>


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
||original|quantized |
|:----:|----:|----:|
|memory (total)|39.12 GB|10.83 GB| 
|memory (break-down)|`torch.float32` 39.12 GB|`torch.bfloat16` 2.10 GB<br>`torch.float8_e4m3fn`    8.73 GB|

More details about `fms-model-optimizer` can be found [here](https://github.com/foundation-model-stack/fms-model-optimizer/tree/main/examples/FP8_QUANT#quickstart).

## Evaluation


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
