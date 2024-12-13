# Bamba

<p align="center">
  <img src="/bamba.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/ibm-fms/bamba-674f1388b9bbc98b413c7bab"> Bamba on Hugging Face</a>&nbsp | <a href="https://github.com/foundation-model-stack/bamba/blob/main/blog/bamba-9b-release.md"> Bamba Blog</a>&nbsp
<br>

Bamba is a repository for training and using [Bamba](https://huggingface.co/ibm-fms/Avengers-Mamba2-9B) models which are based on [Mamba](https://github.com/state-spaces/mamba) models.


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

### Overview
TODO: add model card here

### Checkpoints
We have published our model checkpoints here: TODO: add mamba HF page once public


## Inference
You can utilize our newly contributed HF integration to run inference on our Bamba models:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-fms/Avengers-Mamba2-9B-hf")
tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Avengers-Mamba2-9B-hf")

message = ["TODO: find a prompt here"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

```


## Training

We trained our Bamba model with FSDP using our training repo [here](https://github.com/foundation-model-stack/fms-fsdp/tree/mamba-new).
Note that this training effort was started before FSDP2 and also long before we contributed
`Mamba2-Hybrid` to HF, so we were doing FSDP1 training with [official Mamba implementation](https://github.com/state-spaces/mamba).
For users trying to reproduce the training you now have much more options with our newly
contributed [HF-version of Mamba2-Hybrid]() (TODO: add link once live).


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
