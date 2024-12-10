# Fine-tuning

The below example shows an example of using CodeAlpaca dataset to fine tune the bamba model. 
We will leverage [SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer#supervised-fine-tuning-trainer) for the same.

## Full parameter fine tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import math

# We are tuning on the CodeAlpaca dataset
dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

# We load the model and the tokenizer
# TODO: change path to bamba model when uploaded
model_path = "/ibm-llm-input/flim/Avengers-Jamba-9B-HF"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# We need to add the pad token for training with SFT Trainer
special_tokens_dict = {}
if tokenizer.pad_token is None:
    print("PAD token set to <PAD>, missing in tokenizer")
    special_tokens_dict["pad_token"] = "<PAD>"

# Since we added a new token, we need to resize embeddings of the model.
def tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> dict:
    """Resize tokenizer and embedding.
    Args:
        special_tokens_dict: Dict containing special tokens to be added.
        tokenizer: transformers.PreTrainedTokenizer.
        model: transformers.PreTrainedModel
    Return:
        dict: Metadata on number of added tokens
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    embedding_size = math.ceil(len(tokenizer))
    num_new_tokens = num_new_tokens + embedding_size - len(tokenizer)
    model.resize_token_embeddings(embedding_size)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# Do the resize
tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

# We format the dataset using a prompt
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"

# Set any training arguments
train_args = SFTConfig(per_device_train_batch_size=4,
                       output_dir="/tmp", 
                       gradient_checkpointing=True)

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=train_args, 
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=4096,
    
)
# Start the training
trainer.train()
```
## LoRA Tuning 

To tune only [LoRA adapters](https://arxiv.org/abs/2106.09685), we can additionally specify a [LoRAConfig](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig) to SFT Trainer.

```python
# follow example above for full fine-tuning

# add peft config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=train_args,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=4096
)

trainer.train()
```

### Multi-GPU tuning

For maximum efficiency it is recommended to use A100 or H100 GPU(s). [Full parameter fine tuning](#full-parameter-fine-tuning) is expected to take up more memory than [LoRA tuning](#lora-tuning), and thus may need more than 1 GPU for tuning.

To launch distributed training, you may leverage frameworks such as [accelerate library](https://huggingface.co/docs/accelerate/en/index).

```
accelerate launch --num_processes=2 {myscript.py}
```
`num_processes` should be set to the number of GPUs you want to use. Refer to the [tutorial](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch) for more ways to use accelerate.