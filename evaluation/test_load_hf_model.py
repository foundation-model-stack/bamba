from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)
