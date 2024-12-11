from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "ibm-fms/Bamba-9B-fp8"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
