import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(model_path, tokenizer_path, prompt, max_new_tokens=100):

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Tokenize the input message
    inputs = tokenizer([prompt], return_tensors='pt', return_token_type_ids=False)
    
    # Generate response
    response = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode and print the result
    decoded_output = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    print("\n", decoded_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load an LLM model and generate text.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")

    args = parser.parse_args()
    
    main(args.model_path, args.tokenizer_path, args.prompt, args.max_new_tokens)
