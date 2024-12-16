import argparse

from transformers.models.bamba.convert_mamba_ssm_checkpoint import (
    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from S3.")

    parser.add_argument(
        "--input_model_paths",
        nargs="+",
        default=[],
        help="List of model names to evaluate. (default: [])",
    )

    args = parser.parse_args()

    for input_model_path in args.input_model_paths:
        convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
            input_model_path,
            "fp16",
            input_model_path + "-hf",
            save_model="sharded",
            tokenizer_path=input_model_path,
        )
        print(f"Done converting {input_model_path}")
