# Bamba Evaluation

### Installation

initialize and environment, sometimes `conda` makes it easier to handle cuda installations:
```bash
conda create -n bamba python=3.11 -y
conda activate bamba
```

Instal cuda toolkit:
```bash
conda install nvidia/label/cuda-12.2.1::cuda-toolkit
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

And the relevant branches:
```bash
pip install git+https://github.com/fabianlim/vllm.git@pr-draft
pip install flash-attn
pip install "mamba_ssm @ git+https://github.com/state-spaces/mamba.git"
pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@v1.4.0"
pip install git+https://github.com/fabianlim/transformers.git@pr-draft
```

To run the benchmark, install lm-evaluation-harness and unitxt (www.unitxt.ai), along with some other dependencies

```bash
pip install lm_eval
pip install unitxt
pip install langdetect immutabledict antlr4-python3-runtime==4.11 sacrebleu streamlit boto3 matplotlib loguru
```

Link Unitxt to Lm-Eval-Harness (see https://www.unitxt.ai/en/latest/docs/lm_eval.html)
``` 
python -c 'from lm_eval.tasks.unitxt import task; import os.path; print("class: !function " + task.__file__.replace("task.py", "task.Unitxt"))' > ./unitxt_cards_for_lm_eval/unitxt
```

### Running the benchmark

Running the benchmark can be done using `evaluation/runner.py` that has the following signeture:

```
usage: runner.py [-h] [--run_env {LSF,local}] [--benchmarks BENCHMARKS [BENCHMARKS ...]]
                 [--only_subtasks_to_run ONLY_SUBTASKS_TO_RUN [ONLY_SUBTASKS_TO_RUN ...]] [--output_dir_path OUTPUT_DIR_PATH] [--memory MEMORY]
                 [--req_gpu REQ_GPU] [--cores CORES] [--queue QUEUE] [--python_executable PYTHON_EXECUTABLE] [--path_to_lmeval PATH_TO_LMEVAL]
                 [--model_ids MODEL_IDS [MODEL_IDS ...]] [--limit LIMIT] [--batch_size BATCH_SIZE] [--fp_precision FP_PRECISION]
                 [--debug_run_single_task_per_model] [--dry_run]

Run leaderboard evaluation.

options:
  -h, --help            show this help message and exit
  --run_env {LSF,local}
                        Path to the directory where evaluation results and logs will be saved. (default: 'LSF')
  --benchmarks BENCHMARKS [BENCHMARKS ...]
                        List of benchmarks to evaluate on. (default: ['HFV2'])
  --only_subtasks_to_run ONLY_SUBTASKS_TO_RUN [ONLY_SUBTASKS_TO_RUN ...]
                        List of specific subtasks to run within the chosen benchmarks. If empty, runs all subtasks. (default: [])
  --output_dir_path OUTPUT_DIR_PATH
                        Path to the directory where evaluation results and logs will be saved. (default: 'debug')
  --memory MEMORY       Amount of memory to request for the job. (default: '64g')
  --req_gpu REQ_GPU     Type of GPU to request for the job. (default: 'a100_80gb')
  --cores CORES         Number of CPU cores to request. (default: '8+1')
  --queue QUEUE         Name of the queue to submit the job to. (default: 'nonstandard')
  --python_executable PYTHON_EXECUTABLE
                        Path to the Python executable to use. (default: 'python')
  --path_to_lmeval PATH_TO_LMEVAL
                        Path to lm eval harness repo.
  --model_ids MODEL_IDS [MODEL_IDS ...]
                        List of model names to evaluate. (default: [])
  --limit LIMIT         Limit the number of examples to evaluate per task. Useful for debugging. (default: None, meaning no limit)
  --batch_size BATCH_SIZE
                        Batch size to use during evaluation. (default: 4)
  --fp_precision FP_PRECISION
                        Floating-point precision to use (e.g., 16 for fp16, 32 for fp32). (default: 16)
  --debug_run_single_task_per_model
                        If set, runs only one subtask per model for quick debugging. (default: False)
  --dry_run             If set, runs will not be sent
```

The runner is able to parallelize runs on an [LSF cluster](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=wn-whats-new-in-lsf-101-fix-pack-14), however, a `local` option exists to just run things one by one on a local machine. Running all benchmark will probably take around 12h total.

In case you just want to run the benchmark as is and you do not want to use a model with a chat template, run the following:

```bash
#!/bin/bash

# Define paths and variables in one place
harness_path="path/to/lm-evaluation-harness"
python_path="python"
lm_eval_script="${harness_path}/lm_eval"
pretrained_model="ibm-fms/Bamba-9B"
output_base_path="evaluation_results/debug/ibm-fms_Bamba-9B"
batch_size=4

# Function to run lm_eval with common arguments
run_bamba_eval() {
  local tasks="$1"
  local num_fewshot="$2" # Optional, can be empty
  local extra_args="$3" # Optional, for task-specific arguments

  cd "${harness_path}" || exit
  "${python_path}" "${lm_eval_script}" \
    --model hf \
    --model_args "pretrained=${pretrained_model},dtype=float16" \
    --batch_size "${batch_size}" \
    --tasks "${tasks}" \
    --output_path "${output_base_path}" \
    --cache_requests true \
    --log_samples \
    --trust_remote_code \
    --num_fewshot="${num_fewshot}" \
    "${extra_args}"
}

# Run evaluations for each task
run_bamba_eval "mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions" 5
run_bamba_eval "arc_challenge" 25
run_bamba_eval "hellaswag" 10
run_bamba_eval "truthfulqa_mc2" ""
run_bamba_eval "winogrande" 5
run_bamba_eval "gsm8k" 5
run_bamba_eval "leaderboard_mmlu_pro" ""
run_bamba_eval "leaderboard_bbh" ""
run_bamba_eval "leaderboard_gpqa" ""
run_bamba_eval "leaderboard_ifeval" ""
run_bamba_eval "leaderboard_musr" ""
run_bamba_eval "leaderboard_math_hard" ""
run_bamba_eval "boolq" 5
run_bamba_eval "openbookqa" 5
run_bamba_eval "piqa" 5

echo "Evaluations completed."
```

### Analysing the results

Analysing and serving the results comes down to running:

```bash
path/to/streamlit run evaluation/serve_results.py --server.port 8090 -- --res_dirs path_to_runner_output_1 path_to_runner_output_2
```

### Normalization

All dataset normalization is in `evaluation/normalizations.py` and based on the official noramalizations as done in the HF OpenLLLM leaderboard (V2).